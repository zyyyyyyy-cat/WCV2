from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import torch.distributed as dist
import numpy as np
import os
import argparse
import hashlib
import json
import sys
from pathlib import Path
# from timm.scheduler import create_scheduler
from config import cfg


def write_sam_manifest(cfg, train_loader, val_loaders, num_querys, num_classes, world_size):
    """Record actual source/pretrain/index provenance without reading images."""
    def digest_file(path):
        result = hashlib.sha256()
        with open(str(path), 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                result.update(chunk)
        return result.hexdigest()

    def index_info(rows):
        digest = hashlib.sha256()
        for path, pid, camera, modality in rows:
            relative = os.path.relpath(path, cfg.DATASETS.ROOT_DIR).replace('\\', '/')
            digest.update((json.dumps([relative, int(pid), int(camera), int(modality)],
                                     ensure_ascii=False) + '\n').encode('utf-8'))
        return dict(count=len(rows), ordered_index_sha256=digest.hexdigest())

    root = Path(__file__).resolve().parent
    source = {p.relative_to(root).as_posix(): digest_file(p)
              for p in sorted(root.rglob('*.py'))
              if '__pycache__' not in p.parts and 'tmp' not in p.parts}
    indexes = {}
    for modality in cfg.DATASETS.MODALITIES:
        indexes['train_' + modality] = index_info(train_loader.dataset.dataset[modality])
    for loader, count in zip(val_loaders, num_querys):
        modality = loader.dataset.modality
        rows = loader.dataset.dataset[modality]
        indexes['query_' + modality] = index_info(rows[:count])
        indexes['gallery_' + modality] = index_info(rows[count:])
    manifest = dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                    world_size=world_size, num_classes=num_classes, config=cfg.dump(),
                    pretrain_sha256=digest_file(cfg.MODEL.PRETRAIN_PATH),
                    source_sha256=source, data_indexes=indexes,
                    note='Index hashes cover ordered path/labels, not image file contents.')
    path = Path(cfg.OUTPUT_DIR or '.') / 'run_manifest.json'
    with path.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return str(path)


def setup_cuda_visible_devices(cfg):
    env_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    alias_devices = os.environ.get('CUDA_VISIBLE_DEVICE')
    if env_devices:
        return
    if alias_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = alias_devices
    elif cfg.MODEL.DEVICE_ID:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


def resolve_local_rank(args):
    env_local_rank = os.environ.get('LOCAL_RANK')
    if env_local_rank is not None:
        return int(env_local_rank)
    if args.local_rank is not None and args.local_rank >= 0:
        return int(args.local_rank)
    return 0


def init_distributed(cfg, args):
    """Initialize one NCCL process per GPU when launched by torchrun."""
    local_rank = resolve_local_rank(args)
    if not cfg.MODEL.DIST_TRAIN:
        return local_rank, 0, 1

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP training must be launched with torchrun; missing environment "
            "variables: {}".format(", ".join(missing))
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs and the NCCL backend")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            "LOCAL_RANK={} but only {} CUDA devices are visible".format(
                local_rank, torch.cuda.device_count()
            )
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--local-rank', type=int, dest='local_rank')
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.SAM.ENABLED or cfg.SAM.SAFE_BASELINE:
        from processor.sam_training import validate_sam_config
        validate_sam_config(cfg)

    setup_cuda_visible_devices(cfg)
    args.local_rank, rank, world_size = init_distributed(cfg, args)

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if rank == 0 and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if cfg.MODEL.DIST_TRAIN:
        dist.barrier()

    logger = setup_logger(
        "transreid", output_dir, if_train=True, distributed_rank=rank
    )
    if rank == 0:
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)
        if cfg.MODEL.DIST_TRAIN:
            logger.info(
                "DDP initialized with {} processes; global sampler batch={}, "
                "per-GPU sampler batch={}, actual images per GPU={}".format(
                    world_size,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.SOLVER.IMS_PER_BATCH // world_size,
                    (cfg.SOLVER.IMS_PER_BATCH // world_size)
                    * len(cfg.DATASETS.MODALITIES),
                )
            )

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    if rank == 0:
        logger.info("Running with config:\n{}".format(cfg))

    train_loader, train_loader_normal, val_loaders, num_querys, num_classes, camera_num, view_num = make_dataloader(cfg)

    if rank == 0 and (cfg.SAM.ENABLED or cfg.SAM.SAFE_BASELINE):
        logger.info('Run provenance saved: %s', write_sam_manifest(
            cfg, train_loader, val_loaders, num_querys, num_classes, world_size))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loaders,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_querys, args.local_rank
    )

    if cfg.MODEL.DIST_TRAIN and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
