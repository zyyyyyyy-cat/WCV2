import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, ImageDatasetTest
from timm.data.random_erasing import RandomErasing
from .sampler import PKMSampler
from .whu_mars import WHU_MARS
from .sampler_ddp import PKMSampler_DDP
import torch.distributed as dist

__factory = {
    'WHU-MARS': WHU_MARS,
}


def train_collate_fn(batch):
    has_metadata = len(batch[0]) == 4
    if any(len(sample) != len(batch[0]) for sample in batch):
        raise ValueError("Mixed train sample interfaces in one batch")
    if has_metadata:
        imgs_ls, pids, camids_ls, keys_ls = zip(*batch)
    else:
        imgs_ls, pids, camids_ls = zip(*batch)
    num_modalities = len(imgs_ls[0])
    imgs = [torch.stack([sample[m] for sample in imgs_ls], dim=0) for m in range(num_modalities)]
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = [torch.tensor([sample[m] for sample in camids_ls], dtype=torch.int64) for m in range(num_modalities)]
    if has_metadata:
        keys = [torch.tensor([sample[m] for sample in keys_ls], dtype=torch.int64)
                for m in range(num_modalities)]
        return imgs, pids, camids, keys
    return imgs, pids, camids


def val_collate_fn(batch):
    imgs, pids, camids, modids, img_paths = zip(*batch)
    camidt = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camidt, modids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    modalities = list(cfg.DATASETS.MODALITIES)

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, modalities=modalities,
                                             dataset_dir=cfg.DATASETS.DATASET_DIR,
                                             filter_gd=cfg.DATASETS.FILTER_GD)

    train_set = ImageDataset(dataset.train, modalities=modalities, transform=train_transforms,
                             return_metadata=cfg.WCV1.ENABLED)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER.upper() != 'PKM':
        raise ValueError(
            "Only PKM sampler is supported, but got {}".format(cfg.DATALOADER.SAMPLER)
        )

    if cfg.MODEL.DIST_TRAIN:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DIST_TRAIN=True requires an initialized process group")
        world_size = dist.get_world_size()
        if cfg.SOLVER.IMS_PER_BATCH % world_size != 0:
            raise ValueError(
                "SOLVER.IMS_PER_BATCH ({}) must be divisible by WORLD_SIZE ({})".format(
                    cfg.SOLVER.IMS_PER_BATCH, world_size
                )
            )
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size
        if mini_batch_size % cfg.DATALOADER.NUM_INSTANCE != 0:
            raise ValueError(
                "Per-GPU sampler batch ({}) must be divisible by NUM_INSTANCE ({})".format(
                    mini_batch_size, cfg.DATALOADER.NUM_INSTANCE
                )
            )
        if dist.get_rank() == 0:
            print('DIST_TRAIN START')
        data_sampler = PKMSampler_DDP(
            dataset.train,
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.DATALOADER.NUM_INSTANCE,
            modalities=modalities,
        )
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=PKMSampler(
                dataset.train,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE,
                modalities=modalities,
            ),
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )

    val_loaders = []
    num_querys = []

    for modality in modalities:
        val_data = {modality: dataset.query.get(modality, []) + dataset.gallery.get(modality, [])}
        val_set = ImageDatasetTest(val_data, modality, val_transforms)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )
        val_loaders.append(val_loader)
        num_querys.append(len(dataset.query.get(modality, [])))

    ref_modality = modalities[0]
    train_normal_data = {ref_modality: dataset.train.get(ref_modality, [])}
    train_set_normal = ImageDatasetTest(train_normal_data, ref_modality, val_transforms)
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    return train_loader, train_loader_normal, val_loaders, num_querys, num_classes, cam_num, view_num
