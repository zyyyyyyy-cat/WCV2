"""Strict checkpoint loading, shared manifests and resumable feature extraction."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temporary), str(path))


def code_hashes(root):
    # Include diagnostic sources as well: changing their semantics invalidates cache reuse.
    folders = ['config', 'datasets', 'model', 'loss', 'processor', 'diagnostics', 'utils']
    paths = sorted([p for folder in folders for p in (Path(root) / folder).rglob('*.py')]
                   + [Path(root) / 'diagnose_wcv1.py'])
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}


def make_config(config_file, data_root, dataset_dir, pretrain=None):
    from config import cfg
    c = cfg.clone()
    c.defrost()
    c.merge_from_file(str(config_file))
    c.MODEL.DIST_TRAIN = False
    c.DATASETS.ROOT_DIR = str(data_root)
    c.DATASETS.DATASET_DIR = dataset_dir
    c.DATASETS.FILTER_GD = False
    c.TEST.NECK_FEAT = 'before'
    c.TEST.RE_RANKING = False
    c.TEST.TOP_K_EVAL = 0
    # A full trained checkpoint is loaded strictly, including the classifier.
    # ImageNet initialization is unnecessary when every trained tensor is present.
    c.MODEL.PRETRAIN_CHOICE = 'none'
    if pretrain:
        c.MODEL.PRETRAIN_PATH = str(pretrain)
    c.freeze()
    return c


def make_records(c):
    import re
    from datasets.whu_mars import WHU_MARS
    dataset = WHU_MARS(root=c.DATASETS.ROOT_DIR, dataset_dir=c.DATASETS.DATASET_DIR,
                       modalities=list(c.DATASETS.MODALITIES), filter_gd=False)
    root = Path(dataset.dataset_dir).resolve()
    records = {}
    for split in ['train', 'query', 'gallery']:
        values = getattr(dataset, split)
        rows = []
        for m, name in enumerate(c.DATASETS.MODALITIES):
            for path, pid, cam, _ in values[name]:
                original_pid = int(re.search(r'(\d+)_c\d+', Path(path).name).group(1))
                rows.append(dict(path=Path(path).resolve().relative_to(root).as_posix(),
                                 pid=int(pid), original_pid=original_pid, camid=int(cam),
                                 modality=m, modality_name=str(name), bytes=Path(path).stat().st_size))
        keys = {key: i for i, key in enumerate(sorted(set((r['modality_name'], r['path']) for r in rows)))}
        for row in rows:
            row['sample_key'] = keys[(row['modality_name'], row['path'])]
        records[split] = rows
    return dataset, records


def load_model(c, dataset, checkpoint, device):
    import torch
    from model import make_model
    model = make_model(c, num_class=dataset.num_train_pids,
                       camera_num=dataset.num_train_cams, view_num=dataset.num_train_vids)
    # Only load trusted user checkpoints. Torch 1.12 has no weights_only argument.
    try:
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location='cpu')
    archive = state if isinstance(state, dict) and 'model' in state else None
    if archive is not None:
        state = archive['model']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise ValueError('checkpoint must contain a complete model state_dict')
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    bank = None
    if archive is not None and archive.get('prototype_bank') is not None:
        bank = archive['prototype_bank'].get('prototypes')
    return model, bank


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def transformations(c, training=False):
    import torchvision.transforms as T
    if training:
        from timm.data.random_erasing import RandomErasing
        return T.Compose([T.Resize(c.INPUT.SIZE_TRAIN, interpolation=3),
                          T.RandomHorizontalFlip(p=c.INPUT.PROB), T.Pad(c.INPUT.PADDING),
                          T.RandomCrop(c.INPUT.SIZE_TRAIN), T.ToTensor(),
                          T.Normalize(mean=c.INPUT.PIXEL_MEAN, std=c.INPUT.PIXEL_STD),
                          RandomErasing(probability=c.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')])
    return T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                      T.Normalize(mean=c.INPUT.PIXEL_MEAN, std=c.INPUT.PIXEL_STD)])


class RecordImages:
    def __init__(self, root, records, transform):
        self.root, self.records, self.transform = Path(root), records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(str(self.root / self.records[index]['path'])) as source:
            return self.transform(source.convert('RGB'))


def extract_model(model, records, root, c, cache_dir, label, fingerprint,
                  device, batch_size=256, workers=4):
    import torch
    from torch.utils.data import DataLoader
    cache_dir = Path(cache_dir)
    model_dir = cache_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    before = state_hash(model)
    model.eval()
    for split, rows in records.items():
        if not rows:
            raise ValueError('empty dataset split: ' + split)
        path = model_dir / (split + '.npy')
        marker = model_dir / (split + '.json')
        signature = dict(fingerprint=fingerprint, records_hash=json_hash(rows), shape=[len(rows), 768])
        if marker.exists() and path.exists():
            info = json.loads(marker.read_text(encoding='utf-8'))
            arr = np.load(str(path), mmap_mode='r')
            content_hash = info.pop('feature_sha256', None)
            if info != signature or list(arr.shape) != signature['shape'] or arr.dtype != np.float32 or content_hash != sha256_file(path):
                raise ValueError('incompatible feature cache; choose a new cache-dir: ' + str(path))
            result[split] = arr
            print('Reuse', label, split, arr.shape, flush=True)
            continue
        # An interrupted .partial file is never mistaken for completed features.
        partial = model_dir / (split + '.partial.npy')
        array = np.lib.format.open_memmap(str(partial), mode='w+', dtype=np.float32, shape=(len(rows), 768))
        loader = DataLoader(RecordImages(root, rows, transformations(c)), batch_size=batch_size,
                            num_workers=workers, shuffle=False, pin_memory=str(device).startswith('cuda'))
        offset = 0
        with torch.no_grad():
            for images in loader:
                # Original standalone inference is FP32, pre-BN CLS, no flip/augmentation.
                output = model(images.to(device), mode=1).float().cpu().numpy()
                if output.shape[1:] != (768,) or not np.isfinite(output).all():
                    raise ValueError('invalid extracted CLS')
                array[offset:offset + len(output)] = output
                offset += len(output)
        if offset != len(rows):
            raise RuntimeError('feature count mismatch')
        array.flush()
        del array
        os.replace(str(partial), str(path))
        save_json(marker, dict(signature, feature_sha256=sha256_file(path)))
        result[split] = np.load(str(path), mmap_mode='r')
        print('Extracted', label, split, result[split].shape, flush=True)
    if state_hash(model) != before:
        raise RuntimeError('feature extraction changed model state')
    return result


def cached_features(cache_dir, records, fingerprints):
    result = {}
    for label in ['uad', 'wcv1']:
        result[label] = {}
        for split, rows in records.items():
            path = Path(cache_dir) / label / (split + '.npy')
            info = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            expected = dict(fingerprint=fingerprints[label], records_hash=json_hash(rows), shape=[len(rows), 768])
            arr = np.load(str(path), mmap_mode='r')
            content_hash = info.pop('feature_sha256', None)
            if info != expected or list(arr.shape) != expected['shape'] or arr.dtype != np.float32 or content_hash != sha256_file(path):
                raise ValueError('invalid or mismatched completed cache: ' + str(path))
            result[label][split] = arr
    return result


def deterministic_reader(c, root, records, seed=1234):
    """Same actual augmented input for both models; leave caller RNG untouched."""
    import torch
    dataset = RecordImages(root, records, transformations(c, training=True))

    def read(indices, replay_seed=None):
        indices = [int(i) for i in indices]
        batch_seed = (int(json_hash(indices)[:8], 16) + (seed if replay_seed is None else replay_seed)) % (2 ** 32)
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(batch_seed)
            np.random.seed(batch_seed)
            # CPU augmentation only: torch.manual_seed would also alter CUDA RNG.
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(batch_seed).get_state())
            return torch.stack([dataset[i] for i in indices])
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
    return read
