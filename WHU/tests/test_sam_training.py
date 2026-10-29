"""Small, data-free UAD-SAM-B checks; no WHU training is launched.

CPU: python tests/test_sam_training.py
CUDA AMP: python tests/test_sam_training.py --cuda
DDP: python tests/test_sam_training.py --backend nccl --world-size 4
The DDP command spawns its own processes; do not wrap it in torchrun.
"""
import argparse
import copy
import os
import random
import socket
import tempfile
import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processor.processor import PCA, GPD, PrototypeBank, uad_forward_loss
from processor.sam_training import run_uad_step, sam_radius, validate_sam_config
from loss.wrt_loss import TripletLoss_WRT
from config import cfg as default_cfg


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 8)
        self.drop = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 3)  # retained but unused, like the real backbone

    def forward(self, x):
        return self.norm(self.drop(self.proj(x)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = Backbone()
        self.bottleneck = nn.BatchNorm1d(8)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(8, 2, bias=False)

    def forward(self, imgs, labels=None, cams=None):
        raw = self.base(torch.cat(imgs))
        h = self.bottleneck(raw)
        return self.classifier(h), raw, h


def objective(scores, raw, labels):
    ce = nn.functional.cross_entropy(scores, labels)
    wrt = TripletLoss_WRT()(raw, labels)[0]
    return ce + wrt, ce, wrt


def inputs(device='cpu', offset=0):
    generator = torch.Generator().manual_seed(5 + offset)
    imgs = [torch.randn(8, 3, generator=generator).to(device) for _ in range(3)]
    labels = torch.tensor([0] * 4 + [1] * 4, device=device)
    return imgs, labels, [torch.arange(8, device=device) for _ in range(3)]


def forward_for(model, bank, batch):
    return lambda fixed_targets=None: uad_forward_loss(
        model, *batch, objective, bank, fixed_targets=fixed_targets)


def optimizer_for(model):
    return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)


class RecordingScaler:
    """Public AMP protocol simulator; exercises skip paths without a GPU."""
    def __init__(self):
        self.value, self.tracker = 8., 3
        self.unscales, self.steps, self.updates = 0, 0, 0
        self.bad = False

    def is_enabled(self): return True
    def get_scale(self): return self.value
    def scale(self, loss): return loss * self.value

    def unscale_(self, optimizer):
        self.unscales += 1
        self.bad = False
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.div_(self.value)
                    self.bad |= not bool(torch.isfinite(p.grad).all())

    def step(self, optimizer):
        self.steps += 1
        if not self.bad: optimizer.step()

    def update(self):
        self.updates += 1
        if self.bad:
            self.value *= .5
            self.tracker = 0
        else: self.tracker += 1


class SamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model = Toy().train()
        self.bank = PrototypeBank(2, 8, device='cpu')
        self.batch = inputs()

    def assert_state_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)

    def test_radius_and_config(self):
        cfg = default_cfg.clone()
        cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/UAD_SAM_B.yml'))
        validate_sam_config(cfg)
        self.assertEqual([sam_radius(e, cfg.SAM) for e in (1, 5, 6, 10, 120)], [0., 0., .004, .02, .02])
        cfg.WCV1.ENABLED = True
        with self.assertRaises(ValueError): validate_sam_config(cfg)

    def test_zero_radius_matches_original_finite_step(self):
        reference = copy.deepcopy(self.model)
        reference_bank = copy.deepcopy(self.bank)
        opt1, opt2 = optimizer_for(reference), optimizer_for(self.model)
        rng = torch.get_rng_state()
        l, t, _ = forward_for(reference, reference_bank, self.batch)()
        l.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        opt1.step()
        reference_bank.update(t['ids'], t['centers'], ddp=False)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        result = run_uad_step(self.model, opt2, GradScaler(enabled=False),
                              forward_for(self.model, self.bank, self.batch), 0.,
                              commit=lambda t: self.bank.update(t['ids'], t['centers'], ddp=False))
        self.assertTrue(result['step_succeeded'])
        self.assertIsNone(result['adv_loss'])
        self.assert_state_equal(reference.state_dict(), self.model.state_dict())
        self.assert_state_equal(reference_bank.state_dict(), self.bank.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_scope_targets_rng_bn_and_single_commit(self):
        original = copy.deepcopy(self.model.state_dict())
        heads = {k: v.clone() for k, v in original.items() if not k.startswith('base.')}
        records, commits = [], []
        callback = forward_for(self.model, self.bank, self.batch)
        def forward(fixed_targets=None):
            if fixed_targets is not None:
                for k in ('classifier.weight', 'bottleneck.weight'):
                    self.assertTrue(torch.equal(self.model.state_dict()[k], heads[k]))
                self.assertTrue(torch.equal(self.model.base.fc.weight, original['base.fc.weight']))
            value = callback(fixed_targets)
            records.append((value[1], torch.get_rng_state().clone(), self.model.bottleneck.num_batches_tracked.item()))
            return value
        result = run_uad_step(self.model, optimizer_for(self.model), GradScaler(enabled=False), forward, .02,
                              commit=lambda t: commits.append(t), record_layer_stats=True)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(commits), 1)
        for key in ('centers', 'prototypes', 'ids'):
            self.assertTrue(torch.equal(records[0][0][key], records[1][0][key]))
            self.assertFalse(records[1][0][key].requires_grad)
        self.assertTrue(torch.equal(records[0][1], records[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), records[0][1]))
        self.assertEqual(self.model.bottleneck.num_batches_tracked.item(), 1)
        self.assertAlmostEqual(result['perturbation_norm'], .02, places=6)
        self.assertGreater(result['perturb_abs_proj'], 0.)
        self.assertGreater(result['perturb_rel_proj'], 0.)
        self.assertNotIn('perturb_abs_fc', result)
        self.assertTrue(result['step_succeeded'])
        self.assertFalse(torch.equal(self.model.classifier.weight, heads['classifier.weight']))
        self.assertTrue(torch.equal(self.model.base.fc.weight, original['base.fc.weight']))

    def test_exact_restoration_before_optimizer(self):
        original = copy.deepcopy(self.model.state_dict())
        opt = optimizer_for(self.model)
        real_step = opt.step
        def checked_step(*args, **kwargs):
            for key, value in self.model.base.state_dict().items():
                self.assertTrue(torch.equal(value, original['base.' + key]))
            return real_step(*args, **kwargs)
        opt.step = checked_step
        run_uad_step(self.model, opt, GradScaler(enabled=False), forward_for(self.model, self.bank, self.batch), .02)

    def test_update_is_perturbed_gradient_not_clean_or_average(self):
        self.model.base.drop.p = 0.
        reference = copy.deepcopy(self.model)
        bank = copy.deepcopy(self.bank)
        original = {k: p.detach().clone() for k, p in reference.named_parameters()}
        fn = forward_for(reference, bank, self.batch)
        loss0, targets, _ = fn()
        loss0.backward()
        scope = [p for p in reference.base.parameters() if p.grad is not None]
        norm = torch.stack([p.grad.float().norm() for p in scope]).double().norm()
        with torch.no_grad():
            for p in scope: p.add_(p.grad * (.02 / (norm + 1e-12)))
        reference.zero_grad(set_to_none=True)
        loss1, _, _ = fn(targets)
        loss1.backward()
        grads = {k: p.grad.clone() for k, p in reference.named_parameters() if p.grad is not None}
        total_norm = torch.stack([g.norm() for g in grads.values()]).norm()
        factor = min(1., 1. / (float(total_norm) + 1e-6))
        # Independent first-step SGD equation (momentum buffer starts empty).
        expected = {k: old - .01 * (factor * grads[k] + 1e-4 * old)
                    if k in grads else old for k, old in original.items()}
        run_uad_step(self.model, optimizer_for(self.model), GradScaler(enabled=False),
                     forward_for(self.model, self.bank, self.batch), .02)
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p, expected[name], rtol=0, atol=1e-7, msg=name)

    def test_perturbed_exception_rolls_back(self):
        original = copy.deepcopy(self.model.state_dict())
        clean_rng = []
        base = forward_for(self.model, self.bank, self.batch)
        def failing(fixed_targets=None):
            result = base(fixed_targets)
            if fixed_targets is not None: raise RuntimeError('injected forward exception')
            clean_rng.append(torch.get_rng_state().clone())
            return result
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            run_uad_step(self.model, optimizer_for(self.model), GradScaler(enabled=False), failing, .02)
        self.assert_state_equal(original, self.model.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), clean_rng[0]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_clean_and_adv_overflow_are_atomic(self):
        for overflow_pass in (1, 2):
            self.setUp()
            original = copy.deepcopy(self.model.state_dict())
            opt, scaler, commits = optimizer_for(self.model), RecordingScaler(), []
            calls = [0]
            def overflow(grad):
                calls[0] += 1
                return torch.full_like(grad, float('inf')) if calls[0] == overflow_pass else grad
            handle = self.model.classifier.weight.register_hook(overflow)
            result = run_uad_step(self.model, opt, scaler, forward_for(self.model, self.bank, self.batch), .02,
                                  commit=lambda t: commits.append(t))
            handle.remove()
            self.assertFalse(result['step_succeeded'])
            self.assertEqual((scaler.unscales, scaler.steps, scaler.updates), (1, 1, 1))
            self.assertEqual(scaler.get_scale(), 4.)
            self.assertEqual(scaler.tracker, 0)
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(commits, [])
            self.assert_state_equal(original, self.model.state_dict())

    def test_nonfinite_forward_is_fatal(self):
        before = copy.deepcopy(self.model.state_dict())
        base = forward_for(self.model, self.bank, self.batch)
        def bad(fixed_targets=None):
            loss, targets, metrics = base(fixed_targets)
            return loss * float('nan'), targets, metrics
        with self.assertRaises(FloatingPointError):
            run_uad_step(self.model, optimizer_for(self.model), GradScaler(enabled=False), bad, .02)
        self.assert_state_equal(before, self.model.state_dict())

    @unittest.skipUnless(hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'),
                         'Real CPU GradScaler is available only on newer local torch')
    def test_real_cpu_scaler_continuous_success_and_overflows(self):
        # Local-only coverage of the real public scaler state machine. The
        # separate --cuda/--backend checks exercise torch1.12 CUDA GradScaler.
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        opt = optimizer_for(self.model)
        commits = []
        for overflow_pass in (None, 1, 2, None):
            before = copy.deepcopy(self.model.state_dict())
            count = [0]
            def hook(grad):
                count[0] += 1
                return torch.full_like(grad, float('inf')) if count[0] == overflow_pass else grad
            handle = self.model.classifier.weight.register_hook(hook)
            old_scale = scaler.get_scale()
            result = run_uad_step(self.model, opt, scaler, forward_for(self.model, self.bank, self.batch),
                                  .02, commit=lambda t: commits.append(t))
            handle.remove()
            self.assertEqual(result['step_succeeded'], overflow_pass is None)
            if overflow_pass is not None:
                self.assertEqual(scaler.get_scale(), old_scale * .5)
                self.assertEqual(scaler.state_dict()['_growth_tracker'], 0)
                self.assert_state_equal(before, self.model.state_dict())
        self.assertEqual(len(commits), 2)

    def test_fixed_loss_targets_are_not_regenerated(self):
        raw = torch.randn(24, 8, requires_grad=True)
        labels = self.batch[1]
        _, ids, centers = PCA(raw, labels, 3)
        _, snapshot = GPD(raw, labels, self.bank, ids, centers, 3, return_prototypes=True)
        fixed = torch.randn_like(centers, requires_grad=True)
        pca, _, reused = PCA(raw + .1, labels, 3, fixed_centers=fixed)
        self.assertTrue(torch.equal(reused, fixed.detach()))
        self.bank.prototypes.fill_(123.)
        gpd = GPD(raw, labels, self.bank, ids, centers, 3, fixed_prototypes=snapshot)
        expected = nn.functional.cross_entropy(nn.functional.normalize(raw, dim=1) @ snapshot.t() / .03, labels.repeat(3))
        torch.testing.assert_close(gpd, expected)
        (pca + gpd).backward()
        self.assertIsNone(fixed.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_run_manifest_records_pretrain_and_index_hashes_without_images(self):
        from types import SimpleNamespace
        from train import write_sam_manifest
        with tempfile.TemporaryDirectory() as directory:
            cfg = default_cfg.clone()
            cfg.DATASETS.ROOT_DIR = directory
            cfg.OUTPUT_DIR = directory
            cfg.MODEL.PRETRAIN_PATH = str(Path(directory) / 'pretrain.bin')
            Path(cfg.MODEL.PRETRAIN_PATH).write_bytes(b'provenance fixture')
            cfg.DATASETS.MODALITIES = ['RGB', 'IR', 'Thermal']
            rows = {m: [(str(Path(directory) / (m + '.png')), 0, 1, i)]
                    for i, m in enumerate(cfg.DATASETS.MODALITIES)}
            loader = SimpleNamespace(dataset=SimpleNamespace(dataset=rows))
            vals = [SimpleNamespace(dataset=SimpleNamespace(dataset=rows, modality=m))
                    for m in cfg.DATASETS.MODALITIES]
            rng = torch.get_rng_state()
            path = write_sam_manifest(cfg, loader, vals, [1, 1, 1], 1, 4)
            manifest = json.loads(Path(path).read_text())
            self.assertEqual(manifest['pretrain_sha256'], hashlib.sha256(b'provenance fixture').hexdigest())
            self.assertEqual(manifest['data_indexes']['query_RGB']['count'], 1)
            self.assertEqual(manifest['data_indexes']['gallery_RGB']['count'], 0)
            self.assertIn('processor/sam_training.py', manifest['source_sha256'])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


def accelerator_worker(rank, world, backend, port, use_amp=True):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:{}'.format(port), rank=rank, world_size=world)
    torch.manual_seed(1234)
    model = Toy().to(device).train()
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if device.type == 'cuda' else None,
                                                    find_unused_parameters=True)
    meta = model.module if world > 1 else model
    bank = PrototypeBank(2, 8, device=device)
    opt, scaler = optimizer_for(model), GradScaler(enabled=use_amp and device.type == 'cuda')
    callback, calls, commits = forward_for(model, bank, inputs(device, rank)), [0], [0]
    def forward(fixed_targets=None):
        calls[0] += 1
        if world > 1 and fixed_targets is not None:
            flat = torch.cat([p.detach().flatten() for p in meta.base.parameters()])
            copies = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(copies, flat)
            assert all(torch.equal(flat, x) for x in copies), 'DDP perturbations differ'
        return callback(fixed_targets)
    def commit(t):
        commits[0] += 1
        bank.update(t['ids'], t['centers'], ddp=world > 1)
    # Start below the default large scale so the smoke test must exercise actual
    # updates, not merely a sequence of valid automatic overflow skips.
    if scaler.is_enabled(): scaler = GradScaler(init_scale=128.)
    for _ in range(3):
        result = run_uad_step(model, opt, scaler, forward, .02, commit=commit, distributed=world > 1)
        assert result['step_succeeded'], result
        assert abs(result['perturbation_norm'] - .02) < 2e-6
    assert commits[0] == 3 and meta.bottleneck.num_batches_tracked.item() == 3
    for target_pass in (1, 2):
        if not scaler.is_enabled(): break
        before = copy.deepcopy(meta.state_dict())
        counter = [0]
        def overflow(g):
            counter[0] += 1
            return torch.full_like(g, float('inf')) if counter[0] == target_pass and rank == world - 1 else g
        handle = meta.classifier.weight.register_hook(overflow)
        old_scale = scaler.get_scale()
        result = run_uad_step(model, opt, scaler, forward, .02, commit=commit, distributed=world > 1)
        handle.remove()
        assert not result['step_succeeded'] and scaler.get_scale() == old_scale * .5
        assert commits[0] == 3
        assert all(torch.equal(before[k], v) for k, v in meta.state_dict().items())
    if world > 1: dist.destroy_process_group()
    if rank == 0: print('Accelerator/DDP SAM smoke checks passed (small synthetic model only).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--backend', choices=['nccl', 'gloo'])
    parser.add_argument('--world-size', type=int, default=4)
    args, rest = parser.parse_known_args()
    if args.backend:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        mp.spawn(accelerator_worker, args=(args.world_size, args.backend, port), nprocs=args.world_size, join=True)
    elif args.cuda:
        if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable')
        accelerator_worker(0, 1, 'nccl', 0)
    else:
        unittest.main(argv=[sys.argv[0]] + rest)
