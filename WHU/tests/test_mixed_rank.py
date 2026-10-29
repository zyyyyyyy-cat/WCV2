"""Synthetic loss/DDP checks; no dataset, backbone training or CUDA required.

python tests/test_mixed_rank.py
CUDA_VISIBLE_DEVICES=0,1 python tests/test_mixed_rank.py --backend nccl
"""
import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

spec = importlib.util.spec_from_file_location('mixed_rank', Path(__file__).resolve().parents[1] / 'loss/mixed_rank.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def data(device='cpu'):
    x = torch.tensor([[1., 0., .2], [.8, .2, .3], [0., 1., .1], [.1, .9, .2],
                      [.7, .3, .2], [.2, .8, .1]], device=device)
    return (x, torch.tensor([0, 0, 1, 1, 0, 1], device=device),
            torch.tensor([0, 1, 0, 1, 2, 2], device=device),
            torch.tensor([0, 1, 0, 1, 2, 2], device=device), torch.arange(6, device=device))


def reference(f, pid, cam, modes, keys):
    z = nn.functional.normalize(f, dim=1)
    c = z.detach()
    anchor_losses = []
    for i in range(len(f)):
        p = (pid == pid[i]) & (cam != cam[i]) & (keys != keys[i])
        n = (pid != pid[i]) & (cam != cam[i]) & (keys != keys[i])
        logits = z[i] @ c.t() / .07
        v = torch.logsumexp(logits[n], 0)
        groups = [nn.functional.softplus(v - logits[p & (modes == m)]).mean()
                  for m in range(3) if bool((p & (modes == m)).any())]
        anchor_losses.append(torch.stack(groups).mean())
    return torch.stack([torch.stack([anchor_losses[i] for i in range(len(f)) if modes[i] == m]).mean()
                        for m in range(3) if bool((modes == m).any())]).mean()


class LossTests(unittest.TestCase):
    def test_true_feature_includes_classifier_branch(self):
        model = SmallModel()
        score, returned = model(data()[0])
        # Mimic distinct differentiable DDP output wrappers.
        returned = returned.clone()
        ce = nn.functional.cross_entropy(score, data()[1])
        metric = returned.square().mean()
        observed = torch.autograd.grad(ce + metric, returned, retain_graph=True)[0]
        raw = model.raw_feature
        full = torch.autograd.grad(ce + metric, raw, retain_graph=True)[0]
        ce_grad = torch.autograd.grad(ce, raw, retain_graph=True)[0]
        torch.testing.assert_close(full, observed + ce_grad)
        self.assertGreater(float(ce_grad.norm()), 0)

    def test_reference_scalar_and_single_sided_gradient(self):
        x, *meta = data()
        f = x.clone().requires_grad_()
        loss, stats = mod.mixed_rank_loss(f, *meta, distributed=False)
        ref = reference(f, *meta)
        torch.testing.assert_close(loss, ref)
        torch.testing.assert_close(torch.autograd.grad(loss, f)[0], torch.autograd.grad(ref, f)[0])
        self.assertEqual(stats['valid_fraction'], 1)

    def test_empty_rows_have_finite_zero_gradient(self):
        x, pid, cam, modes, keys = data()
        f = x.requires_grad_()
        loss, stats = mod.mixed_rank_loss(f, pid, torch.zeros_like(cam), modes, keys, distributed=False)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(stats['valid_fraction'], 0)
        self.assertTrue(torch.equal(f.grad, torch.zeros_like(f)))

    def test_duplicate_candidates_keep_first_and_check_metadata(self):
        x, pid, cam, modes, keys = data()
        loss, diag = mod.mixed_rank_loss(x.requires_grad_(), pid, cam, modes, keys, distributed=False)
        # Duplicate an identical complete batch: anchor means and candidates stay same.
        value, diag2 = mod.mixed_rank_loss(x.repeat(2, 1), pid.repeat(2), cam.repeat(2),
                                         modes.repeat(2), keys.repeat(2), distributed=False)
        torch.testing.assert_close(loss, value)
        self.assertEqual(diag['candidates'], diag2['candidates'])
        keys[1] = keys[0]
        with self.assertRaises(ValueError):
            mod.mixed_rank_loss(x, pid, cam, modes, keys, distributed=False)

    def test_budget_and_nonfinite(self):
        f = torch.tensor([1., 2.], requires_grad=True)
        base, aux = f.square().sum(), 100 * f.square().sum()
        value, diag = mod.budgeted_auxiliary_loss(base, aux, f, 1., 1., distributed=False)
        self.assertLessEqual(diag['grad_ratio'], .100001)
        value.backward()
        f = torch.tensor([1., 2.], requires_grad=True)
        base, aux = f.square().sum(), f.sum() * float('nan')
        value, diag = mod.budgeted_auxiliary_loss(base, aux, f, 1., 1., distributed=False)
        self.assertFalse(diag['finite'])
        value.backward()
        self.assertTrue(torch.isfinite(f.grad).all())


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)
        self.unused_head = nn.Linear(4, 2)

    def forward(self, x):
        feat = self.backbone(x)
        self.raw_feature = feat
        return self.head(self.bn(feat)), feat


def distributed_check(rank, backend, init_path):
    world = 2
    if backend == 'nccl':
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend, init_method='file://' + init_path, rank=rank, world_size=world)
    try:
        # Uneven modality/valid counts; rank1 alone has no valid rows in local mode.
        x, pid, cam, modes, keys = data(device)
        indices = torch.tensor([0, 1, 2, 3] if rank == 0 else [4, 5], device=device)
        f = x[indices].clone().requires_grad_()
        local, diag = mod.mixed_rank_loss(f, pid[indices], cam[indices], modes[indices], keys[indices],
                                        distributed=True, gather_candidates=False)
        assert torch.isfinite(local)
        mean = local.detach().clone()
        dist.all_reduce(mean)
        mean /= world
        # Only rank0's local pool has positive/negative rows. Global mean must
        # equal rank0's complete two-modality objective, not half that value.
        torch.testing.assert_close(mean, reference(x[:4], pid[:4], cam[:4], modes[:4], keys[:4]))
        local.backward()
        assert torch.isfinite(f.grad).all()
        # Actual DDP with BN + unused head: intermediate probes then normal backward.
        torch.manual_seed(3)
        model = SmallModel().to(device)
        ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if backend == 'nccl' else None,
                                                 find_unused_parameters=True)
        opt = torch.optim.SGD(ddp.parameters(), lr=.01)
        scaler = torch.cuda.amp.GradScaler(enabled=backend == 'nccl')
        x, pid, cam, modes, keys = data(device)
        # Equal-size slices required for candidate gathering.
        ix = slice(rank * 3, (rank + 1) * 3)
        for step in range(4):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=backend == 'nccl'):
                score, f = ddp(x[ix])
                f = model.raw_feature
                base = nn.functional.cross_entropy(score, pid[ix]) + f.square().mean()
            aux, _ = mod.mixed_rank_loss(f, pid[ix], cam[ix], modes[ix], keys[ix])
            total, budget = mod.budgeted_auxiliary_loss(base, aux, f, scaler.get_scale(), 1.)
            assert all(p.grad is None for p in model.parameters())
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            assert model.backbone.weight.grad is not None
            assert model.unused_head.weight.grad is None
            assert budget['grad_ratio'] <= .100001
            scaler.step(opt)
            scaler.update()
        # All replicas must stay identical after four true DDP steps.
        vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        other = torch.empty_like(vector)
        if rank == 0: other.copy_(vector)
        dist.broadcast(other, 0)
        torch.testing.assert_close(vector, other)
        if rank == 0: print('2-rank {} DDP checks passed (4 steps, BN, unused head, gradient probes)'.format(backend))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    parser.add_argument('--skip-ddp', action='store_true', help='Run only single-process unit checks')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LossTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    if not args.skip_ddp:
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'process_group').replace('\\', '/')
            mp.spawn(distributed_check, args=(args.backend, path), nprocs=2, join=True)
