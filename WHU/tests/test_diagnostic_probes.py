import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.probes import rebuild_bank, run_probes, state_hash


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2))
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, images, label=None, cams=None):
        f = self.backbone(torch.cat(images))
        if getattr(self, 'capture_wcv1_feature', False): self.wcv1_raw_feature = f
        h = self.bn(f)
        return self.head(h), f, h


class ProbeTests(unittest.TestCase):
    def test_bank_is_equal_modality_not_image_count(self):
        f = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
        records = [dict(pid=0, modality=m) for m in [0, 0, 1]]
        torch.testing.assert_close(rebuild_bank(f, records)[0], torch.tensor([2**-.5, 2**-.5]))

    def test_probe_restores_state_and_splits_gradient(self):
        torch.manual_seed(7)
        records = [dict(pid=(i % 8)//4, camid=i % 4, modality=i//8, sample_key=i) for i in range(24)]
        inputs = torch.randn(24, 3)
        model = Toy().eval()
        old = state_hash(model)
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            result = run_probes({'test': model}, records, [[list(range(24)) for _ in range(4)]],
                                lambda ix, seed: inputs[ix], {'test': np.random.RandomState(4).randn(24, 4).astype('float32')},
                                d, device='cpu')
            self.assertTrue(result['state_checks']['test']['unchanged'])
            self.assertGreater(result['rows'], 0)
        self.assertEqual(state_hash(model), old)
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, 'wcv1_raw_feature'))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main()
