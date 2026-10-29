import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics import cache


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)
        self.calls = 0

    def forward(self, images, mode=1):
        self.calls += 1
        return images.mean((2, 3)).repeat(1, 256)


def simple_transform(image):
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255


class CacheTests(unittest.TestCase):
    def test_extract_reuse_and_content_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (3, 4), (20, 30, 40)).save(root / 'sample.jpg')
            rows = [dict(path='sample.jpg')]
            model = Tiny().eval()
            with mock.patch.object(cache, 'transformations', return_value=simple_transform):
                f = cache.extract_model(model, {'train': rows}, root, None, root/'cache', 'uad', 'fp', 'cpu', workers=0)
                self.assertEqual(f['train'].shape, (1, 768))
                cache.extract_model(model, {'train': rows}, root, None, root/'cache', 'uad', 'fp', 'cpu', workers=0)
                self.assertEqual(model.calls, 1)
                del f
                path = root/'cache/uad/train.npy'
                array = np.load(path, mmap_mode='r+')
                array[0, 0] += 1
                array.flush(); del array
                with self.assertRaises(ValueError):
                    cache.extract_model(model, {'train': rows}, root, None, root/'cache', 'uad', 'fp', 'cpu', workers=0)

    def test_reader_repeats_without_rng_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGB', (2, 2)).save(Path(tmp)/'sample.jpg')
            def augmented(image):
                return torch.rand(2) + random.random() + np.random.random()
            with mock.patch.object(cache, 'transformations', return_value=augmented):
                reader = cache.deterministic_reader(None, tmp, [dict(path='sample.jpg')])
                py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                a = reader([0], 42); b = reader([0], 42)
                torch.testing.assert_close(a, b)
                self.assertEqual(py, random.getstate())
                np.testing.assert_equal(ns, np.random.get_state())
                self.assertTrue(torch.equal(ts, torch.get_rng_state()))

    def test_checkpoint_requires_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Tiny()
            path = Path(tmp)/'checkpoint.pth'
            torch.save(model.state_dict(), path)
            dataset = SimpleNamespace(num_train_pids=2, num_train_cams=2, num_train_vids=2)
            module = SimpleNamespace(make_model=lambda *a, **kw: Tiny())
            with mock.patch.dict(sys.modules, {'model': module}):
                loaded, bank = cache.load_model(None, dataset, path, 'cpu')
                torch.testing.assert_close(loaded.classifier.weight, model.classifier.weight)
                torch.save({}, path)
                with self.assertRaises(RuntimeError):
                    cache.load_model(None, dataset, path, 'cpu')


if __name__ == '__main__':
    unittest.main()
