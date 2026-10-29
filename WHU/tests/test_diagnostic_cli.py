"""Cache-only CLI integration and configuration rejection, no trained weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_wcv1 as cli
from diagnostics.cache import cached_features, json_hash, save_json, sha256_file


def fixture(cache):
    records = {'train': [], 'query': [], 'gallery': []}
    for m in range(3):
        for pid in range(16):
            for k in range(16):
                records['train'].append(dict(path=f'train/m{m}/{pid}_c{k%6+1}_{k}.jpg',
                    pid=pid, original_pid=1000+pid, camid=k%6, modality=m,
                    sample_key=len(records['train'])))
    for split, count in [('query', 6), ('gallery', 12)]:
        for i in range(count):
            records[split].append(dict(path=f'{split}/m{i%3}/{1000+i%2}_c{i%6+1}_{i}.jpg',
                pid=1000+i%2, original_pid=1000+i%2, camid=i%6, modality=i%3, sample_key=i))
    fingerprints = {'uad': 'synthetic-uad', 'wcv1': 'synthetic-wcv1'}
    save_json(cache/'manifest.json', dict(records=records, fingerprints=fingerprints))
    arrays = {}
    for name in fingerprints:
        arrays[name] = {}
        for split, rows in records.items():
            array = np.zeros((len(rows), 768), dtype=np.float32)
            for i, row in enumerate(rows):
                array[i, row['pid']%16] = 1.
                array[i, 16+row['camid']] = .01
            arrays[name][split] = array
            path = cache/name/(split+'.npy'); path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
            save_json(path.with_suffix('.json'), dict(fingerprint=fingerprints[name],
                      records_hash=json_hash(rows), shape=[len(rows),768], feature_sha256=sha256_file(path)))
    return records, fingerprints, arrays


class CLITests(unittest.TestCase):
    def test_cache_only_full_ranking_coverage_no_torch_import(self):
        with tempfile.TemporaryDirectory() as d:
            cache, out = Path(d)/'cache', Path(d)/'out'
            fixture(cache)
            argv = ['--stages','ranking,coverage','--cache-dir',str(cache),'--output-dir',str(out),
                    '--coverage-rounds','1','--batches-per-round','1','--probe-groups','1','--chunk-size','64']
            code = ('import sys; import diagnose_wcv1 as m; '
                    'm.main('+repr(argv)+'); assert "torch" not in sys.modules')
            result = subprocess.run([sys.executable,'-c',code], cwd=ROOT, capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            summary = json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['completed_stages'],['ranking','coverage'])
            self.assertTrue(summary['no_optimizer_steps'])
            self.assertEqual(summary['ranking']['protocols']['overall']['transition_counts']['new_failure'],0)
            self.assertTrue((out/'report.md').is_file())
            self.assertEqual(summary['coverage']['models']['uad'],summary['coverage']['models']['wcv1'])

    def test_stale_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            records, fingerprints, _ = fixture(Path(d))
            records['query'][0]['camid'] = 99
            with self.assertRaisesRegex(ValueError,'mismatched'):
                cached_features(d,records,fingerprints)

    def test_unsupported_probe_objective_rejected_before_forward(self):
        from config import cfg
        c = cfg.clone(); c.defrost(); c.DATALOADER.NUM_INSTANCE = 2
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data_cache = root/'data_cache'
            records, _, arrays = fixture(data_cache)
            checkpoint = root/'dummy.pth'; checkpoint.write_bytes(b'not loaded')
            with patch.object(cli,'make_config',return_value=c), \
                 patch.object(cli,'make_records',return_value=(types.SimpleNamespace(dataset_dir=d),records)), \
                 patch.object(cli,'load_model',return_value=(None,None)) as loader, \
                 patch.object(cli,'cached_features',return_value=arrays):
                with self.assertRaisesRegex(ValueError,'probe objective supports'):
                    cli.main(['--stages','probes','--device','cpu','--data-root',d,
                        '--uad-checkpoint',str(checkpoint),'--wcv1-checkpoint',str(checkpoint),
                        '--cache-dir',str(root/'probe_cache'),'--output-dir',str(root/'out'),
                        '--coverage-rounds','1','--batches-per-round','1','--probe-groups','1'])
                self.assertEqual(loader.call_count,2)

    def test_invalid_cli_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in [['--stages','unknown'],['--chunk-size','0']]:
                with self.assertRaises(ValueError):
                    cli.main(['--cache-dir',d,'--output-dir',d]+extra)


if __name__ == '__main__': unittest.main()
