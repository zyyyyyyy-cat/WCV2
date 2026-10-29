import importlib.util
from pathlib import Path
import tempfile
import unittest
import csv
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage_diag', ROOT/'diagnostics/coverage.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def records():
    return [dict(path='train/m%d/%d_%d.jpg'%(m,p,k),pid=p,camid=k%2,
                 modality=m,sample_key=(m*8+p)*4+k)
            for m in range(3) for p in range(8) for k in range(4)]


class CoverageTests(unittest.TestCase):
    def test_original_sampler_equivalence(self):
        spec = importlib.util.spec_from_file_location('source_sampler', ROOT/'datasets/sampler_ddp.py')
        src = importlib.util.module_from_spec(spec); spec.loader.exec_module(src)
        rs = records(); sampler = object.__new__(src.PKMSampler_DDP)
        sampler.pids=list(range(8)); sampler.modality_ls=[0,1,2]
        sampler.num_instances=2; sampler.num_pids_per_batch=2
        sampler.pid_max_count={p:4 for p in sampler.pids}
        sampler.index_dic=defaultdict(lambda:defaultdict(list))
        mapping={m:[] for m in range(3)}
        for i,r in enumerate(rs):
            m=r['modality']; sampler.index_dic[r['pid']][m].append(len(mapping[m])); mapping[m].append(i)
        state=np.random.get_state()
        try:
            np.random.seed(1234)
            expected=[tuple(mapping[m][i] for m,i in enumerate(pos)) for pos in sampler.sample_list()]
        finally: np.random.set_state(state)
        actual=mod._epoch_positions(rs,1234,2,2)
        self.assertEqual(actual,expected)

    def test_plan_layout_determinism_and_no_rng_mutation(self):
        rs=records(); before=np.random.get_state()
        p=mod.generate_sampling_plan(rs,coverage_rounds=2,batches_per_round=2,probe_batches=2,
                                     world_size=2,identities_per_rank=2,instances=2)
        after=np.random.get_state(); np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(p,mod.generate_sampling_plan(rs,coverage_rounds=2,batches_per_round=2,probe_batches=2,
                                     world_size=2,identities_per_rank=2,instances=2))
        for round_ in p['coverage_batches']:
            for batch in round_:
                for rank in batch:
                    self.assertEqual([rs[i]['modality'] for i in rank],[0]*4+[1]*4+[2]*4)

    def test_cached_extrema_and_exposure(self):
        rs=records(); z=np.random.RandomState(2).randn(len(rs),5).astype('float32')
        plan=mod.generate_sampling_plan(rs,coverage_rounds=1,batches_per_round=1,probe_batches=1,
                                       world_size=2,identities_per_rank=2,instances=2)
        with tempfile.TemporaryDirectory() as folder:
            out=mod.analyze_coverage(z,z,rs,plan,folder,chunk_size=3)
            self.assertEqual(out['models']['uad'],out['models']['wcv1'])
            again=mod.analyze_coverage(z,z,rs,plan,folder,chunk_size=3)
            self.assertEqual(out,again)
            a=np.array([0]); norm=mod._normalize(z)
            actual=mod._full_extrema(norm,rs,a,Path(folder),'small',3)[0]
            permitted=[i for i,r in enumerate(rs) if r['camid']!=rs[0]['camid'] and r['pid']!=0]
            expected=max(permitted,key=lambda i:float(norm[0]@norm[i]))
            self.assertEqual(actual['neg_index'],expected)

    def test_global_attraction_coefficients_match_manual_v2(self):
        import torch
        rs=records(); z=mod._normalize(np.random.RandomState(7).randn(len(rs),5).astype('float32'))
        plan=mod.generate_sampling_plan(rs,coverage_rounds=1,batches_per_round=1,probe_batches=1,
                                       world_size=2,identities_per_rank=2,instances=2)
        ix=np.asarray(plan['coverage_batches'][0][0]).reshape(-1)
        keys=np.array([r['sample_key'] for r in rs]); _,first=np.unique(keys[ix],return_index=True)
        c=ix[np.sort(first)]
        sims=torch.tensor(z[ix]@z[c].T,dtype=torch.float64,requires_grad=True)
        pid=np.array([r['pid'] for r in rs]); cam=np.array([r['camid'] for r in rs]); mode=np.array([r['modality'] for r in rs])
        allowed=(cam[ix,None]!=cam[c][None]) & (keys[ix,None]!=keys[c][None])
        p=allowed & (pid[ix,None]==pid[c][None]); n=allowed & ~p
        valid=p.any(1)&n.any(1); terms=[]
        for a in range(len(ix)):
            if not valid[a]: terms.append(None); continue
            v=torch.logsumexp(sims[a,torch.tensor(n[a])]/.07,0)
            groups=[torch.nn.functional.softplus(v-sims[a,torch.tensor(p[a]&(mode[c]==m))]/.07).mean()
                    for m in range(3) if (p[a]&(mode[c]==m)).any()]
            terms.append(torch.stack(groups).mean())
        loss=torch.stack([torch.stack([terms[a] for a in range(len(ix)) if valid[a] and mode[ix[a]]==m]).mean()
                          for m in range(3) if (valid&(mode[ix]==m)).any()]).mean()
        gradient=torch.autograd.grad(loss,sims)[0].numpy()
        expected=float(-gradient[p].sum())
        with tempfile.TemporaryDirectory() as folder:
            mod.analyze_coverage(z,z,rs,plan,folder,chunk_size=5)
            with open(Path(folder)/'uad_positive_relations.csv',encoding='utf-8-sig') as f:
                actual=sum(float(r['positive_similarity_gradient_sum_scaled_global']) for r in csv.DictReader(f))
            self.assertAlmostEqual(actual,expected,places=5)


if __name__=='__main__': unittest.main()
