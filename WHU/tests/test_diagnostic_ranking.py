"""Small known-answer ranking tests; no torch or real dataset required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import ast
import numpy as np

spec = importlib.util.spec_from_file_location('diagnostic_ranking', Path(__file__).resolve().parents[1]/'diagnostics/ranking.py')
ranking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranking)


def rec(pid, cam, mod=0, path='image'):
    return dict(pid=pid, original_pid=1000+pid, camid=cam, modality=mod, path=path, sample_key=path)


class RankingTests(unittest.TestCase):
    def test_matches_original_whu_eval_func(self):
        # Load only the original function AST, avoiding torch/reranking imports.
        source = (Path(__file__).resolve().parents[1]/'utils/metrics.py').read_text(encoding='utf-8-sig')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='eval_func')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<original WHU eval_func>', 'exec'), namespace)
        gallery=[rec(9,0),rec(0,1),rec(1,1),rec(0,2)]
        queries=[rec(0,0),rec(1,0)]
        scores=np.array([[1.,.9,.8,.7],[1.,.7,.9,.8]])
        order=np.argsort(-scores,axis=1,kind='stable')
        cmc,ap,inp=namespace['eval_func'](order,np.array([0,1]),np.array([9,0,1,0]),np.array([0,0]),np.array([0,1,1,2]),max_rank=3,return_minp=True)
        ours=[ranking._one(s,q,gallery,np.arange(4)) for s,q in zip(scores,queries)]
        self.assertAlmostEqual(ap,np.mean([m['AP'] for m in ours]))
        self.assertAlmostEqual(inp,np.mean([m['INP'] for m in ours]))
        self.assertAlmostEqual(float(cmc[0]),np.mean([m['R1'] for m in ours]))

    def test_known_ap_inp_and_same_camera(self):
        q = rec(0, 0)
        gs = [rec(9,0),rec(0,1),rec(1,1),rec(0,2)]
        m = ranking._one(np.array([1.,.9,.8,.7]),q,gs,np.arange(4))
        self.assertAlmostEqual(m['AP'], (1+2/3)/2)
        self.assertAlmostEqual(m['INP'], 2/3)
        self.assertEqual(m['first_positive_rank'],1)
        self.assertEqual(m['last_positive_rank'],3)
        self.assertEqual(m['top_negative'],2)
        self.assertIsNone(ranking._one(np.ones(1),q,[rec(0,0)],np.arange(1)))

    def test_pair_transition_closure_and_crossed_candidates(self):
        qs=[rec(0,0,path='q0'),rec(1,0,path='q1'),rec(77,0,path='invalid')]
        gs=[rec(0,1,path='p0'),rec(1,1,path='p1'),rec(2,1,path='n2')]
        uq=np.array([[1.,0.],[0.,1.],[1.,1.]])
        ug=np.array([[1.,0.],[0.,1.],[-1.,0.]])
        wq=uq.copy()
        wg=np.array([[.6,.8],[.8,.6],[1.,0.]])
        with tempfile.TemporaryDirectory() as out:
            report=ranking.run_ranking(uq,ug,wq,wg,qs,gs,out,chunk_size=1)
            overall=report['protocols']['overall']
            self.assertEqual(overall['valid_queries'],2)
            self.assertEqual(overall['invalid_queries'],1)
            self.assertEqual(overall['transition_counts']['new_failure'],2)
            self.assertAlmostEqual(overall['rank1_closure_error'],0)
            self.assertEqual(report['protocols']['A_to_G']['status'],'SKIPPED')
            import csv
            with open(Path(out)/'overall_candidate_union.csv',encoding='utf-8-sig') as f:
                union=list(csv.DictReader(f))
            self.assertTrue(any(r['gallery_path']=='n2' and r['query_path']=='q0' for r in union))
            with open(Path(out)/'overall_queries.csv',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            for r in rows:
                self.assertAlmostEqual(float(r['fixed_margin_change']),float(r['fixed_positive_change'])-float(r['fixed_negative_change']))

    def test_chunk_invariance_and_invalid_input(self):
        qs=[rec(0,0,path='q0'),rec(1,0,path='q1')]
        gs=[rec(0,1,path='g0'),rec(1,1,path='g1')]
        f=np.eye(2,dtype=np.float32)
        with tempfile.TemporaryDirectory() as out:
            a=ranking.run_ranking(f,f,f,f,qs,gs,Path(out)/'a',1)
            b=ranking.run_ranking(f,f,f,f,qs,gs,Path(out)/'b',4)
            self.assertEqual(a,b)
            with self.assertRaises(ValueError):
                ranking.run_ranking(f*np.nan,f,f,f,qs,gs,out)

    def test_all_four_transition_classes(self):
        qs=[rec(i,0,path='q'+str(i)) for i in range(4)]
        gs=[rec(i,1,path='p'+str(i)) for i in range(4)]+[rec(10+i,1,path='n'+str(i)) for i in range(4)]
        q=np.eye(5,dtype=np.float32)[:4]
        def gallery(positive_scores):
            x=np.zeros((8,5),dtype=np.float32)
            for i,s in enumerate(positive_scores):
                x[i,i]=s; x[i,4]=np.sqrt(1-s*s)
                x[4+i,i]=.9; x[4+i,4]=np.sqrt(1-.9**2)
            return x
        with tempfile.TemporaryDirectory() as out:
            s=ranking.run_ranking(q,gallery([.99,.5,.99,.5]),q,gallery([.5,.99,.99,.5]),qs,gs,out)['protocols']['overall']
            self.assertEqual(s['transition_counts'],dict(new_failure=1,repaired=1,kept_success=1,kept_failure=1))
            self.assertAlmostEqual(s['rank1_closure_error'],0)


if __name__=='__main__':
    unittest.main()
