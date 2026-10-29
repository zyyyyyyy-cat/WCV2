"""Paired full-gallery diagnostics. No model, labels never change scores."""
import csv
import json
from pathlib import Path
import numpy as np


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('features must be a finite [N,D] array')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _protocols(q, g):
    qi, gi = np.arange(len(q)), np.arange(len(g))
    yield 'overall', qi, gi
    names = ['RGB', 'NIR', 'TIR']
    for a in range(3):
        for b in range(3):
            yield names[a] + '_to_' + names[b], qi[[r['modality'] == a for r in q]], gi[[r['modality'] == b for r in g]]
    for a, b, name in [(True, False, 'A_to_G'), (False, True, 'G_to_A')]:
        yield name, qi[[(r['camid'] >= 5) == a for r in q]], gi[[(r['camid'] >= 5) == b for r in g]]
    def gd(r):
        return 1000 <= r.get('original_pid', r['pid']) < 2000 and r['camid'] < 5
    yield '1000_GD', qi[[gd(r) for r in q]], gi[[gd(r) for r in g]]


def _one(scores, q, gallery, global_indices):
    legal = np.array([r['camid'] != q['camid'] for r in gallery], dtype=bool)
    pos = np.array([r['pid'] == q['pid'] for r in gallery], dtype=bool) & legal
    if not pos.any():
        return None
    # Stable ties make diagnostics reproducible. No top-K approximation.
    order = np.argsort(-scores, kind='stable')
    order = order[legal[order]]
    ranks = np.flatnonzero(pos[order]) + 1
    negative = order[~pos[order]]
    bp = int(order[ranks[0] - 1])
    bn = int(negative[0]) if len(negative) else None
    return dict(AP=float(np.mean(np.arange(1, len(ranks) + 1) / ranks)),
                INP=float(len(ranks) / ranks[-1]), first_positive_rank=int(ranks[0]),
                last_positive_rank=int(ranks[-1]), positive_count=len(ranks),
                R1=int(ranks[0] <= 1), R5=int(ranks[0] <= 5), R10=int(ranks[0] <= 10),
                best_positive=int(global_indices[bp]), top_negative=int(global_indices[bn]) if bn is not None else None,
                best_positive_score=float(scores[bp]), top_negative_score=float(scores[bn]) if bn is not None else None)


def _write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ranking(uad_query, uad_gallery, wcv1_query, wcv1_gallery,
                query_records, gallery_records, outdir, chunk_size=64):
    """Write paired query metrics and crossed extrema for every legal protocol.

    Inputs use the exact same ordered record lists for both checkpoints.
    AP/INP/Rk in outputs are fractions. Candidate IDs index gallery_records.
    Returns a JSON-serializable report; invalid queries are recorded, not zeroed.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    uq, ug, wq, wg = map(_normalize, (uad_query, uad_gallery, wcv1_query, wcv1_gallery))
    if uq.shape != wq.shape or ug.shape != wg.shape or uq.shape[1] != ug.shape[1]:
        raise ValueError('feature shape mismatch')
    if len(uq) != len(query_records) or len(ug) != len(gallery_records):
        raise ValueError('record count mismatch')
    for r in query_records + gallery_records:
        if r['modality'] not in (0, 1, 2) or r['camid'] < 0:
            raise ValueError('expected modality 0..2 and zero-based camera')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {'units': 'fraction; score=cosine', 'tie_policy': 'stable gallery order', 'protocols': {}}
    for name, qi, gi in _protocols(query_records, gallery_records):
        rows, candidates, invalid = [], [], []
        transition = dict(new_failure=0, repaired=0, kept_success=0, kept_failure=0)
        sums = {m: {k: 0. for k in ['AP', 'INP', 'R1', 'R5', 'R10']} for m in ['uad', 'wcv1']}
        grec = [gallery_records[i] for i in gi]
        for start in range(0, len(qi), chunk_size):
            qs = qi[start:start + chunk_size]
            us, ws = uq[qs] @ ug[gi].T, wq[qs] @ wg[gi].T
            for offset, i in enumerate(qs):
                q = query_records[i]
                u, w = _one(us[offset], q, grec, gi), _one(ws[offset], q, grec, gi)
                if u is None:
                    assert w is None
                    invalid.append(dict(query_index=int(i), query_path=q['path'], reason='no_cross_camera_positive'))
                    continue
                status = ('kept_success' if w['R1'] else 'new_failure') if u['R1'] else ('repaired' if w['R1'] else 'kept_failure')
                transition[status] += 1
                row = dict(query_index=int(i), query_path=q['path'], pid=q['pid'], camid=q['camid'], modality=q['modality'], transition=status)
                for model, metrics in [('uad', u), ('wcv1', w)]:
                    row.update({model + '_' + k: v for k, v in metrics.items()})
                    for k in sums[model]:
                        sums[model][k] += metrics[k]
                row['delta_AP'] = w['AP'] - u['AP']
                row['delta_INP'] = w['INP'] - u['INP']
                row['best_positive_envelope_change'] = w['best_positive_score'] - u['best_positive_score']
                if u['top_negative_score'] is not None and w['top_negative_score'] is not None:
                    row['top_negative_envelope_change'] = w['top_negative_score'] - u['top_negative_score']
                    row['actual_margin_change'] = row['best_positive_envelope_change'] - row['top_negative_envelope_change']
                else:
                    row['top_negative_envelope_change'] = None
                    row['actual_margin_change'] = None
                # Union of extrema under both models, scored under both models.
                union = sorted(set(x for x in [u['best_positive'], w['best_positive'], u['top_negative'], w['top_negative']] if x is not None))
                cross = {}
                for j in union:
                    g = gallery_records[j]
                    su, sw = float(uq[i] @ ug[j]), float(wq[i] @ wg[j])
                    role = '|'.join(m + '_' + role for m, met in [('uad', u), ('wcv1', w)] for role in ['best_positive', 'top_negative'] if met[role] == j)
                    candidates.append(dict(query_index=int(i), query_path=q['path'], gallery_index=j, gallery_path=g['path'], gallery_pid=g['pid'], gallery_camid=g['camid'], gallery_modality=g['modality'], roles=role, same_identity=int(g['pid'] == q['pid']), uad_score=su, wcv1_score=sw, delta_score=sw-su))
                    cross[j] = (su, sw)
                # Fixed UAD extrema make an additive, exactly closing margin decomposition.
                if u['top_negative'] is not None:
                    pu, pw = cross[u['best_positive']]
                    nu, nw = cross[u['top_negative']]
                    row['fixed_positive_change'] = pw-pu
                    row['fixed_negative_change'] = nw-nu
                    row['fixed_margin_change'] = (pw-pu)-(nw-nu)
                    if pw < pu and nw > nu:
                        cause = 'both_adverse'
                    elif pw < pu:
                        cause = 'positive_weakened'
                    elif nw > nu:
                        cause = 'negative_strengthened'
                    else:
                        cause = 'neither_on_uad_extrema'
                else:
                    row.update(fixed_positive_change=None, fixed_negative_change=None, fixed_margin_change=None)
                    cause = 'no_legal_negative'
                row['extrema_change_category'] = cause
                dp, dn = row['best_positive_envelope_change'], row['top_negative_envelope_change']
                adverse_p = max(-dp, 0.)
                adverse_n = max(dn, 0.) if dn is not None else 0.
                row['adverse_envelope_dominance'] = ('positive_weakened' if adverse_p > adverse_n else
                    ('negative_strengthened' if adverse_n > adverse_p else ('equal_adverse' if adverse_p else 'none')))
                row['negative_identity_switched'] = int(u['top_negative'] is not None and w['top_negative'] is not None and gallery_records[u['top_negative']]['pid'] != gallery_records[w['top_negative']]['pid'])
                rows.append(row)
        n = len(rows)
        summary = dict(status='ok' if n else 'SKIPPED', query_count=len(qi), gallery_count=len(gi), valid_queries=n, invalid_queries=len(invalid), transition_counts=transition)
        for model in sums:
            summary[model] = {k: v/n for k, v in sums[model].items()} if n else None
            if n:
                summary[model].update(mAP=summary[model]['AP'], mINP=summary[model]['INP'])
        summary['rank1_closure_error'] = (summary['wcv1']['R1']-summary['uad']['R1']-(transition['repaired']-transition['new_failure'])/n) if n else None
        summary['new_failure_categories'] = {c: sum(r['transition']=='new_failure' and r['extrema_change_category']==c for r in rows) for c in ['both_adverse','positive_weakened','negative_strengthened','neither_on_uad_extrema','no_legal_negative']}
        summary['head_tail_joint'] = dict(new_failure_inp_up=sum(r['transition']=='new_failure' and r['delta_INP']>0 for r in rows), new_failure_inp_down=sum(r['transition']=='new_failure' and r['delta_INP']<0 for r in rows), new_failure_inp_equal=sum(r['transition']=='new_failure' and r['delta_INP']==0 for r in rows))
        _write_csv(outdir / (name + '_queries.csv'), rows)
        _write_csv(outdir / (name + '_candidate_union.csv'), candidates)
        _write_csv(outdir / (name + '_invalid_queries.csv'), invalid)
        report['protocols'][name] = summary
    (outdir / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
