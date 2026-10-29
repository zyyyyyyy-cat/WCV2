"""Training-only PKM candidate exposure and positive-relation diagnostics."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def _epoch_positions(records, seed, identities_per_rank, instances):
    """Exact RandomState counterpart of PKMSampler_DDP.sample_list()."""
    rng = np.random.RandomState(seed)
    index = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        index[int(r['pid'])][int(r['modality'])].append(i)
    pids = sorted(index)
    groups = defaultdict(dict)
    for pid in pids:
        if any(not index[pid][m] for m in range(3)):
            raise ValueError('PKM requires every training identity in all modalities')
        need = max(max(len(index[pid][m]), instances) for m in range(3))
        need -= need % instances
        for m in range(3):
            idxs = list(index[pid][m]); cur = len(idxs)
            if cur >= need:
                idxs = rng.choice(idxs, size=need, replace=False)
            else:
                k, remainder = divmod(need, cur)
                idxs = idxs * k
                if remainder:
                    idxs += rng.choice(index[pid][m], size=remainder, replace=False).tolist()
            rng.shuffle(idxs)
            groups[pid][m] = [idxs[i:i + instances] for i in range(0, len(idxs), instances)]
    positions = []
    available = pids[:]
    while len(available) >= identities_per_rank:
        selected = rng.choice(available, identities_per_rank, replace=False).tolist()
        for pid in selected:
            positions.extend(zip(*(groups[pid][m].pop(0) for m in range(3))))
            if not groups[pid][0]:
                available.remove(pid)
    return positions


def _batches(records, count, seed, world_size, identities_per_rank, instances):
    result = []; epoch = 0
    local = identities_per_rank * instances
    width = world_size * local
    while len(result) < count:
        positions = _epoch_positions(records, seed + epoch, identities_per_rank, instances)
        if len(positions) < width:
            raise ValueError('Insufficient PKM groups for one full simulated four-rank batch')
        for start in range(0, len(positions) - width + 1, width):
            batch = np.asarray(positions[start:start + width], dtype=np.int64)
            result.append([batch[r * local:(r + 1) * local].T.reshape(-1).tolist()
                           for r in range(world_size)])
            if len(result) == count:
                break
        epoch += 1
    return result


def generate_sampling_plan(records, seed=1234, coverage_rounds=8, batches_per_round=8,
                           probe_batches=16, world_size=4, identities_per_rank=16, instances=4):
    if not records or min(coverage_rounds, batches_per_round, probe_batches,
                          world_size, identities_per_rank, instances) <= 0:
        raise ValueError('Sampling sizes must be positive')
    return dict(seed=seed, world_size=world_size, identities_per_rank=identities_per_rank,
                instances=instances, sampling='PKMSampler_DDP.sample_list equivalent; fresh seeded organizations',
                coverage_batches=[_batches(records, batches_per_round, seed + r * 1000,
                                          world_size, identities_per_rank, instances)
                                  for r in range(coverage_rounds)],
                probe_batches=_batches(records, probe_batches, seed + 100000,
                                       world_size, identities_per_rank, instances))


def _normalize(features):
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or not np.isfinite(f).all():
        raise ValueError('Training features must be finite [N,D] arrays')
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def _write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields, restval=0)
        writer.writeheader(); writer.writerows(rows)


def _full_extrema(z, records, anchors, outdir, name, chunk_size):
    """Resume by completed anchor chunks; content-addressed identity prevents stale reuse."""
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(z).view(np.uint8)); digest.update(anchors.tobytes())
    digest.update(('algorithm-v2-ranks/chunk-%d' % chunk_size).encode())
    folder = outdir / ('full_extrema_' + name + '_' + digest.hexdigest()[:16])
    folder.mkdir(parents=True, exist_ok=True)
    pid = np.array([r['pid'] for r in records]); cam = np.array([r['camid'] for r in records])
    keys = np.array([r['sample_key'] for r in records])
    result = {}
    for start in range(0, len(anchors), chunk_size):
        ix = anchors[start:start + chunk_size]; target = folder / ('chunk_%07d.npz' % start)
        if target.exists():
            with np.load(target) as saved:
                data = {k: saved[k] for k in saved.files}
        else:
            sim = z[ix] @ z.T
            allowed = (cam[ix, None] != cam[None]) & (keys[ix, None] != keys[None])
            pos = allowed & (pid[ix, None] == pid[None]); neg = allowed & ~pos
            ps = np.where(pos, sim, -np.inf); ns = np.where(neg, sim, -np.inf)
            pi = ps.argmax(1); ni = ns.argmax(1)
            columns = np.arange(len(z))[None, :]
            pscore = ps[np.arange(len(ix)), pi]; nscore = ns[np.arange(len(ix)), ni]
            prank = 1 + (allowed & ((sim > pscore[:, None]) |
                         ((sim == pscore[:, None]) & (columns < pi[:, None])))).sum(1)
            nrank = 1 + (allowed & ((sim > nscore[:, None]) |
                         ((sim == nscore[:, None]) & (columns < ni[:, None])))).sum(1)
            data = dict(anchor=ix, pos_index=pi, neg_index=ni,
                        pos_score=pscore, neg_score=nscore,
                        first_positive_rank=np.where(pos.any(1), prank, 0),
                        first_negative_rank=np.where(neg.any(1), nrank, 0))
            temporary = target.with_suffix('.tmp.npz')
            np.savez(temporary, **data)
            temporary.replace(target)
        for j, anchor in enumerate(data['anchor']):
            result[int(anchor)] = {k: data[k][j].item() for k in data if k != 'anchor'}
    return result


def analyze_coverage(features_uad, features_wcv1, records, plan, outdir, chunk_size=64, tau=.07):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0 or tau <= 0:
        raise ValueError('chunk_size and tau must be positive')
    if any(str(r['path']).replace('\\', '/').split('/')[0] != 'train' for r in records):
        raise ValueError('Candidate diagnostics accept train/ records only; no query/test identities')
    batches = [(r, b, np.asarray(batch).reshape(-1)) for r, round_ in enumerate(plan['coverage_batches'])
               for b, batch in enumerate(round_)]
    anchors = np.unique(np.concatenate([ix for _, _, ix in batches]))
    pids = np.array([r['pid'] for r in records]); cams = np.array([r['camid'] for r in records])
    mods = np.array([r['modality'] for r in records]); keys = np.array([r['sample_key'] for r in records])
    summary = {'scope': 'training identities only; eval-mode cached features, not historical training replay', 'models': {}}
    for name, features in [('uad', features_uad), ('wcv1', features_wcv1)]:
        if len(features) != len(records):
            raise ValueError('Feature/record length mismatch')
        z = _normalize(features)
        full = _full_extrema(z, records, anchors, outdir, name, chunk_size)
        rows = []; relations = defaultdict(lambda: defaultdict(float))
        for round_id, batch_id, ix in batches:
            _, first = np.unique(keys[ix], return_index=True); candidates = ix[np.sort(first)]
            scores = z[ix] @ z[candidates].T
            masks = (cams[ix, None] != cams[candidates][None]) & (keys[ix, None] != keys[candidates][None])
            pos_masks = masks & (pids[ix, None] == pids[candidates][None])
            neg_masks = masks & ~pos_masks
            valid_masks = pos_masks.any(1) & neg_masks.any(1)
            valid_counts = np.array([(valid_masks & (mods[ix] == m)).sum() for m in range(3)])
            active_mods = int((valid_counts > 0).sum())
            for a, anchor in enumerate(ix):
                allowed = (cams[candidates] != cams[anchor]) & (keys[candidates] != keys[anchor])
                positive = allowed & (pids[candidates] == pids[anchor]); negative = allowed & ~positive
                valid = bool(positive.any() and negative.any()); s = scores[a]
                neg_scores = s[negative]
                nmax = float(neg_scores.max()) if len(neg_scores) else float('-inf')
                pmax = float(s[positive].max()) if positive.any() else float('-inf')
                ent = eff = np.nan
                if len(neg_scores):
                    logit = neg_scores / tau; top = logit.max()
                    weights = np.exp(logit - top); weights /= weights.sum()
                    ent = float(-(weights * np.log(np.maximum(weights, 1e-30))).sum()); eff = float(np.exp(ent))
                    v = float(top + np.log(np.exp(logit - top).sum()))
                f = full[int(anchor)]; full_valid = np.isfinite(f['neg_score']) and np.isfinite(f['pos_score'])
                danger_pid = pids[f['neg_index']] if np.isfinite(f['neg_score']) else -1
                order = np.argsort(-s, kind='stable')
                filtered = order[allowed[order]]
                pranks = np.flatnonzero(positive[filtered]); nranks = np.flatnonzero(negative[filtered])
                rows.append(dict(model=name, round=round_id, batch=batch_id, anchor_index=int(anchor),
                    path=records[anchor]['path'], modality=int(mods[anchor]), platform='G' if cams[anchor] < 5 else 'A',
                    valid=int(valid), batch_pos=pmax, batch_neg=nmax, full_pos=f['pos_score'], full_neg=f['neg_score'],
                    batch_first_positive_rank=int(pranks[0]+1) if len(pranks) else 0,
                    batch_first_negative_rank=int(nranks[0]+1) if len(nranks) else 0,
                    full_first_positive_rank=f['first_positive_rank'],
                    full_first_negative_rank=f['first_negative_rank'],
                    full_negative_path=records[f['neg_index']]['path'] if np.isfinite(f['neg_score']) else '',
                    negative_gap=f['neg_score']-nmax if len(neg_scores) else np.nan,
                    danger_identity_present=int(bool(np.any(negative & (pids[candidates] == danger_pid)))),
                    danger_image_present=int(bool(np.any(negative & (keys[candidates] == keys[f['neg_index']]))))
                        if np.isfinite(f['neg_score']) else 0,
                    batch_separated_full_wrong=int(valid and full_valid and pmax > nmax and f['pos_score'] <= f['neg_score']),
                    negative_count=len(neg_scores), negative_entropy=ent, effective_negative_count=eff))
                nonempty = sum(bool((positive & (mods[candidates] == m)).any()) for m in range(3))
                for m in range(3):
                    all_m = positive & (mods[candidates] == m); count_m = int(all_m.sum())
                    for platform in ['G', 'A']:
                        selected = all_m & ((cams[candidates] < 5) if platform == 'G' else (cams[candidates] >= 5))
                        key = (int(mods[anchor]), 'G' if cams[anchor] < 5 else 'A', m, platform)
                        acc = relations[key]; acc['anchor_observations'] += 1; acc['positive_count'] += selected.sum()
                        acc['missing'] += not selected.any(); acc['similarity_sum'] += s[selected].sum()
                        if valid and selected.any():
                            x = v - s[selected] / tau
                            sigmoid = np.exp(-np.logaddexp(0., -x))
                            logit_coeff = sigmoid.sum() / (nonempty * count_m)
                            global_factor = 1.0 / (active_mods * valid_counts[mods[anchor]])
                            acc['positive_logit_gradient_sum_per_anchor_unscaled'] += logit_coeff
                            acc['positive_similarity_gradient_sum_per_anchor_unscaled'] += logit_coeff / tau
                            acc['positive_logit_gradient_sum_scaled_global'] += global_factor * logit_coeff
                            acc['positive_similarity_gradient_sum_scaled_global'] += global_factor * logit_coeff / tau
        _write_csv(outdir / (name + '_candidate_exposure.csv'), rows)
        relrows = []
        for key, a in sorted(relations.items()):
            relrows.append(dict(zip(['anchor_modality','anchor_platform','gallery_modality','gallery_platform'],key),
                **dict(a), missing_rate=a['missing']/a['anchor_observations'],
                mean_positive_similarity=a['similarity_sum']/a['positive_count'] if a['positive_count'] else None))
        _write_csv(outdir / (name + '_positive_relations.csv'), relrows)
        valid_rows = [r for r in rows if r['valid']]
        summary['models'][name] = dict(observations=len(rows), valid=len(valid_rows),
            danger_identity_present_rate=float(np.mean([r['danger_identity_present'] for r in valid_rows])) if valid_rows else None,
            danger_image_present_rate=float(np.mean([r['danger_image_present'] for r in valid_rows])) if valid_rows else None,
            batch_separated_full_wrong_rate=float(np.mean([r['batch_separated_full_wrong'] for r in valid_rows])) if valid_rows else None)
    summary['gradient_columns'] = dict(
        per_anchor_unscaled='Sum of positive attraction coefficient magnitudes per anchor; before global averaging',
        scaled_global='Sum after 1/(active_anchor_modalities * global_valid_count_of_anchor_modality); before lambda',
        ddp_rank_local='DDP backward uses world_size times scaled_global on each rank, then averages ranks',
        interpretation='Similarity/logit derivatives only, not parameter gradients; summed across sampled batches')
    (outdir / 'coverage_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
