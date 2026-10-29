"""Read-only FP32 probes. Sequential local-BN simulation, never optimizer steps.

Reconstructed banks and terminal checkpoints are diagnostic proxies, not a
reconstruction of historical optimizer/AMP/DDP state.
"""
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def rebuild_bank(features, records):
    """Unit sample -> modality mean/unit -> equal-modality mean/unit."""
    x = F.normalize(torch.as_tensor(np.asarray(features).copy()).float(), dim=1)
    pids = torch.tensor([r['pid'] for r in records])
    mods = torch.tensor([r['modality'] for r in records])
    bank = torch.zeros(int(pids.max()) + 1, x.shape[1])
    for pid in pids.unique():
        centers = [F.normalize(x[(pids == pid) & (mods == m)].mean(0), dim=0)
                   for m in mods[pids == pid].unique()]
        bank[pid] = F.normalize(torch.stack(centers).mean(0), dim=0)
    return bank


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _buffers(model):
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def _restore_buffers(model, state):
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(state[n])


def _vector(grads, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                      for g, p in zip(grads, params)])


def _cos(a, b):
    den = float(a.norm() * b.norm())
    return float(a.dot(b)) / den if den > 0 else None


def _aux_parts(feat, candidate, local_meta, candidate_meta, global_counts):
    """Full, positive-only gradient and negative-only gradient of SAME loss."""
    z = F.normalize(feat.float(), dim=1)
    logits = z @ candidate.T / .07
    pid, cam, mod, key = local_meta.T
    cp, cc, cm, ck = candidate_meta.T
    mask = (cam[:, None] != cc) & (key[:, None] != ck)
    pos = mask & (pid[:, None] == cp)
    neg = mask & (pid[:, None] != cp)
    valid = pos.any(1) & neg.any(1)
    zero = feat.sum() * 0
    totals = [zero, zero, zero]
    if not valid.any():
        return totals
    l, p, n = logits[valid], pos[valid], neg[valid]
    v = torch.logsumexp(l.masked_fill(~n, -float('inf')), dim=1)
    matrices = [F.softplus(v[:, None] - l),
                F.softplus(v.detach()[:, None] - l),
                F.softplus(v[:, None] - l.detach())]
    active = int((global_counts > 0).sum())
    for t, matrix in enumerate(matrices):
        ell = torch.zeros(len(l), device=l.device)
        groups = torch.zeros_like(ell)
        for m in range(3):
            pm = p & (cm[None] == m)
            count = pm.sum(1)
            ell = ell + (matrix * pm).sum(1) / count.clamp_min(1)
            groups += (count > 0).float()
        ell = ell / groups.clamp_min(1)
        for m in range(3):
            if global_counts[m] > 0:
                # rank-local W=4 loss; parameter average below divides by four.
                totals[t] = totals[t] + 4 * ell[mod[valid] == m].sum() / (active * global_counts[m])
    return totals


def _candidate_data(features, meta):
    keep, seen = [], set()
    for i, key in enumerate(meta[:, 3].tolist()):
        if key not in seen:
            keep.append(i); seen.add(key)
    c, cm = features[keep], meta[keep]
    mask = (meta[:, 1, None] != cm[None, :, 1]) & (meta[:, 3, None] != cm[None, :, 3])
    valid = (mask & (meta[:, 0, None] == cm[None, :, 0])).any(1) & (mask & (meta[:, 0, None] != cm[None, :, 0])).any(1)
    counts = torch.stack([(valid & (meta[:, 2] == m)).sum() for m in range(3)])
    return c, cm, counts


def run_probes(models, records, probe_batches, load_batch, train_features_by_model,
               outdir, device='cuda', archived_banks=None, seed=1234):
    """load_batch(indices, seed) supplies identical augmented images to each model.

    probe_batches shape [groups,4,192] (tests may use smaller valid PKM batches).
    Gradients are measured at original F and averaged last-block parameters.
    Each bank variant is explicitly labeled. No batch or bank state is retained.
    """
    from processor.processor import PCA, GPD
    from loss.wrt_loss import TripletLoss_WRT
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    archived_banks = archived_banks or {}
    rows, state_checks = [], {}
    modes = ['ce', 'wrt', 'proca_weighted', 'gpd', 'base', 'aux', 'positive', 'negative']
    initial_rng = _rng()
    for model_name, model in models.items():
        before = state_hash(model)
        buffs = _buffers(model)
        training_flags = {n: m.training for n, m in model.named_modules()}
        capture_old = getattr(model, 'capture_wcv1_feature', None)
        raw_existed = hasattr(model, 'wcv1_raw_feature')
        raw_old = getattr(model, 'wcv1_raw_feature', None)
        last = model.base.blocks[-1] if hasattr(model, 'base') and hasattr(model.base, 'blocks') else model.backbone
        params = [p for p in last.parameters() if p.requires_grad]
        banks = {'reconstructed': rebuild_bank(train_features_by_model[model_name], records)}
        if model_name in archived_banks:
            banks['archived'] = torch.as_tensor(archived_banks[model_name]).float()
        try:
            model.train(); model.capture_wcv1_feature = True
            for group_id, group in enumerate(probe_batches):
                if len(group) != 4:
                    raise ValueError('probe batch must contain exactly four local ranks')
                # Load deterministic tensors once, reused for every bank and model.
                inputs = [load_batch(ix, seed + group_id * 4 + rank).cpu() for rank, ix in enumerate(group)]
                metas = [torch.tensor([[records[i][k] for k in ['pid', 'camid', 'modality', 'sample_key']]
                                      for i in ix], dtype=torch.long, device=device) for ix in group]
                caches, replay = [], []
                for rank, (images, meta) in enumerate(zip(inputs, metas)):
                    _restore_buffers(model, buffs)
                    random.seed(seed + group_id * 4 + rank); np.random.seed(seed + group_id * 4 + rank)
                    torch.manual_seed(seed + group_id * 4 + rank)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + group_id * 4 + rank)
                    replay.append(_rng())
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                        chunks = images.to(device).chunk(3)
                        _, f, _ = model(chunks, meta[:len(meta)//3, 0], None)
                        caches.append(F.normalize(f.float(), dim=1).detach())
                candidate, candidate_meta, counts = _candidate_data(torch.cat(caches), torch.cat(metas))
                for bank_label, bank in banks.items():
                    bank = bank.to(device)
                    accum = {k: None for k in modes}
                    fgrads = {k: [] for k in modes}
                    fmods, values = [], {k: [] for k in modes}
                    for rank, (images, meta) in enumerate(zip(inputs, metas)):
                        _restore_buffers(model, buffs); _restore_rng(replay[rank])
                        with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                            n = len(meta)//3
                            if n % 4:
                                raise ValueError('local first-modality length must be divisible by K=4')
                            score, returned, unused_bn = model(images.to(device).chunk(3), meta[:n, 0], None)
                            f = getattr(model, 'wcv1_raw_feature', returned)
                            torch.testing.assert_close(F.normalize(f.detach(), dim=1), caches[rank])
                            ce = F.cross_entropy(score, meta[:, 0])
                            wrt = TripletLoss_WRT()(f, meta[:, 0])[0]
                            pca, ids, centers = PCA(f, meta[:n, 0], 3, group_size=4)
                            gpd = GPD(f, meta[:n, 0], SimpleNamespace(prototypes=bank), ids, centers, 3)
                            full, positive, negative = _aux_parts(f, candidate, meta, candidate_meta, counts)
                            losses = [ce, wrt, .01*pca, gpd, ce+wrt+.01*pca+gpd, full, positive, negative]
                            for k, loss in zip(modes, losses):
                                if not bool(torch.isfinite(loss)):
                                    raise FloatingPointError('nonfinite probe loss: %s/%s/group%s/%s' % (model_name, bank_label, group_id, k))
                                gf = torch.autograd.grad(loss, f, retain_graph=True)[0].detach().float()
                                gp = _vector(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)
                                if not bool(torch.isfinite(gf).all() and torch.isfinite(gp).all()):
                                    raise FloatingPointError('nonfinite probe gradient: %s/%s/group%s/%s' % (model_name, bank_label, group_id, k))
                                fgrads[k].append(gf.cpu()); values[k].append(float(loss.detach()))
                                accum[k] = gp / 4 if accum[k] is None else accum[k] + gp / 4
                            fmods.append(meta[:, 2].cpu())
                        if hasattr(model, 'wcv1_raw_feature'): model.wcv1_raw_feature = None
                        del score, f, returned, unused_bn, losses, loss, gf, gp
                        del ce, wrt, pca, gpd, full, positive, negative, ids, centers
                    fg = {k: torch.cat(v).reshape(-1) for k, v in fgrads.items()}
                    e0, ea = float(fg['base'].square().sum()), float(fg['aux'].square().sum())
                    weight = min(1., .1*math.sqrt(e0/max(ea, 1e-24))) if e0 > 0 and ea > 0 else 0.
                    fm = torch.cat(fmods)
                    for scope in ['feature_all', 'feature_mod0', 'feature_mod1', 'feature_mod2', 'last_block_parameter']:
                        vecs = accum if scope == 'last_block_parameter' else (
                            fg if scope == 'feature_all' else {k: torch.cat(fgrads[k])[fm == int(scope[-1])].reshape(-1) for k in modes})
                        for a in modes:
                            for b in modes:
                                if modes.index(b) < modes.index(a): continue
                                rows.append(dict(model=model_name, bank=bank_label, group=group_id, scope=scope,
                                                 loss_a=a, loss_b=b, norm_a=float(vecs[a].norm()), norm_b=float(vecs[b].norm()),
                                                 cosine=_cos(vecs[a], vecs[b]), budget_weight=weight,
                                                 mean_loss_a=sum(values[a])/4, mean_loss_b=sum(values[b])/4))
                    # Direct regression check: split gradients sum to original auxiliary.
                    torch.testing.assert_close(fg['aux'], fg['positive'] + fg['negative'], atol=1e-5, rtol=1e-4)
                    del accum, fgrads, fg
                # A group is a read-only transaction, not merely restored
                # before the next forward or when the entire run finishes.
                _restore_buffers(model, buffs)
                _restore_rng(initial_rng)
        finally:
            _restore_buffers(model, buffs)
            for n, m in model.named_modules(): m.training = training_flags[n]
            if capture_old is None: delattr(model, 'capture_wcv1_feature')
            else: model.capture_wcv1_feature = capture_old
            if raw_existed: model.wcv1_raw_feature = raw_old
            elif hasattr(model, 'wcv1_raw_feature'): delattr(model, 'wcv1_raw_feature')
            _restore_rng(initial_rng)
            after = state_hash(model)
            state_checks[model_name] = {'before_sha256': before, 'after_sha256': after, 'unchanged': before == after}
            if before != after: raise RuntimeError('probe mutated persistent model state')
    with (outdir/'gradient_pairs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['model']); w.writeheader(); w.writerows(rows)
    summary = dict(groups=len(probe_batches), rows=len(rows), state_checks=state_checks,
                   precision='FP32', mode='sequential four-local-BN simulation; no optimizer or bank update',
                   limits='terminal gradient snapshots; reconstructed bank is not historical training state')
    (outdir/'gradient_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
