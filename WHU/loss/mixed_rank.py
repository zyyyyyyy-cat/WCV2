"""V2 mixed-gallery ranking; compatible with torch 1.12 (no autograd gather)."""
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp


def _distributed(enabled):
    return enabled and dist.is_available() and dist.is_initialized()


def _sum(value, enabled):
    if _distributed(enabled):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def mixed_rank_loss(features, pids, camids, modalities, sample_keys,
                    distributed=True, tau=0.07, margin=0.0, num_modalities=3,
                    gather_candidates=True):
    """Return rank-local DDP-correct objective and globally reduced diagnostics.

    Metadata are equal-length int64 tensors on the feature device. Modality
    codes are zero based. Only candidates are detached; every local anchor is
    differentiable. Candidate duplicates keep the first (rank, local index).
    """
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError('tau must be finite and positive')
    if num_modalities <= 0 or not math.isfinite(float(margin)):
        raise ValueError('invalid modality count or margin')
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('expected a nonempty [N,D] feature batch')
    meta = torch.stack([x.to(device=features.device, dtype=torch.long)
                        for x in (pids, camids, modalities, sample_keys)], 1)
    if meta.shape != (len(features), 4):
        raise ValueError('metadata must have one item per feature')
    world = dist.get_world_size() if _distributed(distributed) else 1
    with amp.autocast(enabled=False):
        z = F.normalize(features.float(), dim=1)
        candidate, candidate_meta = z.detach().contiguous(), meta.contiguous()
        if world > 1 and gather_candidates:
            # Check sizes before feature gather to avoid silently padded negatives.
            shape = torch.tensor(list(z.shape), device=z.device, dtype=torch.long)
            shapes = [torch.empty_like(shape) for _ in range(world)]
            dist.all_gather(shapes, shape)
            if any(not torch.equal(s, shape) for s in shapes):
                raise ValueError('mixed rank requires equal per-rank feature shapes')
            zs = [torch.empty_like(candidate) for _ in range(world)]
            ms = [torch.empty_like(candidate_meta) for _ in range(world)]
            dist.all_gather(zs, candidate)
            dist.all_gather(ms, candidate_meta)
            candidate, candidate_meta = torch.cat(zs), torch.cat(ms)
        # One small metadata transfer, deterministic and identical on all ranks.
        seen, keep = {}, []
        for i, (pid, cam, mod, key) in enumerate(candidate_meta.cpu().tolist()):
            if not 0 <= mod < num_modalities:
                raise ValueError('modality outside configured zero-based range')
            if key in seen:
                if seen[key] != (pid, cam, mod):
                    raise ValueError('sample key collision or inconsistent metadata')
            else:
                seen[key] = (pid, cam, mod)
                keep.append(i)
        idx = torch.tensor(keep, device=z.device, dtype=torch.long)
        candidate, cm = candidate[idx], candidate_meta[idx]
        logits = (z @ candidate.t()) / tau
        camera_ok = meta[:, 1, None] != cm[None, :, 1]
        distinct = meta[:, 3, None] != cm[None, :, 3]
        positive = camera_ok & distinct & (meta[:, 0, None] == cm[None, :, 0])
        negative = camera_ok & distinct & (meta[:, 0, None] != cm[None, :, 0])
        valid = positive.any(1) & negative.any(1)
        # Empty rows never enter logsumexp. Graph-connected zeros support ranks
        # with no valid anchors while other ranks still perform backward.
        local = z.sum() * 0.0
        counts = torch.stack([(valid & (meta[:, 2] == m)).sum()
                              for m in range(num_modalities)]).float()
        global_counts = _sum(counts.clone(), distributed)
        active = int((global_counts > 0).sum().item())
        stats_sum = z.new_zeros(6)
        if bool(valid.any()):
            l = logits[valid]
            p, n = positive[valid], negative[valid]
            v = torch.logsumexp(l.masked_fill(~n, float('-inf')), 1)
            terms = F.softplus(v[:, None] - l + margin / tau)
            grouped, groups = l.new_zeros(len(l)), l.new_zeros(len(l))
            for m in range(num_modalities):
                pm = p & (cm[None, :, 2] == m)
                pc = pm.sum(1)
                grouped = grouped + (terms * pm).sum(1) / pc.clamp_min(1)
                groups = groups + (pc > 0).float()
            ell = grouped / groups.clamp_min(1)
            anchor_mod = meta[valid, 2]
            if active:
                for m in range(num_modalities):
                    if global_counts[m] > 0:
                        local = local + world * ell[anchor_mod == m].sum() / (active * global_counts[m])
            sat = torch.sigmoid(v[:, None] - l + margin / tau)
            stats_sum = torch.stack([ell.detach().sum(), valid.sum().float(),
                                     p.sum().float(), n.sum().float(),
                                     ((sat > .95) & p).sum().float(), p.sum().float()])
        _sum(stats_sum, distributed)
        log_loss = _sum(local.detach().clone(), distributed) / world
        nv = max(float(stats_sum[1]), 1.0)
        diag = dict(aux_loss=float(log_loss), valid_fraction=float(stats_sum[1]) / (len(z) * world),
                    positive_count=float(stats_sum[2]) / nv, negative_count=float(stats_sum[3]) / nv,
                    candidates=len(keep), saturation=float(stats_sum[4]) / max(float(stats_sum[5]), 1.0))
        diag.update({'valid_mod{}'.format(m): float(global_counts[m]) for m in range(num_modalities)})
        return local, diag


def budgeted_auxiliary_loss(base_loss, aux_loss, features, scale, ramp,
                           rho=0.1, max_weight=1.0, distributed=True):
    """Probe real intermediate F, then retain normal final backward to parameters.

    This is not autograd.grad on DDP parameters. Probe gradients never populate
    parameter .grad. The ratio controls feature energy, not parameter updates.
    """
    if not (math.isfinite(float(scale)) and scale > 0):
        raise ValueError('gradient scale must be finite and positive')
    if not (math.isfinite(float(ramp)) and 0 <= ramp <= 1
            and math.isfinite(float(rho)) and rho >= 0
            and math.isfinite(float(max_weight)) and max_weight >= 0):
        raise ValueError('invalid ramp, rho or max_weight')
    g0 = torch.autograd.grad(base_loss * scale, features, retain_graph=True)[0].float() / scale
    ga = torch.autograd.grad(aux_loss * scale, features, retain_graph=True)[0].float() / scale
    with torch.no_grad():
        e0, ea, dot = g0.square().sum(), ga.square().sum(), (g0 * ga).sum()
        ok = (torch.isfinite(g0).all() & torch.isfinite(ga).all()
              & torch.isfinite(e0) & torch.isfinite(ea) & torch.isfinite(dot))
        values = torch.stack([e0, ea, dot, (~ok).float()])
        _sum(values, distributed)
        finite = bool(torch.isfinite(values).all() and values[3] == 0)
        e0, ea, dot = [float(x) for x in values[:3]]
        weight = 0.0
        if finite and e0 > 0 and ea > 0:
            weight = float(ramp) * min(float(max_weight), float(rho) * math.sqrt(e0 / max(ea, 1e-24)))
        ratio = weight * math.sqrt(ea / e0) if weight > 0 else 0.0
        cosine = dot / math.sqrt(e0 * ea) if finite and e0 > 0 and ea > 0 else 0.0
    # Never include a poisoned graph via zero * NaN.
    combined = base_loss + weight * aux_loss if finite and weight > 0 else base_loss
    return combined, dict(weight=weight, grad_ratio=ratio, grad_cosine=cosine, finite=finite)
