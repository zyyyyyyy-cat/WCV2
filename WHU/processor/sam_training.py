"""Backbone-scoped UAD SAM step, compatible with PyTorch 1.12.

The callback owns AMP forward/loss construction. Both backwards are ordinary
DDP backwards; clean gradients must therefore be synchronized (no no_sync()).
The callback must not update the prototype bank: commit(targets) does it once,
only after a successful outer optimizer step. No optimizer is created here.
"""
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm


def _rng(model):
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            {i: torch.cuda.get_rng_state(i).clone() for i in devices})


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    for device, value in state[3].items():
        torch.cuda.set_rng_state(value, device)


def _bn_state(model):
    return [(module, {name: value.detach().clone()
                     for name, value in module.named_buffers(recurse=False)
                     if value is not None})
            for module in model.modules() if isinstance(module, _BatchNorm)]


@torch.no_grad()
def _restore_bn(state):
    for module, buffers in state:
        for name, value in buffers.items():
            getattr(module, name).copy_(value)


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach(v) for v in value)
    if isinstance(value, list):
        return [_detach(v) for v in value]
    return value


def _all_finite(condition, device, distributed):
    flag = torch.tensor(0 if bool(condition) else 1, device=device, dtype=torch.int32)
    if distributed:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError('distributed=True requires an initialized process group')
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item()) == 0


def _finite_gradients(parameters, scale=1.0):
    checks = [torch.isfinite(p.grad.detach() / scale).all()
              for p in parameters if p.grad is not None]
    return bool(torch.stack(checks).all()) if checks else False


def _skip_scale(scaler, optimizer, parameters):
    # Mark overflow BEFORE the sole unscale so the public GradScaler API skips
    # every rank and resets the growth tracker. update(new_scale) does not reset
    # that tracker in torch 1.12. Do not flatten noncontiguous grads via copies.
    if not scaler.is_enabled():
        raise FloatingPointError('Nonfinite gradients in non-AMP UAD step')
    grad = next((p.grad for p in parameters if p.grad is not None and p.grad.numel()), None)
    if grad is None:
        raise RuntimeError('Cannot mark AMP overflow without a participating gradient')
    with torch.no_grad():
        grad[(0,) * grad.ndim] = float('inf')
    scaler.unscale_(optimizer)
    scaler.step(optimizer)  # found_inf is guaranteed on every rank; no SGD call.
    scaler.update()


def run_uad_step(model, optimizer, scaler, forward_loss, rho, eps=1e-12,
                 commit=None, distributed=False, max_grad_norm=1.0,
                 record_layer_stats=False):
    """Run one ordinary UAD or two-forward backbone-SAM optimizer step.

    forward_loss(fixed_targets=None) returns (loss, targets, metrics). Clean
    targets are detached/cloned and passed unchanged to adversarial forward.
    Loss must be a scalar; the callback must reuse already-augmented inputs.
    rho=0 is exactly one forward/backward with the same finite-step handling.

    Clean scaled gradients are read manually for perturbation, so GradScaler
    unscale_ is called only ONCE, on the final gradients. The perturbation scope
    is model[.module].base with non-None clean gradients. All optimizer
    parameters receive the final SGD update. DDP already averages gradients;
    do NOT reduce the backbone norm again across ranks.

    On failure BN rolls back to pre-clean and RNG resumes after clean. On
    success only clean BN statistics survive. Every perturbation is restored
    by copy_ before optimizer.step, including on exceptions. A commit callback
    failure AFTER the optimizer update is fatal and cannot undo optimizer state.
    """
    if not math.isfinite(float(rho)) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0:
        raise ValueError('max_grad_norm must be finite and positive')
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if not parameters:
        raise ValueError('optimizer has no parameters')
    device = parameters[0].device
    meta = model.module if hasattr(model, 'module') else model
    if rho > 0 and not hasattr(meta, 'base'):
        raise ValueError('SAM model must expose its backbone as .base')
    before_bn, before_rng = _bn_state(model), _rng(model)
    clean_rng, clean_bn = None, None
    backups = []
    committed = False
    scale = float(scaler.get_scale())
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError('AMP scale must be finite and positive')
    info = dict(rho=float(rho), grad_norm=0.0, perturbation_norm=0.0,
                step_succeeded=False, amp_scale=scale, skip_reason=None,
                clean_metrics=None, adv_metrics=None, clean_loss=None, adv_loss=None)

    @torch.no_grad()
    def restore_parameters():
        for parameter, original in backups:
            parameter.copy_(original)
        backups.clear()

    def skip(reason):
        info['skip_reason'] = reason
        restore_parameters()
        _restore_bn(before_bn)
        _restore_rng(clean_rng if clean_rng is not None else before_rng)
        _skip_scale(scaler, optimizer, parameters)
        optimizer.zero_grad(set_to_none=True)
        info['amp_scale_after'] = float(scaler.get_scale())
        return info

    try:
        optimizer.zero_grad(set_to_none=True)
        clean_loss, targets, clean_metrics = forward_loss()
        if clean_loss.ndim != 0:
            raise ValueError('forward_loss must return a scalar loss')
        targets = _detach(targets)
        info['clean_loss'] = float(clean_loss.detach())
        info['clean_metrics'] = _detach(clean_metrics)
        clean_rng, clean_bn = _rng(model), _bn_state(model)
        if not _all_finite(torch.isfinite(clean_loss.detach()), device, distributed):
            raise FloatingPointError('Nonfinite clean forward loss on at least one rank')
        scaler.scale(clean_loss).backward()
        if not _all_finite(_finite_gradients(parameters, scale), device, distributed):
            return skip('nonfinite_clean')

        if rho > 0:
            scope = [p for p in meta.base.parameters() if p.requires_grad and p.grad is not None]
            if not scope:
                raise RuntimeError('SAM found no participating backbone gradient')
            norms = torch.stack([(p.grad.detach().float() / scale).norm(2) for p in scope])
            norm = norms.double().norm(2)
            if not _all_finite(torch.isfinite(norm), device, distributed):
                raise FloatingPointError('Nonfinite backbone gradient norm')
            info['grad_norm'] = float(norm)
            with torch.no_grad():
                for parameter in scope:
                    original = parameter.detach().clone()
                    backups.append((parameter, original))
                    delta = (parameter.grad.detach().float() / scale) * (float(rho) / (norm + eps))
                    parameter.add_(delta.to(parameter.dtype))
                actual = torch.stack([(p.detach().float() - old.float()).norm(2) for p, old in backups]).norm(2)
                info['perturbation_norm'] = float(actual)
                if record_layer_stats:
                    names = {id(p): name for name, p in meta.base.named_parameters()}
                    groups = {}
                    for p, old in backups:
                        name = names[id(p)]
                        parts = name.split('.')
                        group = '.'.join(parts[:2]) if parts[0] == 'blocks' else parts[0]
                        delta_norm = (p.detach().float() - old.float()).norm().double()
                        weight_norm = old.float().norm().double()
                        groups.setdefault(group, []).append((delta_norm, weight_norm))
                    for group, values in groups.items():
                        delta_norm = torch.stack([v[0] for v in values]).norm()
                        weight_norm = torch.stack([v[1] for v in values]).norm()
                        info['perturb_abs_' + group] = float(delta_norm)
                        info['perturb_rel_' + group] = float(delta_norm / (weight_norm + eps))
            if not _all_finite(torch.stack([torch.isfinite(p).all() for p in scope]).all(), device, distributed):
                raise FloatingPointError('Nonfinite SAM perturbed parameters')
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(before_rng)
            adv_loss, _, adv_metrics = forward_loss(fixed_targets=targets)
            if adv_loss.ndim != 0:
                raise ValueError('forward_loss must return a scalar loss')
            info['adv_loss'] = float(adv_loss.detach())
            info['adv_metrics'] = _detach(adv_metrics)
            if not _all_finite(torch.isfinite(adv_loss.detach()), device, distributed):
                raise FloatingPointError('Nonfinite perturbed forward loss on at least one rank')
            scaler.scale(adv_loss).backward()
            final_loss = adv_loss
            restore_parameters()
            _restore_bn(clean_bn)
            _restore_rng(clean_rng)
        else:
            final_loss = clean_loss

        if not _all_finite(_finite_gradients(parameters, scale), device, distributed):
            return skip('nonfinite_final')
        scaler.unscale_(optimizer)
        clip_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        if not _all_finite(torch.isfinite(clip_norm) and _finite_gradients(parameters), device, distributed):
            raise FloatingPointError('Nonfinite final clipped gradient')
        info['optimizer_grad_norm'] = float(clip_norm)
        info['clip_factor'] = min(1.0, max_grad_norm / (float(clip_norm) + 1e-6))
        info['loss_increase'] = (info['adv_loss'] - info['clean_loss']) if rho > 0 else 0.0
        scaler.step(optimizer)
        committed = True
        scaler.update()
        info['step_succeeded'] = True
        info['amp_scale_after'] = float(scaler.get_scale())
        if commit is not None:
            commit(targets)
        return info
    finally:
        restore_parameters()
        if not committed:
            _restore_bn(before_bn)
            optimizer.zero_grad(set_to_none=True)
        # A failed/exceptional adversarial pass must not advance global RNG twice.
        _restore_rng(clean_rng if clean_rng is not None else before_rng)


def sam_radius(epoch, cfg):
    if not cfg.ENABLED or epoch <= cfg.START_EPOCH:
        return 0.0
    return float(cfg.RHO_MAX) * min(1.0, float(epoch - cfg.START_EPOCH) / cfg.RAMP_EPOCHS)


def validate_sam_config(cfg):
    """Reject unsupported combinations rather than silently changing the plan."""
    s = cfg.SAM
    if cfg.WCV1.ENABLED:
        raise ValueError('UAD-SAM-B / safe baseline and WCV1 are mutually exclusive')
    if s.SCOPE != 'backbone' or s.ADAPTIVE or s.STABLE or not s.FIX_STEP_TARGETS or not s.REPLAY_RNG:
        raise ValueError('Only backbone SAM, fixed targets, RNG replay, nonadaptive and nonstable are implemented')
    if not math.isfinite(s.RHO_MAX) or s.RHO_MAX < 0 or not math.isfinite(s.EPS) or s.EPS <= 0:
        raise ValueError('Invalid SAM radius or epsilon')
    if s.START_EPOCH < 0 or s.RAMP_EPOCHS <= 0:
        raise ValueError('Invalid SAM warmup/ramp')
    if (cfg.SOLVER.LOSS_TYPE.lower().strip() != 'base+pca+gpd' or
            cfg.MODEL.METRIC_LOSS_TYPE != 'wrt' or cfg.MODEL.IF_LABELSMOOTH != 'off' or
            cfg.MODEL.ID_LOSS_WEIGHT != 1.0 or cfg.MODEL.TRIPLET_LOSS_WEIGHT != 1.0 or
            cfg.MODEL.PCA_LOSS_WEIGHT != 0.01 or cfg.MODEL.GPD_MOMENTUM != 0.2):
        raise ValueError('UAD-SAM-B requires original CE/WRT/0.01 ProCA/GPD weights and bank momentum')
    if (list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal'] or
            cfg.DATALOADER.SAMPLER.upper() != 'PKM' or cfg.DATALOADER.NUM_INSTANCE != 4):
        raise ValueError('UAD-SAM-B requires RGB/IR/Thermal PKM K=4')
    if cfg.SOLVER.OPTIMIZER_NAME != 'SGD':
        raise ValueError('UAD-SAM-B requires SGD')
    if cfg.TEST.NECK_FEAT != 'before' or str(cfg.TEST.FEAT_NORM).lower() != 'yes' or cfg.TEST.RE_RANKING or cfg.TEST.TOP_K_EVAL != 0:
        raise ValueError('UAD-SAM-B requires normalized pre-BN CLS, complete gallery and no reranking')
