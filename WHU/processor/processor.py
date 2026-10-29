import logging
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from loss.mixed_rank import mixed_rank_loss, budgeted_auxiliary_loss
from .sam_training import run_uad_step, sam_radius, validate_sam_config


def PCA(feat, target, num_modalities, group_size=4, fixed_centers=None,
        return_per_modality=False):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    if fixed_centers is None:
        g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    else:
        if fixed_centers.shape != centers[0].shape:
            raise ValueError('Fixed ProCA centers do not match identity groups')
        g_cent = fixed_centers.detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    if return_per_modality:
        per_modality = torch.stack([(center - g_cent).pow(2).sum(1).mean()
                                    for center in centers])
        return loss, ids, g_cent, per_modality
    return loss, ids, g_cent


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03,
        fixed_prototypes=None, return_prototypes=False):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        if fixed_prototypes is None:
            P = bank.prototypes.clone()
            norms = P.norm(dim=1)
            need = norms[ids] < 1e-6
            if need.any():
                P[ids[need]] = g_cent[need]
            P = F.normalize(P, dim=1)
        else:
            if fixed_prototypes.shape != bank.prototypes.shape:
                raise ValueError('Fixed GPD prototypes do not match bank')
            P = fixed_prototypes.detach()

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return (loss, P.detach().clone()) if return_prototypes else loss


def uad_forward_loss(model, imgs, target, camids, loss_fn, bank, group_size=4,
                     pca_weight=0.01, fixed_targets=None):
    """One UAD forward.  SAM reuses clean detached targets, never the graph."""
    modalities = len(imgs)
    target_rep = target.repeat(modalities)
    with amp.autocast(enabled=imgs[0].is_cuda):
        scores, raw, _ = model(imgs, target, camids)
        base, ce, wrt = loss_fn(scores, raw, target_rep)
        pc, ids, centers, pc_mod = PCA(
            raw, target, modalities, group_size,
            fixed_centers=None if fixed_targets is None else fixed_targets['centers'],
            return_per_modality=True)
        if fixed_targets is not None and not torch.equal(ids, fixed_targets['ids']):
            raise ValueError('SAM identity group order changed between passes')
        gp, prototypes = GPD(
            raw, target, bank, ids, centers, modalities, tau=0.03,
            fixed_prototypes=None if fixed_targets is None else fixed_targets['prototypes'],
            return_prototypes=True)
        total = base + pca_weight * pc + gp
        with torch.no_grad():
            metrics = dict(ce=float(ce.detach()), wrt=float(wrt.detach()),
                           proca=float(pc.detach()), gpd=float(gp.detach()),
                           acc=float((scores.argmax(1) == target_rep).float().mean()))
            # Reporting only: no extra network pass, RNG consumption or gradients.
            ce_mod = F.cross_entropy(scores, target_rep, reduction='none').reshape(modalities, -1).mean(1)
            gp_logits = (F.normalize(raw, dim=1) @ prototypes.t()) / 0.03
            gp_mod = F.cross_entropy(gp_logits, target_rep, reduction='none').reshape(modalities, -1).mean(1)
            for m in range(modalities):
                metrics['ce_mod{}'.format(m)] = float(ce_mod[m])
                metrics['gpd_mod{}'.format(m)] = float(gp_mod[m])
                metrics['proca_mod{}'.format(m)] = float(pc_mod[m].detach())
    targets = fixed_targets if fixed_targets is not None else dict(
        ids=ids.detach().clone(), centers=centers.detach().clone(),
        prototypes=prototypes.detach().clone())
    return total, targets, metrics


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(local_rank)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    if is_main:
        logger.info('Feature dimension: {}; identities: {}'.format(feat_dim, num_classes))
    if use_aux_modules:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=cfg.MODEL.GPD_MOMENTUM,
            device=device,
        )

    group_size = cfg.DATALOADER.NUM_INSTANCE
    rank_cfg = cfg.WCV1
    sam_cfg = cfg.SAM
    use_safe_uad = bool(sam_cfg.ENABLED or sam_cfg.SAFE_BASELINE)
    if use_safe_uad:
        validate_sam_config(cfg)
        if is_main:
            logger.info('UAD-SAM-B: enabled=%s safe_baseline=%s rho_max=%s start=%s ramp=%s scope=backbone fixed_targets=True replay_rng=True',
                        sam_cfg.ENABLED, sam_cfg.SAFE_BASELINE, sam_cfg.RHO_MAX,
                        sam_cfg.START_EPOCH, sam_cfg.RAMP_EPOCHS)
    total_steps, successful_steps = 0, 0
    model_meta.capture_wcv1_feature = bool(rank_cfg.ENABLED)
    if rank_cfg.ENABLED:
        if rank_cfg.TAU <= 0 or rank_cfg.RHO < 0 or rank_cfg.MAX_WEIGHT < 0:
            raise ValueError("Invalid WCV1 tau/rho/max_weight")
        if rank_cfg.START_EPOCH < 0 or rank_cfg.RAMP_EPOCHS <= 0:
            raise ValueError("Invalid WCV1 start/ramp epochs")
        if is_main:
            logger.info("WCV1 V2 enabled: tau=%s margin=%s rho=%s max_weight=%s start=%s ramp=%s gather=%s",
                        rank_cfg.TAU, rank_cfg.MARGIN, rank_cfg.RHO, rank_cfg.MAX_WEIGHT,
                        rank_cfg.START_EPOCH, rank_cfg.RAMP_EPOCHS, rank_cfg.GATHER_CANDIDATES)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        n_iter = 0
        epoch_success, epoch_skipped = 0, 0
        for n_iter, batch in enumerate(train_loader):
            if len(batch) == 4:
                imgs, vid, camids, sample_keys = batch
            else:
                imgs, vid, camids = batch
                sample_keys = None
            if rank_cfg.ENABLED and sample_keys is None:
                raise ValueError("WCV1 requires stable training sample keys")
            rank_stats = None
            sam_stats = None
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            if use_safe_uad:
                def forward_loss(fixed_targets=None):
                    return uad_forward_loss(model, imgs, target, camids, loss_fn, proto_bank,
                                            group_size, cfg.MODEL.PCA_LOSS_WEIGHT, fixed_targets)

                def commit_bank(targets):
                    proto_bank.update(targets['ids'], targets['centers'], ddp=dist_on)

                sam_stats = run_uad_step(
                    model, optimizer, scaler, forward_loss,
                    rho=sam_radius(epoch, sam_cfg), eps=sam_cfg.EPS,
                    commit=commit_bank, distributed=dist_on, max_grad_norm=1.0,
                    record_layer_stats=((n_iter + 1) % log_period == 0))
                total_steps += 1
                successful_steps += int(sam_stats['step_succeeded'])
                epoch_success += int(sam_stats['step_succeeded'])
                epoch_skipped += int(not sam_stats['step_succeeded'])
                clean = sam_stats['clean_metrics']
                for meter, key in zip(meter_ls, ('ce', 'wrt', 'proca', 'gpd')):
                    meter.update(clean[key], target_rep.numel())
                loss_meter.update(sam_stats['clean_loss'], target.numel())
                acc_meter.update(clean['acc'], 1)
            else:
                with amp.autocast(enabled=True):
                    cls_score, global_feat, feat = model(imgs, target, camids)
                    if rank_cfg.ENABLED:
                        # Query gradients at the true pre-BN ancestor, not a DDP
                        # output wrapper that may omit the CE branch from dL/dF.
                        global_feat = model_meta.wcv1_raw_feature

                    loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                    meter_ls[0].update(il.item(), target_rep.shape[0])
                    meter_ls[1].update(tl.item(), target_rep.shape[0])

                    if use_aux_modules:
                        loss_pca, ids, g_cent = PCA(
                            global_feat,
                            target,
                            num_modalities=num_modalities,
                            group_size=group_size,
                        )
                        meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                        loss_gpd = GPD(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=0.03,
                        )
                        meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                        if 'pca' in loss_type:
                            loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                        if 'gpd' in loss_type:
                            loss += loss_gpd

                if rank_cfg.ENABLED and epoch > rank_cfg.START_EPOCH:
                    if num_modalities != 3:
                        raise ValueError("WCV1 V2 expects RGB/NIR/TIR")
                    with amp.autocast(enabled=False):
                        modal_ids = torch.arange(num_modalities, device=global_feat.device).repeat_interleave(target.numel())
                        camera_flat = torch.cat(camids, dim=0)
                        key_flat = torch.cat([key.to(global_feat.device) for key in sample_keys], dim=0)
                        aux_loss, rank_stats = mixed_rank_loss(
                            global_feat, target_rep, camera_flat, modal_ids, key_flat,
                            distributed=dist_on, gather_candidates=rank_cfg.GATHER_CANDIDATES,
                            tau=rank_cfg.TAU, margin=rank_cfg.MARGIN,
                            num_modalities=num_modalities)
                        ramp = min(1.0, float(epoch - rank_cfg.START_EPOCH) / rank_cfg.RAMP_EPOCHS)
                        loss, budget_stats = budgeted_auxiliary_loss(
                            loss, aux_loss, global_feat, scaler.get_scale(), ramp,
                            rho=rank_cfg.RHO, max_weight=rank_cfg.MAX_WEIGHT,
                            distributed=dist_on)
                        rank_stats.update(budget_stats)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if use_aux_modules:
                    with torch.no_grad():
                        proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

                acc = (cls_score.max(1)[1] == target_rep).float().mean()
                loss_meter.update(loss.item(), target.shape[0])
                acc_meter.update(acc, 1)

            torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
                if sam_stats is not None:
                    flat = {k: v for k, v in sam_stats.items()
                            if k not in ('clean_metrics', 'adv_metrics')}
                    for phase in ('clean', 'adv'):
                        for k, v in (sam_stats[phase + '_metrics'] or {}).items():
                            flat[phase + '_' + k] = v
                    flat.update(epoch_success=epoch_success, epoch_skipped=epoch_skipped,
                                total_steps=total_steps, successful_steps=successful_steps)
                    logger.info('SAM Epoch[%s] %s', epoch,
                                ', '.join('{}={}'.format(k, v) for k, v in sorted(flat.items())))
                if rank_stats is not None:
                    logger.info('WCV1 Epoch[%s] %s', epoch,
                                ', '.join('{}={:.6g}'.format(k, v) for k, v in sorted(rank_stats.items())))

        end_time = time.time()
        if is_main and use_safe_uad:
            logger.info('SAM Epoch[%s] summary: success=%s skipped=%s total_steps=%s successful_steps=%s',
                        epoch, epoch_success, epoch_skipped, total_steps, successful_steps)
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            rank_rng_states = None
            if rank_cfg.ENABLED or use_safe_uad:
                rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state(local_rank))
                if dist_on:
                    rank_rng_states = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(rank_rng_states, rng)
                else:
                    rank_rng_states = [rng]
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )

            if is_main and (rank_cfg.ENABLED or use_safe_uad):
                # Archival state, separate from the model-only test.py artifact.
                # No automatic resume entrypoint is implied by this snapshot.
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(dict(
                    model=model_to_save.state_dict(), epoch=epoch,
                    optimizer=optimizer.state_dict(),
                    optimizer_center=optimizer_center.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    prototype_bank=proto_bank.state_dict() if use_aux_modules else None,
                    rng_by_rank=rank_rng_states, config=cfg.dump(),
                    sam_steps=dict(total=total_steps, successful=successful_steps),
                    checkpoint_phase='post_training_pre_evaluation'),
                    os.path.join(cfg.OUTPUT_DIR, 'training_state_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode)
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
