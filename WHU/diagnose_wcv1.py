"""No-retraining WCV1/UAD diagnosis. See WCV1_DIAGNOSTICS.md for server commands."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from diagnostics.cache import (cached_features, code_hashes, deterministic_reader, extract_model,
                               json_hash, load_model, make_config, make_records, save_json,
                               sha256_file)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages', default='extract,ranking,coverage,probes',
                   help='comma-separated subset: extract,ranking,coverage,probes')
    p.add_argument('--config', default='configs/WCV1.yml')
    p.add_argument('--data-root')
    p.add_argument('--dataset-dir', default='WHU-MARS')
    p.add_argument('--uad-checkpoint')
    p.add_argument('--wcv1-checkpoint')
    p.add_argument('--uad-archive', help='optional trusted training_state checkpoint, for historical bank only')
    p.add_argument('--wcv1-archive')
    p.add_argument('--pretrain', help='optional ImageNet file for provenance hash; full trained weights load strictly')
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--coverage-rounds', type=int, default=8)
    p.add_argument('--batches-per-round', type=int, default=8)
    p.add_argument('--probe-groups', type=int, default=16)
    p.add_argument('--uad-log')
    p.add_argument('--wcv1-log')
    p.add_argument('--tolerance-pp', type=float, default=.02)
    return p


def last_log_metrics(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    positions = [i for i, line in enumerate(lines) if 'Validation Results' in line]
    if not positions:
        raise ValueError('no validation results in log: ' + str(path))
    i = positions[-1]
    block = '\n'.join(lines[i + 1:i + 7])
    expressions = {'mAP': r'mAP:\s*([\d.]+)%', 'R1': r'Rank-1\s*:\s*([\d.]+)%',
                   'R5': r'Rank-5\s*:\s*([\d.]+)%', 'R10': r'Rank-10\s*:\s*([\d.]+)%'}
    values = {k: float(re.search(v, block).group(1)) / 100. for k, v in expressions.items()}
    epoch = re.search(r'Epoch:\s*(\d+)', lines[i])
    starts = [j for j, line in enumerate(lines[:i]) if 'Starting modality-pair specific evaluation' in line]
    tail = '\n'.join(lines[starts[-1]:i]) if starts else ''
    protocols = {'overall': values}
    names = ['RGB', 'NIR', 'TIR']
    for a, b, r1, ap, inp in re.findall(r'Pair \(Q:(\d), G:(\d)\).*?Rank-1: ([\d.]+)%, mAP: ([\d.]+)%, mINP: ([\d.]+)%', tail):
        protocols[names[int(a)-1] + '_to_' + names[int(b)-1]] = dict(R1=float(r1)/100, mAP=float(ap)/100, mINP=float(inp)/100)
    for direction, ap, r1, r10 in re.findall(r'(A->G|G->A) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-10: ([\d.]+)%', tail):
        protocols[direction.replace('->', '_to_')] = dict(mAP=float(ap)/100, R1=float(r1)/100, R10=float(r10)/100)
    return {'epoch': int(epoch.group(1)) if epoch else None, 'metrics': values, 'protocols': protocols}


def log_consistency(ranking, log_paths, tolerance_pp):
    checks = {}
    for label, path in log_paths.items():
        if not path:
            checks[label] = {'status': 'NOT_CHECKED', 'reason': 'no log supplied'}
            continue
        reference = last_log_metrics(path)
        diffs = {}
        for protocol, metrics in reference['protocols'].items():
            actual = ranking['protocols'][protocol][label]
            if actual is None:
                raise ValueError('logged protocol has no valid cached queries: ' + protocol)
            diffs.update({protocol + '/' + k: 100 * (actual[k] - v) for k, v in metrics.items()})
        checks[label] = {'status': 'PASS' if all(abs(v) <= tolerance_pp for v in diffs.values()) else 'MISMATCH',
                         'log_sha256': sha256_file(path), 'log_epoch': reference['epoch'],
                         'tolerance_pp': tolerance_pp, 'difference_pp': diffs}
    return checks


def render_summary(outdir, result):
    lines = ['# WCV1 免重训诊断运行报告', '',
             '本报告由缓存排名与固定状态梯度探针生成，未执行优化器更新。数值是本次诊断输出，不代表重新训练结果。', '',
             '## 完成阶段', '', ', '.join(result['completed_stages']), '',
             '## 解释边界', '',
             '- eval缓存的训练池覆盖是候选曝光代理，不是历史训练的逐步回放。',
             '- 终态梯度只说明当前模型；没有中期checkpoint就不能还原中期因果。',
             '- 重建bank与归档bank分别标识；仅归档bank也不等于完整现场恢复。',
             '- 正例覆盖同时输出per_anchor_unscaled和scaled_global（全局均值后）系数；两者均未乘预算λ，也不是参数梯度贡献。', '']
    if 'ranking' in result:
        lines += ['## 原始距离重算', '', '|协议|有效query|UAD mAP/R1/mINP %|WCV1 mAP/R1/mINP %|新增失败/修复|', '|---|---:|---|---|---|']
        for name, v in result['ranking']['protocols'].items():
            if v['status'] != 'ok':
                lines.append('|{}|0|SKIPPED|SKIPPED|—|'.format(name)); continue
            a, b = v['uad'], v['wcv1']
            fmt = lambda x: '/'.join('{:.4f}'.format(100*x[k]) for k in ['mAP', 'R1', 'mINP'])
            counts = v['transition_counts']
            lines.append('|{}|{}|{}|{}|{}/{}|'.format(name, v['valid_queries'], fmt(a), fmt(b), counts['new_failure'], counts['repaired']))
        lines += ['', '各协议查询明细、极值并集交叉分数与首尾联合变化见ranking目录。', '',
                  '日志一致性：', '', '```json', json.dumps(result.get('log_consistency', {}), ensure_ascii=False, indent=2), '```']
    if 'coverage' in result:
        lines += ['', '## 训练候选曝光', '', '```json', json.dumps(result['coverage'], ensure_ascii=False, indent=2), '```']
    if 'probes' in result:
        lines += ['', '## 固定状态梯度', '', '```json', json.dumps(result['probes'], ensure_ascii=False, indent=2), '```']
    lines += ['', '## 后续修正的判据（候选方向，不自动宣称因果）', '',
              '1. 新增失败以最佳正例变弱为主，且正例吸引占辅助主导：优先修订正例监督分配。',
              '2. 全库危险负身份很少进入池、batch正确而全库错误：优先改善负例曝光，保留随机覆盖。',
              '3. AG相关正例长期缺失：先修采样覆盖，再讨论loss权重。',
              '4. 总体cos为正而特定模态/末block明显冲突：定位对应监督，不直接增加全模型投影。',
              '5. 只有中期权重的同清单比较支持持续偏移时，才优先考虑后期衰减。',
              '以上需联合查看CSV与对照条件，再确定一个主修正版；诊断脚本不自动改训练代码。']
    (Path(outdir) / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    args = arguments().parse_args(argv)
    stages = set(args.stages.split(','))
    allowed = {'extract', 'ranking', 'coverage', 'probes'}
    if not stages or not stages <= allowed:
        raise ValueError('unknown or empty stage selection')
    for key in ['batch_size', 'chunk_size', 'coverage_rounds', 'batches_per_round', 'probe_groups']:
        if getattr(args, key) <= 0:
            raise ValueError(key + ' must be positive')
    if args.workers < 0 or args.tolerance_pp < 0:
        raise ValueError('invalid workers or tolerance')
    root = Path(__file__).resolve().parent
    cache, out = Path(args.cache_dir), Path(args.output_dir)
    cache.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'manifest.json'
    models, archives = {}, {}
    cfg = dataset = None
    result = {'completed_stages': [], 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
              'arguments': vars(args), 'no_optimizer_steps': True,
              'python_version': sys.version}
    needs_models = bool(stages & {'extract', 'probes'})
    if needs_models:
        if not all([args.data_root, args.uad_checkpoint, args.wcv1_checkpoint]):
            raise ValueError('extract/probes require data-root and both complete checkpoints')
        import torch
        result['execution_environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda, device=args.device)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; choose --device cpu for supported diagnostics')
        cfg = make_config(root / args.config, args.data_root, args.dataset_dir, args.pretrain)
        dataset, records = make_records(cfg)
        source = code_hashes(root)
        weights = {label: sha256_file(path) for label, path in [('uad', args.uad_checkpoint), ('wcv1', args.wcv1_checkpoint)]}
        provenance = dict(records=records, records_sha256=json_hash(records), config=cfg.dump(),
                          code_sha256=source, checkpoint_sha256=weights,
                          pretrain_sha256=sha256_file(args.pretrain) if args.pretrain else None,
                          dataset_dir=args.dataset_dir)
        fingerprints = {label: json_hash(dict(provenance=provenance, model=label)) for label in weights}
        # Root paths are provenance too. Relocation requires a new cache rather than unsafe silent reuse.
        manifest = dict(provenance, fingerprints=fingerprints,
                        environment=dict(torch=torch.__version__, cuda=torch.version.cuda), schema_version=1)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous['fingerprints'] != fingerprints:
                raise ValueError('cache fingerprint changed; use a new cache-dir')
        else:
            save_json(manifest_path, manifest)
        for label, checkpoint in [('uad', args.uad_checkpoint), ('wcv1', args.wcv1_checkpoint)]:
            models[label], bank = load_model(cfg, dataset, checkpoint, args.device)
            if bank is not None:
                archives[label] = bank
        # Optional separate training archive never replaces model weights.
        for label, path in [('uad', args.uad_archive), ('wcv1', args.wcv1_archive)]:
            if path:
                try:
                    state = torch.load(path, map_location='cpu', weights_only=False)
                except TypeError:
                    state = torch.load(path, map_location='cpu')
                archived_model = {k[7:] if k.startswith('module.') else k: v for k, v in state['model'].items()}
                loaded_model = models[label].state_dict()
                if set(archived_model) != set(loaded_model) or any(
                        not torch.equal(archived_model[k].cpu(), loaded_model[k].detach().cpu()) for k in loaded_model):
                    raise ValueError('archive model differs from selected checkpoint: ' + label)
                archives[label] = state['prototype_bank']['prototypes']
                result.setdefault('archive_sha256', {})[label] = sha256_file(path)
        for label, bank in archives.items():
            if list(bank.shape) != [dataset.num_train_pids, 768] or not torch.isfinite(bank).all():
                raise ValueError('invalid archived bank shape or values: ' + label)
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        records, fingerprints = manifest['records'], manifest['fingerprints']
    result['provenance'] = dict(manifest_path=str(manifest_path), fingerprints=fingerprints,
                                diagnostic_code_sha256=code_hashes(root))
    if 'extract' in stages:
        features = {label: extract_model(model, records, dataset.dataset_dir, cfg, cache, label,
                                        fingerprints[label], args.device, args.batch_size, args.workers)
                    for label, model in models.items()}
        result['completed_stages'].append('extract')
    else:
        features = cached_features(cache, records, fingerprints)
    if 'ranking' in stages:
        from diagnostics.ranking import run_ranking
        result['ranking'] = run_ranking(features['uad']['query'], features['uad']['gallery'],
                                        features['wcv1']['query'], features['wcv1']['gallery'],
                                        records['query'], records['gallery'], out / 'ranking', args.chunk_size)
        result['log_consistency'] = log_consistency(result['ranking'], {'uad': args.uad_log, 'wcv1': args.wcv1_log}, args.tolerance_pp)
        result['completed_stages'].append('ranking')
        save_json(out / 'run_summary.json', result)
        render_summary(out, result)
        if any(c['status'] == 'MISMATCH' for c in result['log_consistency'].values()):
            raise RuntimeError('cached ranking differs from supplied log; inspect report before causal diagnosis')
    if stages & {'coverage', 'probes'}:
        from diagnostics.coverage import generate_sampling_plan
        plan = generate_sampling_plan(records['train'], seed=args.seed, coverage_rounds=args.coverage_rounds,
                                      batches_per_round=args.batches_per_round, probe_batches=args.probe_groups)
        save_json(out / 'sampling_plan.json', plan)
    if 'coverage' in stages:
        from diagnostics.coverage import analyze_coverage
        result['coverage'] = analyze_coverage(features['uad']['train'], features['wcv1']['train'],
                                             records['train'], plan, out / 'coverage', args.chunk_size)
        result['completed_stages'].append('coverage')
        save_json(out / 'run_summary.json', result); render_summary(out, result)
    if 'probes' in stages:
        if (cfg.DATALOADER.NUM_INSTANCE != 4 or len(cfg.DATASETS.MODALITIES) != 3
                or set(cfg.SOLVER.LOSS_TYPE.lower().split('+')) != {'base', 'pca', 'gpd'}
                or cfg.MODEL.ID_LOSS_WEIGHT != 1 or cfg.MODEL.TRIPLET_LOSS_WEIGHT != 1
                or cfg.MODEL.PCA_LOSS_WEIGHT != .01 or cfg.MODEL.METRIC_LOSS_TYPE != 'wrt'
                or cfg.MODEL.IF_LABELSMOOTH != 'off'):
            raise ValueError('probe objective supports the recorded UAD CE/WRT/.01ProCA/GPD, K4, three-modality setup only')
        from diagnostics.probes import run_probes
        reader = deterministic_reader(cfg, dataset.dataset_dir, records['train'], args.seed)
        result['probes'] = run_probes(models, records['train'], plan['probe_batches'], reader,
                                      {label: value['train'] for label, value in features.items()},
                                      out / 'probes', args.device, archives, args.seed)
        result['completed_stages'].append('probes')
    save_json(out / 'run_summary.json', result)
    render_summary(out, result)
    print('Completed:', ', '.join(result['completed_stages']), 'Report:', out / 'report.md', flush=True)
    return result


if __name__ == '__main__':
    main()
