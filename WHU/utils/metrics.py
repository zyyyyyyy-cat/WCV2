import csv
import os

import torch
import numpy as np
from utils.reranking import re_ranking
import torch.distributed as dist
from collections import defaultdict

DISTMAT_Q_CHUNK = 4000

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu()

def iter_euclidean_distance_chunks(qf, gf, q_chunk_size=DISTMAT_Q_CHUNK):
    num_q = qf.shape[0]
    gf_t = gf.t().contiguous()
    gf_sq = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()

    for start in range(0, num_q, q_chunk_size):
        end = min(start + q_chunk_size, num_q)
        q_chunk = qf[start:end]
        dist_chunk = torch.pow(q_chunk, 2).sum(dim=1, keepdim=True) + gf_sq
        dist_chunk.addmm_(q_chunk, gf_t, beta=1, alpha=-2)
        yield dist_chunk.cpu()

def compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK):
    indices_chunks = []
    for dist_chunk in iter_euclidean_distance_chunks(qf, gf, q_chunk_size=q_chunk_size):
        if top_k:
            k = min(top_k, dist_chunk.shape[1])
            idx_chunk = torch.topk(dist_chunk, k=k, dim=1, largest=False).indices
        else:
            idx_chunk = torch.argsort(dist_chunk, dim=1)
        indices_chunks.append(idx_chunk.numpy().astype(np.int32))
        del dist_chunk

    if not indices_chunks:
        width = min(top_k, gf.shape[0]) if top_k else gf.shape[0]
        return np.empty((0, width), dtype=np.int32)
    return np.concatenate(indices_chunks, axis=0)

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def top_k_indices(distmat, top_k):
    num_q, num_g = distmat.shape
    k = min(top_k, num_g)
    if k <= 0:
        return np.empty((num_q, 0), dtype=np.int32)
    top_k_idx = np.argpartition(distmat, kth=k - 1, axis=1)[:, :k]
    top_k_dist = np.take_along_axis(distmat, top_k_idx, axis=1)
    sorted_order = np.argsort(top_k_dist, axis=1)
    indices = np.take_along_axis(top_k_idx, sorted_order, axis=1)
    return indices.astype(np.int32)

def eval_func_top_k(indices, q_pids, g_pids, q_camids, g_camids, q_mids, g_mids, max_rank=50):
    """Evaluation with market1501 metric, optimized for large datasets by limiting to top_k gallery samples per query."""
    num_q, num_g = indices.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_mid = q_mids[q_idx]
        
        valid_gallery = (g_pids == q_pid) & (g_camids != q_camid)
        num_rel = np.sum(valid_gallery)
        if num_rel == 0:
            # this condition is true when query identity does not appear in gallery
            continue
        
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # market1501 metric
        remove = (g_camids[order] == q_camid) # sysu metric
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if orig_cmc.size == 0:
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if cmc.shape[0] < max_rank:
            pad_width = max_rank - cmc.shape[0]
            if cmc.shape[0] > 0:
                cmc = np.pad(cmc, (0, pad_width), mode='edge')
            else:
                cmc = np.pad(cmc, (0, pad_width), mode='constant')
        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP

def eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50,
              return_minp=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = indices.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # market1501 metric
        remove = (g_camids[order] == q_camid) # sysu dataset
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        # Precision at the last relevant gallery position (mINP protocol).
        last_relevant = np.nonzero(orig_cmc)[0][-1]
        all_INP.append(num_rel / (last_relevant + 1.0))

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    if return_minp:
        return all_cmc, mAP, mINP
    return all_cmc, mAP

_gloo_pg = None
def get_gloo_group():
    global _gloo_pg
    if _gloo_pg is None and dist.is_initialized() and dist.get_world_size() > 1:
        _gloo_pg = dist.new_group(backend="gloo")
    return _gloo_pg

MODALITY_NAMES = {1: "RGB", 2: "NIR", 3: "TIR"}


def _domain_protocol_definitions():
    d = {"RGB-G": (1, "G"), "RGB-A": (1, "A"),
         "NIR-G": (2, "G"), "NIR-A": (2, "A"),
         "TIR-G": (3, "G"), "TIR-A": (3, "A")}
    pairs = [
        ("same_modality_same_platform", "RGB-G", "RGB-G"),
        ("same_modality_same_platform", "RGB-A", "RGB-A"),
        ("same_modality_same_platform", "NIR-G", "NIR-G"),
        ("same_modality_same_platform", "NIR-A", "NIR-A"),
        ("same_modality_same_platform", "TIR-G", "TIR-G"),
        ("same_modality_same_platform", "TIR-A", "TIR-A"),
        ("modality_only_ground", "RGB-G", "NIR-G"),
        ("modality_only_ground", "NIR-G", "RGB-G"),
        ("modality_only_ground", "RGB-G", "TIR-G"),
        ("modality_only_ground", "TIR-G", "RGB-G"),
        ("modality_only_ground", "NIR-G", "TIR-G"),
        ("modality_only_ground", "TIR-G", "NIR-G"),
        ("modality_only_aerial", "RGB-A", "NIR-A"),
        ("modality_only_aerial", "NIR-A", "RGB-A"),
        ("modality_only_aerial", "RGB-A", "TIR-A"),
        ("modality_only_aerial", "TIR-A", "RGB-A"),
        ("modality_only_aerial", "NIR-A", "TIR-A"),
        ("modality_only_aerial", "TIR-A", "NIR-A"),
        ("platform_only", "RGB-G", "RGB-A"),
        ("platform_only", "RGB-A", "RGB-G"),
        ("platform_only", "NIR-G", "NIR-A"),
        ("platform_only", "NIR-A", "NIR-G"),
        ("platform_only", "TIR-G", "TIR-A"),
        ("platform_only", "TIR-A", "TIR-G"),
        ("modality_and_platform_ground_to_aerial", "RGB-G", "NIR-A"),
        ("modality_and_platform_ground_to_aerial", "RGB-G", "TIR-A"),
        ("modality_and_platform_ground_to_aerial", "NIR-G", "RGB-A"),
        ("modality_and_platform_ground_to_aerial", "NIR-G", "TIR-A"),
        ("modality_and_platform_ground_to_aerial", "TIR-G", "RGB-A"),
        ("modality_and_platform_ground_to_aerial", "TIR-G", "NIR-A"),
        ("modality_and_platform_aerial_to_ground", "RGB-A", "NIR-G"),
        ("modality_and_platform_aerial_to_ground", "RGB-A", "TIR-G"),
        ("modality_and_platform_aerial_to_ground", "NIR-A", "RGB-G"),
        ("modality_and_platform_aerial_to_ground", "NIR-A", "TIR-G"),
        ("modality_and_platform_aerial_to_ground", "TIR-A", "RGB-G"),
        ("modality_and_platform_aerial_to_ground", "TIR-A", "NIR-G")]
    return [(cat, qn, gn, *d[qn], *d[gn]) for cat, qn, gn in pairs]


def _platform_mask(camids, platform):
    return camids < 5 if platform == "G" else camids >= 5


def _count_valid_queries(q_pids, q_camids, g_pids, g_camids):
    return sum(bool(np.any((g_pids == pid) & (g_camids != cam)))
               for pid, cam in zip(q_pids, q_camids))


def evaluate_domain_protocols(qf, gf, q_pids, q_camids, q_mids,
                              g_pids, g_camids, g_mids, max_rank,
                              reranking=False, distmat=None, logger=None,
                              csv_path=None, overall_cmc=None, overall_mAP=None):
    rows = [{
        "category": "overall_official", "query_domain": "ALL",
        "gallery_domain": "ALL", "query_count": len(q_pids),
        "gallery_count": len(g_pids),
        "valid_query_count": _count_valid_queries(q_pids, q_camids, g_pids, g_camids),
        "rank1": float(overall_cmc[0]), "mAP": float(overall_mAP),
        "mINP": "", "status": "ok"}]
    if logger:
        logger.info("Detailed modality-platform evaluation (same-camera gallery samples excluded for every query):")
        logger.info("  Category                                  Query->Gallery       Query Gallery ValidQ  Rank-1     mAP    mINP")
    for cat, qn, gn, qm, qp, gm, gp in _domain_protocol_definitions():
        qi = np.where((q_mids == qm) & _platform_mask(q_camids, qp))[0]
        gi = np.where((g_mids == gm) & _platform_mask(g_camids, gp))[0]
        row = {"category": cat, "query_domain": qn, "gallery_domain": gn,
               "query_count": len(qi), "gallery_count": len(gi),
               "valid_query_count": 0, "rank1": "", "mAP": "", "mINP": "",
               "status": "ok"}
        if not len(qi) or not len(gi):
            row["status"] = "empty_query_or_gallery"
        else:
            qpid, qcam = q_pids[qi], q_camids[qi]
            gpid, gcam = g_pids[gi], g_camids[gi]
            row["valid_query_count"] = _count_valid_queries(qpid, qcam, gpid, gcam)
            if not row["valid_query_count"]:
                row["status"] = "no_valid_query_after_same_camera_exclusion"
            else:
                if reranking:
                    indices = np.argsort(distmat[qi][:, gi], axis=1).astype(np.int32)
                else:
                    qti, gti = torch.from_numpy(qi).long(), torch.from_numpy(gi).long()
                    indices = compute_indices_chunked(qf.index_select(0, qti),
                        gf.index_select(0, gti), top_k=0,
                        q_chunk_size=DISTMAT_Q_CHUNK)
                cmc, ap, inp = eval_func(indices, qpid, gpid, qcam, gcam,
                                         max_rank=max_rank, return_minp=True)
                row.update(rank1=float(cmc[0]), mAP=float(ap), mINP=float(inp))
        rows.append(row)
        if logger:
            if row["status"] == "ok":
                logger.info(f"  {cat:<41} {qn:>5}->{gn:<5} {len(qi):7d} {len(gi):7d} {row['valid_query_count']:6d} {row['rank1']:7.2%} {row['mAP']:7.2%} {row['mINP']:7.2%}")
            else:
                logger.info(f"  {cat:<41} {qn:>5}->{gn:<5} {len(qi):7d} {len(gi):7d} {row['valid_query_count']:6d}  SKIPPED ({row['status']})")
    if csv_path:
        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = ["category", "query_domain", "gallery_domain", "query_count",
                  "gallery_count", "valid_query_count", "rank1", "mAP", "mINP", "status"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        logger and logger.info(f"=> Detailed protocol CSV saved to {csv_path}")
    return rows


class R1_mAP_eval:
    def __init__(self, max_rank=50, feat_norm=True, reranking=False,
                 top_k=50000, logger=None, detailed_domain_eval=False,
                 domain_report_path=None):
        self.max_rank   = max_rank
        self.feat_norm  = feat_norm
        self.reranking  = reranking
        self.top_k      = top_k
        self.logger     = logger
        self.detailed_domain_eval = detailed_domain_eval
        self.domain_report_path = domain_report_path
        self.reset()

    def reset(self):
        self.feats   = defaultdict(list)
        self.pids    = defaultdict(list)
        self.camids  = defaultdict(list)
        self.modids  = defaultdict(list)
        self.query_nums = {}

        self.qf, self.gf = [], []
        self.q_pids, self.q_camids, self.q_mids = [], [], []
        self.g_pids, self.g_camids, self.g_mids = [], [], []

    def update(self, output, mode: int):
        feat, pid, camid, modid = output
        self.feats[mode].append(feat.cpu())
        self.pids[mode].extend(np.asarray(pid))
        self.camids[mode].extend(np.asarray(camid))
        self.modids[mode].extend(np.asarray(modid))

    def set_query_num(self, mode: int, num_query: int):
        self.query_nums[mode] = num_query

    def split_all(self):
        for mode, num_query in self.query_nums.items():
            if not self.feats[mode]:
                continue
            feats_all = torch.cat(self.feats[mode], dim=0)
            qf_cur, gf_cur = feats_all[:num_query], feats_all[num_query:]

            self.qf.append(qf_cur)
            self.gf.append(gf_cur)

            self.q_pids.extend(self.pids[mode][:num_query])
            self.q_camids.extend(self.camids[mode][:num_query])
            self.q_mids.extend(self.modids[mode][:num_query])

            self.g_pids.extend(self.pids[mode][num_query:])
            self.g_camids.extend(self.camids[mode][num_query:])
            self.g_mids.extend(self.modids[mode][num_query:])

            self.feats[mode].clear()
            self.pids[mode].clear()
            self.camids[mode].clear()
            self.modids[mode].clear()

            self.logger and self.logger.info(f"=> Split mode {mode}: "
                                             f"query {num_query}, total query {len(self.q_pids)}")

    def _gather_to_rank0(self):
        multi_proc = dist.is_initialized() and dist.get_world_size() > 1
        if not multi_proc:
            # Single process (single GPU / DataParallel) path.
            return True, None

        # -------- DDP (world_size > 1) --------
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        gloo_pg    = get_gloo_group()

        local_pack = None
        if self.qf:
            local_pack = dict(
                qf=torch.cat(self.qf, dim=0).cpu(),
                gf=torch.cat(self.gf, dim=0).cpu(),
                qpid=np.asarray(self.q_pids, dtype=np.int64),
                qcam=np.asarray(self.q_camids, dtype=np.int64),
                qmid=np.asarray(self.q_mids, dtype=np.int64),
                gpid=np.asarray(self.g_pids, dtype=np.int64),
                gcam=np.asarray(self.g_camids, dtype=np.int64),
                gmid=np.asarray(self.g_mids, dtype=np.int64)
            )

        if rank == 0:
            gather_list = [None] * world_size
            dist.gather_object(local_pack, gather_list, dst=0, group=gloo_pg)
            return True, gather_list
        else:
            dist.gather_object(local_pack, dst=0, group=gloo_pg)
            return False, None

    def compute(self):
        is_rank0, packs = self._gather_to_rank0()
        if not is_rank0:
            return (None,)*7

        if packs is None:
            qf = torch.cat(self.qf, dim=0)
            gf = torch.cat(self.gf, dim=0)
            q_pids, q_camids, q_mids = map(np.asarray,
                                           (self.q_pids, self.q_camids, self.q_mids))
            g_pids, g_camids, g_mids = map(np.asarray,
                                           (self.g_pids, self.g_camids, self.g_mids))
        else:
            self.qf, self.gf = [], []
            self.q_pids, self.q_camids, self.q_mids = [], [], []
            self.g_pids, self.g_camids, self.g_mids = [], [], []

            for pack in packs:
                if pack is None:
                    continue
                self.qf.append(pack["qf"])
                self.gf.append(pack["gf"])
                qpid_pack = pack["qpid"]
                qcam_pack = pack["qcam"]
                qmid_pack = pack["qmid"]
                gpid_pack = pack["gpid"]
                gcam_pack = pack["gcam"]
                gmid_pack = pack["gmid"]

                self.q_pids.extend(qpid_pack.tolist() if isinstance(qpid_pack, np.ndarray) else qpid_pack)
                self.q_camids.extend(qcam_pack.tolist() if isinstance(qcam_pack, np.ndarray) else qcam_pack)
                self.q_mids.extend(qmid_pack.tolist() if isinstance(qmid_pack, np.ndarray) else qmid_pack)
                self.g_pids.extend(gpid_pack.tolist() if isinstance(gpid_pack, np.ndarray) else gpid_pack)
                self.g_camids.extend(gcam_pack.tolist() if isinstance(gcam_pack, np.ndarray) else gcam_pack)
                self.g_mids.extend(gmid_pack.tolist() if isinstance(gmid_pack, np.ndarray) else gmid_pack)

            qf = torch.cat(self.qf, dim=0)
            gf = torch.cat(self.gf, dim=0)
            q_pids, q_camids, q_mids = map(np.asarray,
                                           (self.q_pids, self.q_camids, self.q_mids))
            g_pids, g_camids, g_mids = map(np.asarray,
                                           (self.g_pids, self.g_camids, self.g_mids))

        if self.feat_norm:
            self.logger and self.logger.info("The test feature is normalized")
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)

        distmat = None
        if self.reranking:
            self.logger and self.logger.info('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            self.logger and self.logger.info(f'=> Computing DistMat in q-chunks (chunk={DISTMAT_Q_CHUNK})')

        if self.top_k:
            self.logger and self.logger.info('=> Computing top_k_indices')
            if self.reranking:
                indices = top_k_indices(distmat, self.top_k)
            else:
                indices = compute_indices_chunked(qf, gf, top_k=self.top_k, q_chunk_size=DISTMAT_Q_CHUNK)
            cmc, mAP = eval_func_top_k(indices, q_pids, g_pids,
                                       q_camids, g_camids, q_mids, g_mids)
        else:
            if self.reranking:
                indices = np.argsort(distmat, axis=1).astype(np.int32)
            else:
                indices = compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK)
            cmc, mAP = eval_func(indices, q_pids, g_pids,
                                 q_camids, g_camids)


        modality_pair_results = []
        if q_mids.size > 0 and g_mids.size > 0: # Check if modality information is available
            unique_q_mod_ids = sorted(np.unique(q_mids))
            unique_g_mod_ids = sorted(np.unique(g_mids))
            self.logger and self.logger.info(f"Starting modality-pair specific evaluation. Query mods: {unique_q_mod_ids}, Gallery mods: {unique_g_mod_ids}")

            for q_mod_target in unique_q_mod_ids:
                for g_mod_target in unique_g_mod_ids:
                    query_indices_for_pair = np.where(q_mids == q_mod_target)[0]
                    gallery_indices_for_pair = np.where(g_mids == g_mod_target)[0]

                    q_pids_pair = q_pids[query_indices_for_pair]
                    q_camids_pair = q_camids[query_indices_for_pair]
                    g_pids_pair = g_pids[gallery_indices_for_pair]
                    g_camids_pair = g_camids[gallery_indices_for_pair]

                    # Calculate indices for this pair (full argsort, int32)
                    if self.reranking:
                        distmat_pair = distmat[query_indices_for_pair][:, gallery_indices_for_pair]
                        indices_pair = np.argsort(distmat_pair, axis=1).astype(np.int32)
                    else:
                        query_indices_tensor = torch.from_numpy(query_indices_for_pair).long()
                        gallery_indices_tensor = torch.from_numpy(gallery_indices_for_pair).long()
                        qf_pair = qf.index_select(0, query_indices_tensor)
                        gf_pair = gf.index_select(0, gallery_indices_tensor)
                        indices_pair = compute_indices_chunked(
                            qf_pair, gf_pair, top_k=0, q_chunk_size=DISTMAT_Q_CHUNK
                        )
                    
                    # Use the same global eval_func for pairs. Ensure it can handle potentially smaller inputs.
                    pair_cmc, pair_mAP, pair_mINP = eval_func(
                        indices_pair, q_pids_pair, g_pids_pair,
                        q_camids_pair, g_camids_pair, max_rank=self.max_rank,
                        return_minp=True,
                    )
                     
                    modality_pair_results.append(
                        (q_mod_target, g_mod_target, pair_cmc, pair_mAP, pair_mINP)
                    )
                    self.logger.info(
                        f"  Pair (Q:{q_mod_target}, G:{g_mod_target}) -> "
                        f"Rank-1: {pair_cmc[0]:.2%}, mAP: {pair_mAP:.2%}, "
                        f"mINP: {pair_mINP:.2%}"
                    )

            modality_names = {1: "RGB", 2: "NIR", 3: "TIR"}
            pair_lookup = {
                (q_mod, g_mod): (pair_cmc, pair_mAP, pair_mINP)
                for q_mod, g_mod, pair_cmc, pair_mAP, pair_mINP
                in modality_pair_results
            }

            if (2, 1) in pair_lookup and (3, 1) in pair_lookup:
                self.logger.info("VI-ReID results (Table 5 protocol):")
                self.logger.info("  Setting   Rank-1     mAP    mINP")
                for q_mod, g_mod in ((2, 1), (3, 1)):
                    pair_cmc, pair_mAP, pair_mINP = pair_lookup[(q_mod, g_mod)]
                    self.logger.info(
                        f"  {modality_names[q_mod]}->{modality_names[g_mod]}"
                        f"   {pair_cmc[0]:7.2%} {pair_mAP:7.2%} {pair_mINP:7.2%}"
                    )

            if all((q_mod, g_mod) in pair_lookup
                   for q_mod in (1, 2, 3) for g_mod in (1, 2, 3)):
                self.logger.info("Modality-pair results (Table 6 protocol):")
                self.logger.info(
                    "  Query Gallery  Rank-1     mAP    mINP"
                )
                for q_mod in (1, 2, 3):
                    for g_mod in (1, 2, 3):
                        pair_cmc, pair_mAP, pair_mINP = pair_lookup[(q_mod, g_mod)]
                        self.logger.info(
                            f"  {modality_names[q_mod]:>5} {modality_names[g_mod]:>7}"
                            f" {pair_cmc[0]:7.2%} {pair_mAP:7.2%} {pair_mINP:7.2%}"
                        )
        else:
            self.logger and self.logger.info("Modality IDs (q_mids or g_mids) are empty or not available. Skipping pair-specific evaluation.")

        # AG-ReID protocol: cameras 1-5 are ground and cameras 6-7 are aerial
        # in WHU-MARS (camera ids are zero-based after dataset loading). Keep
        # all modalities and restrict only the query/gallery viewpoints, as in
        # the paper's Table 4 protocol.
        viewpoint_results = []
        ground_q = np.where(q_camids < 5)[0]
        aerial_q = np.where(q_camids >= 5)[0]
        ground_g = np.where(g_camids < 5)[0]
        aerial_g = np.where(g_camids >= 5)[0]

        viewpoint_pairs = (
            ("A->G", aerial_q, ground_g),
            ("G->A", ground_q, aerial_g),
        )
        if all(len(indices_) > 0 for _, q_indices, g_indices in viewpoint_pairs
               for indices_ in (q_indices, g_indices)):
            self.logger and self.logger.info(
                "AG-ReID results (Table 4 protocol; all modalities):"
            )
            for name, q_indices, g_indices in viewpoint_pairs:
                q_index = torch.from_numpy(q_indices).long()
                g_index = torch.from_numpy(g_indices).long()
                pair_indices = compute_indices_chunked(
                    qf.index_select(0, q_index),
                    gf.index_select(0, g_index),
                    top_k=0,
                    q_chunk_size=DISTMAT_Q_CHUNK,
                )
                pair_cmc, pair_mAP = eval_func(
                    pair_indices,
                    q_pids[q_indices], g_pids[g_indices],
                    q_camids[q_indices], g_camids[g_indices],
                    max_rank=self.max_rank,
                )
                viewpoint_results.append((name, pair_cmc, pair_mAP))
                rank10 = pair_cmc[9] if len(pair_cmc) >= 10 else float("nan")
                self.logger.info(
                    f"  {name} -> mAP: {pair_mAP:.2%}, "
                    f"Rank-1: {pair_cmc[0]:.2%}, Rank-10: {rank10:.2%}"
                )
        else:
            self.logger and self.logger.info(
                "AG-ReID evaluation skipped: both ground and aerial query/gallery "
                "samples are required."
            )

        if self.detailed_domain_eval:
            evaluate_domain_protocols(
                qf, gf, q_pids, q_camids, q_mids,
                g_pids, g_camids, g_mids, self.max_rank,
                reranking=self.reranking,
                distmat=distmat,
                logger=self.logger,
                csv_path=self.domain_report_path,
                overall_cmc=cmc,
                overall_mAP=mAP,
            )

        self.reset()
        return cmc, mAP, modality_pair_results, viewpoint_results, q_pids, q_camids, qf, gf
