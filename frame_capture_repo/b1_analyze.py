# -*- coding: utf-8 -*-
"""B1 分析: 跨 frame family 共享表示结构（离线，读 b1_hidden_{name}/*.npy）.

统一机制判据（方案文档 B1 节）——对每层计算:
  1. 跨家族方向对齐: 同任务内 family-mean 方向 d_f 的两两 cosine（任务平均）
  2. 共享子空间: 各家族"任务×dim"方向矩阵去均值后取 top-3 主子空间，
     两两主角度奇异值均值（overlap ∈ [0,1]）
  3. CKA（linear）: 家族条件化表示矩阵两两相似
  4. RSA: 家族 f1/f2 的任务-任务 RDM 上三角 Spearman
  5. frame vs content: 内容方向（改答案扰动）与各家族方向 cosine（应显著更低）
  6. 有效秩: 全帧效应矩阵（14 帧 × 任务堆叠）参与率秩
  7. nonsense 控制: 无意义前缀方向与家族方向 cosine（应低于家族间对齐）

判据: >=4 个自然语言家族在同一后段窗口显著对齐（cosine 阈值 0.2 且高于
nonsense/lexical 控制）；共享子空间在 held-out 上有效（留待 B4）。

输出: results/b1_representation_{name}.json（逐层 + 汇总窗口）
用法: D:/conda_envs/flashhead/python.exe b1_analyze.py [model ...]
"""
import glob
import json
import os
import sys

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")

FAM_OF_STD14 = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]
IDX_NEUTRAL, IDX_FRAMES, IDX_PERTURB, IDX_NONSENSE = 0, slice(1, 15), slice(15, 17), slice(17, 19)


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 0.5
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def r6(x):
    x = float(x)
    return round(x, 6) if x == x else None


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else float("nan")


def effective_rank(S):
    S = np.asarray(S, float)
    p = S / (S.sum() + 1e-15)
    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(np.exp(ent))


def analyze_model(name):
    files = sorted(glob.glob(os.path.join(R, f"b1_hidden_{name}", "*.npy")))
    if not files:
        print(f"[skip] {name}: no hidden files")
        return None
    stacks = {os.path.basename(f)[:-4]: np.load(f) for f in files}
    tids = sorted(stacks)
    n_renders, n_layers, dim = stacks[tids[0]].shape
    print(f"[{name}] tasks={len(tids)} renders={n_renders} layers={n_layers} dim={dim}", flush=True)

    # 家族成员帧在 d=hs[1:15]-hs[0] 内的行索引（d 的行号 == 帧编号 0..13）
    fam_idx = {f: [j for j in range(14) if FAM_OF_STD14[j] == f] for f in FAMS}

    layers_out = []
    for l in range(n_layers):
        # 收集方向
        dirs = {tid: {} for tid in tids}          # 家族方向
        cdirs, ndirs = {}, {}
        for tid in tids:
            hs = stacks[tid][:, l, :].astype(np.float64)  # (19, dim)
            neutral = hs[IDX_NEUTRAL]
            d = hs[IDX_FRAMES] - neutral               # (14, dim)
            for f in FAMS:
                dirs[tid][f] = d[fam_idx[f]].mean(axis=0)
            cdirs[tid] = (hs[IDX_PERTURB] - neutral).mean(axis=0)
            ndirs[tid] = (hs[IDX_NONSENSE] - neutral).mean(axis=0)

        # 1. 跨家族 cosine（任务平均）
        ali = np.zeros((6, 6))
        for i, f1 in enumerate(FAMS):
            for j, f2 in enumerate(FAMS):
                cs = [float(np.dot(dirs[t][f1], dirs[t][f2]) /
                            (np.linalg.norm(dirs[t][f1]) * np.linalg.norm(dirs[t][f2]) + 1e-12))
                      for t in tids]
                ali[i, j] = np.mean(cs)
        off = ali[np.triu_indices(6, 1)]
        n_aligned = 0
        for i, f1 in enumerate(FAMS):
            others = [ali[i, j] for j in range(6) if j != i]
            if np.mean(others) > 0.2:
                n_aligned += 1

        # 2. 主子空间 overlap（top-3，任务中心化）
        sub_ov = np.zeros((6, 6))
        subU = {}
        for i, f in enumerate(FAMS):
            X = np.stack([dirs[t][f] for t in tids])
            X = X - X.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            subU[f] = Vt[:3]
        for i, f1 in enumerate(FAMS):
            for j, f2 in enumerate(FAMS):
                s = np.linalg.svd(subU[f1] @ subU[f2].T, compute_uv=False)
                sub_ov[i, j] = float(np.clip(s, 0, 1).mean())
        off_ov = sub_ov[np.triu_indices(6, 1)]

        # 3. CKA（家族条件化表示矩阵）
        cka = np.zeros((6, 6))
        mats = {}
        for f in FAMS:
            mats[f] = np.stack([dirs[t][f] for t in tids])
        for i, f1 in enumerate(FAMS):
            for j, f2 in enumerate(FAMS):
                cka[i, j] = linear_cka(mats[f1], mats[f2])
        off_cka = cka[np.triu_indices(6, 1)]

        # 4. RSA（RDM 上三角 spearman）
        rsa = np.zeros((6, 6))
        rdms = {}
        for f in FAMS:
            M = mats[f]
            nrm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
            C = (M / nrm) @ (M / nrm).T
            iu = np.triu_indices(len(tids), 1)
            rdms[f] = 1 - C[iu]
        for i, f1 in enumerate(FAMS):
            for j, f2 in enumerate(FAMS):
                rsa[i, j] = spearman(rdms[f1], rdms[f2])
        off_rsa = rsa[np.triu_indices(6, 1)]

        # 5. frame vs content
        fc_cos = []
        for t in tids:
            nc = np.linalg.norm(cdirs[t]) + 1e-12
            for f in FAMS:
                nf = np.linalg.norm(dirs[t][f]) + 1e-12
                fc_cos.append(float(np.dot(dirs[t][f], cdirs[t]) / (nf * nc)))
        # 内容方向在共享子空间内的投影占比
        proj_ratios = []
        for t in tids:
            Xall = np.stack([dirs[t][f] for f in FAMS])          # (6, dim)
            Xc = Xall - Xall.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            keep = Vt[:3]
            nc = np.linalg.norm(cdirs[t]) + 1e-12
            proj = np.linalg.norm(keep.T @ (keep @ cdirs[t]))
            proj_ratios.append(float(proj / nc))

        # 6. 有效秩（全部帧方向堆叠）
        eff = []
        for t in tids:
            hs = stacks[t][:, l, :].astype(np.float64)
            d = hs[IDX_FRAMES] - hs[IDX_NEUTRAL]
            d = d - d.mean(axis=0, keepdims=True)
            S = np.linalg.svd(d, compute_uv=False)
            eff.append(effective_rank(S))
        eff_rank = float(np.mean(eff))

        # 7. nonsense 控制
        ns_cos = []
        for t in tids:
            nn = np.linalg.norm(ndirs[t]) + 1e-12
            for f in FAMS:
                nf = np.linalg.norm(dirs[t][f]) + 1e-12
                ns_cos.append(float(np.dot(dirs[t][f], ndirs[t]) / (nf * nn)))

        layers_out.append({
            "layer": l,
            "family_alignment_matrix": [[r6(v) for v in row] for row in ali],
            "offdiag_alignment_mean": r6(np.mean(off)),
            "n_families_aligned_gt0.2": int(n_aligned),
            "subspace_overlap_mean_top3": r6(np.mean(off_ov)),
            "cka_mean": r6(np.mean(off_cka)),
            "rsa_mean": r6(np.mean(off_rsa)),
            "frame_content_cos_mean": r6(np.mean(fc_cos)),
            "content_proj_in_frame_subspace": r6(np.mean(proj_ratios)),
            "effective_rank_frame_dirs": r6(eff_rank),
            "nonsense_family_cos_mean": r6(np.mean(ns_cos)),
        })

    # 汇总: 后段窗口（最后 1/4 层）判据
    late0 = int(n_layers * 0.75)
    late = [x for x in layers_out if x["layer"] >= late0]
    best = max(layers_out, key=lambda x: x["offdiag_alignment_mean"])
    summary = {
        "late_window_layers": [x["layer"] for x in late],
        "late_mean_alignment": r6(np.mean([x["offdiag_alignment_mean"] for x in late])),
        "late_max_n_families_aligned": int(max(x["n_families_aligned_gt0.2"] for x in late)),
        "late_mean_subspace_overlap": r6(np.mean([x["subspace_overlap_mean_top3"] for x in late])),
        "late_mean_frame_content_cos": r6(np.mean([x["frame_content_cos_mean"] for x in late])),
        "late_mean_nonsense_cos": r6(np.mean([x["nonsense_family_cos_mean"] for x in late])),
        "best_layer": best["layer"],
        "gate_ge4_families_in_late_window": bool(max(x["n_families_aligned_gt0.2"] for x in late) >= 4),
        "gate_alignment_gt_nonsense": bool(
            np.mean([x["offdiag_alignment_mean"] for x in late])
            > np.mean([x["nonsense_family_cos_mean"] for x in late])),
        "gate_alignment_gt_content": bool(
            np.mean([x["offdiag_alignment_mean"] for x in late])
            > np.mean([x["frame_content_cos_mean"] for x in late])),
    }
    out = {"model": name, "n_tasks": len(tids), "n_layers": n_layers, "dim": dim,
           "summary": summary, "layers": layers_out}
    fp = os.path.join(R, f"b1_representation_{name}.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[{name}] summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"saved {fp}", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        names = [os.path.basename(d)[len("b1_hidden_"):]
                 for d in glob.glob(os.path.join(R, "b1_hidden_*")) if os.path.isdir(d)]
    for n in names:
        analyze_model(n)
    print("B1 ANALYZE DONE", flush=True)
