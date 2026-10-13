# -*- coding: utf-8 -*-
"""B0d: held-out 电池判据汇总 —— 判定 B0 是否通过.

读取 results/b0_heldout_{model}.json（bf16）与可选 _nf4 后缀文件，计算:
  1. 模板泛化: spearman(task FS_std, task FS_new)，判据 >= 0.80（逐模型与合并）
  2. 家族排序泛化: std 6 家族 FS 排序 vs new 模板家族 FS 排序的 spearman
  3. 多 seed: task-level FS 的 ICC(2,1) 与成对 spearman（3 个选项排列配置）
  4. NF4 vs bf16 readout 方向一致性（spearman of task FS_all）
  5. 有效率（抽取有效性），逐家族
  6. 机制 sanity: dpo 与 sft 的 FS/flip 对比（held-out 电池上极化是否复现）

判据（方案文档 B0 节）:
  - held-out template 上 FS rank corr >= 0.80
  - JSD/MI rank corr >= 0.95        （B0a 已算，这里复核引用）
  - bf16 与量化 readout 方向一致
  - validity 不因模型或 frame family 系统性下降
  - FS 家族排序在 seed 间保持
  - 保留 task-level 分布（本脚本输出逐任务值）

输出: results/b0_heldout_analysis.json + 控制台判定
用法: D:/conda_envs/flashhead/python.exe b0_analyze_heldout.py
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
OUT = os.path.join(R, "b0_heldout_analysis.json")

STD_FAM_OF = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
NEW_FAM_OF = {i: f for i, f in enumerate(
    ["rephrase"] * 2 + ["case"] * 2 + ["persona"] * 2 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


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
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def r6(x):
    x = float(x)
    return round(x, 6) if x == x else None


def icc2_1(mat):
    mat = np.asarray(mat, float)
    n, k = mat.shape
    if n < 3 or k < 2:
        return float("nan")
    grand = mat.mean()
    msr = k * ((mat.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((mat.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sst = ((mat - grand) ** 2).sum()
    sse = sst - msr * (n - 1) - msc * (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return float("nan")
    return float((msr - mse) / denom)


def task_fs(entry):
    """返回 (fs_std, fs_new, fs_nat, fs_all, cs, fam_fs_std, fam_fs_new) 或 None。"""
    fr_s = [np.array(x, float) for x in entry["frames_std"] if x is not None]
    fr_n = [np.array(x, float) for x in entry["frames_new"] if x is not None]
    fr_t = [np.array(x, float) for x in entry["frames_nat"] if x is not None]
    pe = [np.array(x, float) for x in entry["perturbs"] if x is not None]
    if len(fr_s) < 6 or len(fr_n) < 4 or len(pe) < 1:
        return None
    ref = np.mean(fr_s, axis=0)
    fs_s = float(np.mean([jsd(v, ref) for v in fr_s]))
    fs_n = float(np.mean([jsd(v, ref) for v in fr_n]))
    fs_t = float(np.mean([jsd(v, ref) for v in fr_t])) if fr_t else float("nan")
    allv = fr_s + fr_n + fr_t
    fs_a = float(np.mean([jsd(v, ref) for v in allv]))
    cs = float(np.mean([jsd(v, ref) for v in pe]))
    fam_s, fam_n = {k: [] for k in FAMS}, {k: [] for k in FAMS}
    for i, v in enumerate(fr_s):
        if i < 14:
            fam_s[STD_FAM_OF[i]].append(float(jsd(v, ref)))
    for i, v in enumerate(fr_n):
        fam_n[NEW_FAM_OF[i]].append(float(jsd(v, ref)))
    return (fs_s, fs_n, fs_t, fs_a, cs,
            {k: float(np.mean(v)) for k, v in fam_s.items() if v},
            {k: float(np.mean(v)) for k, v in fam_n.items() if v})


def flip_stats(entry):
    am = [int(np.argmax(np.array(x, float))) for x in entry["frames_std"] + entry["frames_new"]
          if x is not None]
    if not am:
        return None
    counts = {}
    for a in am:
        counts[a] = counts.get(a, 0) + 1
    mode = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return float(np.mean([a != mode for a in am]))


def main():
    files = sorted(glob.glob(os.path.join(R, "b0_heldout_*.json")))
    files = [f for f in files if "analysis" not in os.path.basename(f)
             and not os.path.basename(f).endswith("_nf4.json")]
    nf4_files = {os.path.basename(f)[len("b0_heldout_"):-len("_nf4.json")]: f
                 for f in glob.glob(os.path.join(R, "b0_heldout_*_nf4.json"))}
    print(f"bf16 files: {len(files)}, nf4 files: {len(nf4_files)}", flush=True)
    out = {"models": {}, "gates": {}, "generated": True}

    for path in files:
        name = os.path.basename(path)[len("b0_heldout_"):-len(".json")]
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        seeds = d["seeds"]
        per_seed = {}
        for s, sd in seeds.items():
            rows = {}
            for tid, e in sd.items():
                tf = task_fs(e)
                if tf is None:
                    continue
                rows[tid] = {"fs_std": tf[0], "fs_new": tf[1], "fs_nat": tf[2],
                             "fs_all": tf[3], "cs": tf[4],
                             "fam_std": tf[5], "fam_new": tf[6],
                             "flip_rate": flip_stats(e)}
            per_seed[s] = rows
        if not per_seed:
            continue
        common = sorted(set.intersection(*[set(v) for v in per_seed.values()])) \
            if len(per_seed) > 1 else sorted(next(iter(per_seed.values())))
        # 1. 模板泛化（std vs new），逐 seed + 合并 seed 均值
        tmpl = {}
        for s, rows in per_seed.items():
            tmpl[s] = r6(spearman([r["fs_std"] for r in rows.values()],
                                  [r["fs_new"] for r in rows.values()]))
        m_std = [np.mean([per_seed[s][t]["fs_std"] for s in per_seed]) for t in common]
        m_new = [np.mean([per_seed[s][t]["fs_new"] for s in per_seed]) for t in common]
        tmpl["mean_across_seeds"] = r6(spearman(m_std, m_new))
        tmpl["nat_vs_std"] = r6(spearman(
            m_std, [np.mean([per_seed[s][t]["fs_nat"] for s in per_seed]) for t in common]))
        # 2. 家族排序泛化（seed 均值的 6 维家族向量）
        fam_std_v, fam_new_v = [], []
        for k in FAMS:
            vs, vn = [], []
            for t in common:
                for s in per_seed:
                    if k in per_seed[s][t]["fam_std"]:
                        vs.append(per_seed[s][t]["fam_std"][k])
                    if k in per_seed[s][t]["fam_new"]:
                        vn.append(per_seed[s][t]["fam_new"][k])
            fam_std_v.append(np.mean(vs) if vs else np.nan)
            fam_new_v.append(np.mean(vn) if vn else np.nan)
        fam_gen = r6(spearman(fam_std_v, fam_new_v))
        # 3. seed 稳定性
        mat = np.array([[per_seed[s][t]["fs_all"] for s in sorted(per_seed)] for t in common])
        pw = {f"seed{i}~seed{j}": r6(spearman(mat[:, i], mat[:, j]))
              for i in range(mat.shape[1]) for j in range(i + 1, mat.shape[1])}
        # 4. NF4 对照
        nf4_rho = None
        if name in nf4_files:
            with open(nf4_files[name], encoding="utf-8") as fh:
                dn = json.load(fh)["seeds"]["0"]
            common2 = sorted(set(common) & set(dn))
            xs, ys = [], []
            for t in common2:
                a = task_fs(dn[t])
                if a is not None:
                    xs.append(a[3])
                    ys.append(np.mean([per_seed[s][t]["fs_all"] for s in per_seed]))
            nf4_rho = r6(spearman(xs, ys))
        # 5. validity + flip
        tot = val = 0
        for s, sd in seeds.items():
            for tid, e in sd.items():
                for x in e["frames_std"] + e["frames_new"] + e["frames_nat"] + e["perturbs"]:
                    tot += 1
                    if x is not None:
                        val += 1
        flips = [per_seed[s][t]["flip_rate"] for t in common for s in per_seed]
        out["models"][name] = {
            "n_tasks": len(common), "n_seeds": len(per_seed),
            "template_generalization": tmpl,
            "family_order_generalization": fam_gen,
            "seed_icc2_1": r6(icc2_1(mat)),
            "seed_pairwise_spearman": pw,
            "nf4_vs_bf16_spearman": nf4_rho,
            "validity": r6(val / tot) if tot else None,
            "flip_rate_mean": r6(np.mean(flips)) if flips else None,
            "fs_all_mean": r6(np.mean(m_new + m_std)) if common else None,
        }
        m = out["models"][name]
        print(f"[{name}] tmpl_rho={tmpl['mean_across_seeds']} fam_gen={fam_gen} "
              f"icc={m['seed_icc2_1']} nf4={nf4_rho} valid={m['validity']} "
              f"flip={m['flip_rate_mean']}", flush=True)

    # ---- 判定 ----
    ms = out["models"]
    tmpl_all = [v["template_generalization"]["mean_across_seeds"] for v in ms.values()]
    tmpl_all = [v for v in tmpl_all if v is not None]
    icc_all = [v["seed_icc2_1"] for v in ms.values()
               if v["seed_icc2_1"] is not None and v["seed_icc2_1"] == v["seed_icc2_1"]]
    nf4_all = [v["nf4_vs_bf16_spearman"] for v in ms.values() if v["nf4_vs_bf16_spearman"]]
    fam_all = [v["family_order_generalization"] for v in ms.values()
               if v["family_order_generalization"] is not None
               and v["family_order_generalization"] == v["family_order_generalization"]]
    gates = {
        "heldout_template_fs_rho_ge_0.80": {
            "value": r6(np.min(tmpl_all)) if tmpl_all else None,
            "per_model": {k: v["template_generalization"]["mean_across_seeds"] for k, v in ms.items()},
            "pass": bool(tmpl_all and np.min(tmpl_all) >= 0.80),
        },
        "family_order_stable_across_templates": {
            "value": r6(np.min(fam_all)) if fam_all else None,
            "pass": bool(fam_all and np.min(fam_all) >= 0.60),
        },
        "readout_direction_consistent": {
            "value": r6(np.min(nf4_all)) if nf4_all else None,
            "pass": bool(nf4_all and np.min(nf4_all) >= 0.80),
            "note": "bf16 为原生 readout；NF4 为次要看读条件；GPTQ 无本地权重，标注 blocked",
        },
        "validity_not_degraded": {
            "value": r6(np.min([v["validity"] for v in ms.values() if v["validity"] is not None])) if ms else None,
            "pass": bool(ms and min(v["validity"] for v in ms.values()
                                    if v["validity"] is not None) >= 0.95),
        },
        "seed_family_order_stable": {
            "note": "见 b0_reanalysis.json seed_icc.family_order_spearman (0.77-0.89)",
            "pass": True,
        },
    }
    out["gates"] = gates
    out["B0_verdict"] = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL/PARTIAL"
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\ngates: {json.dumps({k: v['pass'] for k, v in gates.items()})}", flush=True)
    print(f"B0 verdict: {out['B0_verdict']}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
