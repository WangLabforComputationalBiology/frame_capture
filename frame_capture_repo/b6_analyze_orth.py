# -*- coding: utf-8 -*-
"""B6 分析: 正交化消融的可分性判据.

读 results/b6_orth_{model}.json，对每个模型×层算:
  ΔFS / ΔCS / Δflip（相对 baseline，paired bootstrap 95% CI）
  separability_ratio = (ΔCS_orth − ΔCS_shared) / (0 − ΔCS_shared)
      = 1  → 正交化后 CS 完全不受损（表示层可分）
      = 0  → 与粗暴消融一样伤 CS（物理共用）
  并对比 ablate_orthrandom（同秩随机对照）。

输出: results/b6_orth_analysis.json + 控制台判读。
用法: /e/Anaconda/python.exe b6_analyze_orth.py
"""
import glob
import json
import os
import sys

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")
OUT = os.path.join(R, "b6_orth_analysis.json")


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def r6(x):
    try:
        x = float(x)
    except Exception:
        return None
    return round(x, 6) if x == x else None


def cond_metrics(ps):
    fr = [np.array(x, float) for x in ps[1:15] if x is not None]
    pe = [np.array(x, float) for x in ps[15:17] if x is not None]
    neu = ps[0]
    out = {"valid": len(fr)}
    if len(fr) < 6 or neu is None:
        out["valid"] = 0
        return out
    ref = np.mean(fr, axis=0)
    out["fs"] = float(np.mean([jsd(v, ref) for v in fr]))
    out["cs"] = float(np.mean([jsd(v, ref) for v in pe])) if pe else float("nan")
    out["neutral_am"] = int(np.argmax(np.array(neu, float)))
    am = [int(np.argmax(v)) for v in fr]
    counts = {}
    for a in am:
        counts[a] = counts.get(a, 0) + 1
    mode = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    out["flip"] = float(np.mean([a != mode for a in am]))
    return out


def boot_ci(deltas, B=2000, seed=7):
    rng = np.random.RandomState(seed)
    d = np.asarray(deltas, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return [None, None]
    bs = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(B)]
    return [r6(np.percentile(bs, 2.5)), r6(np.percentile(bs, 97.5))]


def analyze_model(name):
    fp = os.path.join(R, f"b6_orth_{name}.json")
    if not os.path.exists(fp):
        return None
    with open(fp, encoding="utf-8") as fh:
        d = json.load(fh)
    layers = sorted({int(k.split(":")[0][1:]) for k in d["results"]})
    conds = sorted({k.split(":", 1)[1] for k in d["results"]})
    out = {"model": name, "source": os.path.basename(fp), "layers": {}}
    for l in layers:
        per = {}
        for c in conds:
            key = f"L{l}:{c}"
            if key not in d["results"]:
                continue
            rows = {}
            for tid, ps in d["results"][key].items():
                m = cond_metrics(ps)
                if m["valid"]:
                    rows[tid] = m
            per[c] = rows
        if "baseline" not in per:
            continue
        L = out["layers"][str(l)] = {}

        def tids_for(c):
            return sorted(set(per["baseline"]) & set(per.get(c, {})))

        for c in conds:
            if c == "baseline" or c not in per:
                continue
            tids = tids_for(c)
            if not tids:
                L[c] = {"note": "no valid overlap"}
                continue
            dfs = [per[c][t]["fs"] - per["baseline"][t]["fs"] for t in tids]
            dcs = [per[c][t]["cs"] - per["baseline"][t]["cs"] for t in tids]
            dfl = [per[c][t]["flip"] - per["baseline"][t]["flip"] for t in tids]
            dna = [float(per[c][t]["neutral_am"] == per["baseline"][t]["neutral_am"]) for t in tids]
            base_fs = np.mean([per["baseline"][t]["fs"] for t in tids])
            base_cs = np.mean([per["baseline"][t]["cs"] for t in tids])
            L[c] = {
                "n": len(tids),
                "base_fs": r6(base_fs), "base_cs": r6(base_cs),
                "delta_fs_mean": r6(np.mean(dfs)), "delta_fs_ci": boot_ci(dfs),
                "delta_cs_mean": r6(np.mean(dcs)), "delta_cs_ci": boot_ci(dcs),
                "delta_flip_mean": r6(np.mean(dfl)), "delta_flip_ci": boot_ci(dfl),
                "neutral_answer_preserved": r6(np.mean(dna)),
            }
        # ---- 核心判据: 可分性 ----
        sh, orr = L.get("ablate_shared"), L.get("ablate_orth")
        rnd = L.get("ablate_orthrandom")
        mat = L.get("ablate_orth_matched")
        if sh and orr:
            cs_sh = sh["delta_cs_mean"]
            cs_or = orr["delta_cs_mean"]
            L["_separability"] = {
                "delta_cs_shared": cs_sh,
                "delta_cs_orth": cs_or,
                "delta_cs_orth_matched": (mat or {}).get("delta_cs_mean") if mat else None,
                "delta_fs_orth_matched": (mat or {}).get("delta_fs_mean") if mat else None,
                "delta_fs_orth_matched_ci": (mat or {}).get("delta_fs_ci") if mat else None,
                "delta_cs_random_control": (rnd or {}).get("delta_cs_mean"),
                "delta_fs_shared": sh["delta_fs_mean"],
                "delta_fs_orth": orr["delta_fs_mean"],
                # CS 保留比例: 1 = 完全不伤 CS, 0 = 与粗暴消融同样伤
                "cs_damage_recovery": r6(1.0 - (cs_or / cs_sh)) if cs_sh not in (0, None) else None,
                "fs_effect_retained": r6(orr["delta_fs_mean"] / sh["delta_fs_mean"])
                if sh["delta_fs_mean"] not in (0, None) else None,
            }
    out["layers_json"] = layers
    return out


def verdict(all_models):
    """判据（FC_REFILL_PLAN §5 Step 2）+ 实测新增的第四档:
    PASS    = ΔCS_orth → ≈0 (|ΔCS| < 20% of |ΔCS_shared|) 且 ΔFS_orth 显著为负
    PARTIAL = ΔCS 改善但仍 >20%，或 ΔFS 消失
    FAIL    = ΔCS 与粗暴消融同量级
    SIGN-FLIP = ΔCS 由负转正（"删掉的不只是伤 CS 的那部分"，CS 反而变好）
               —— 实测 qwen15-sft L20/L24 属此档，说明 FS/CS 共用成分。
    matched 列一并输出，用于报告里堵秩选择混淆。
    """
    rows = []
    for name, m in all_models.items():
        if not m:
            continue
        for l, lay in m["layers"].items():
            s = lay.get("_separability")
            if not s:
                continue
            cs_sh, cs_or = s["delta_cs_shared"], s["delta_cs_orth"]
            cs_m = s.get("delta_cs_orth_matched")
            fs_m = s.get("delta_fs_orth_matched")
            fs_or = s["delta_fs_orth"]
            fs_ci = (lay.get("ablate_orth") or {}).get("delta_fs_ci", [None, None])
            fs_m_ci = s.get("delta_fs_orth_matched_ci", [None, None])
            fs_sig = fs_ci[0] is not None and fs_ci[1] is not None and fs_ci[1] < 0
            # matched 的 FS 显著性（是否 ns）
            fs_m_sig = (fs_m_ci[0] is not None and fs_m_ci[1] is not None
                        and (fs_m_ci[1] < 0 or fs_m_ci[0] > 0))
            cs_ratio = abs(cs_or) / abs(cs_sh) if cs_sh not in (0, None) else None
            if cs_ratio is None:
                v = "NA"
            elif cs_or is not None and cs_sh is not None and cs_or > 0 > cs_sh:
                v = "SIGN-FLIP"
            elif cs_ratio < 0.20 and fs_sig:
                v = "PASS"
            elif cs_ratio < 0.60:
                v = "PARTIAL"
            else:
                v = "FAIL"
            rows.append({"model": name, "layer": l, "verdict": v,
                         "cs_ratio_vs_shared": r6(cs_ratio),
                         "delta_cs_shared": cs_sh, "delta_cs_orth": cs_or,
                         "delta_cs_orth_matched": cs_m,
                         "delta_cs_orthrandom": s["delta_cs_random_control"],
                         "delta_fs_shared": s["delta_fs_shared"],
                         "delta_fs_orth": fs_or, "fs_orth_ci": fs_ci,
                         "delta_fs_orth_matched": fs_m, "fs_orth_matched_ci": fs_m_ci,
                         "fs_orth_matched_significant": bool(fs_m_sig) if fs_m is not None else None,
                         "fs_effect_retained": s["fs_effect_retained"]})
    return rows


if __name__ == "__main__":
    models = {}
    for fp in sorted(glob.glob(os.path.join(R, "b6_orth_*.json"))):
        if "analysis" in fp:
            continue
        name = os.path.basename(fp)[len("b6_orth_"):-len(".json")]
        models[name] = analyze_model(name)
    rows = verdict(models)
    print(f"{'model':16s} {'L':>3s} {'verdict':10s} {'cs_ratio':>9s} "
          f"{'dCS_shared':>11s} {'dCS_orth':>10s} {'dCS_match':>10s} {'dCS_rnd':>9s} "
          f"{'dFS_shared':>11s} {'dFS_orth':>10s} {'dFS_match':>10s} {'fs_retain':>9s}")
    for r in rows:
        print(f"{r['model']:16s} {r['layer']:>3s} {r['verdict']:10s} "
              f"{str(r['cs_ratio_vs_shared']):>9s} {str(r['delta_cs_shared']):>11s} "
              f"{str(r['delta_cs_orth']):>10s} {str(r['delta_cs_orth_matched']):>10s} "
              f"{str(r['delta_cs_orthrandom']):>9s} "
              f"{str(r['delta_fs_shared']):>11s} {str(r['delta_fs_orth']):>10s} "
              f"{str(r['delta_fs_orth_matched']):>10s} {str(r['fs_effect_retained']):>9s}")
    out = {"rows": rows,
           "models": {k: (v or {}).get("layers") for k, v in models.items()}}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}")
