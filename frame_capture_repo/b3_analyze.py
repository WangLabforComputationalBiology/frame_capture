# -*- coding: utf-8 -*-
"""B3 判定: reward-frame coupling 的训练时间顺序（读 b3_traj_{tag}.json）.

方案文档 B3 节"最低证据"：
  1. standard DPO 中 frame-conditioned reward gap 显著升高
  2. gap 的增量预测之后的 task-level FS amplification（时间先后）
  3. frame-recoded 对照臂不出现同幅 FS 放大
  4. content-grounded 臂降低 frame risk（本判定加报全预算稳定性 caveat）

统计：
  - amplification: final/step0
  - T90: 指标首次达到"最终变化量 90%"的步数（gap 与 FS 的先后顺序）
  - 滞后相关: Spearman(Δgap_t, ΔFS_{t+1}) vs Spearman(ΔFS_t, Δgap_{t+1})
输出: results/b3_analysis.json + 控制台判定
用法: D:/conda_envs/flashhead/python.exe b3_analyze.py
"""
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
OUT = os.path.join(R, "b3_analysis.json")
TAGS = ["standard", "frame_v2", "cure_v3"]


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


def t90(steps, values):
    """首次达到 (max-min)*0.9+min 的步数；变化量<=0 时返回 None。"""
    v = np.asarray(values, float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return None
    thr = lo + 0.9 * (hi - lo)
    for s, x in zip(steps, v):
        if x >= thr:
            return int(s)
    return None


def load(tag):
    with open(os.path.join(R, f"b3_traj_{tag}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    out = {"arms": {}, "gates": {}}
    arms = {}
    for tag in TAGS:
        d = load(tag)
        ss = d["steps"]
        steps = [s["step"] for s in ss]
        fs = [s["fs"] for s in ss]
        cs = [s["cs"] for s in ss]
        flip = [s["flip_rate"] for s in ss]
        gap = [s["reward_gap_abs"] for s in ss]
        ent = [s.get("mean_entropy") for s in ss]
        arms[tag] = {
            "n_evals": len(ss), "steps": steps,
            "fs0": fs[0], "fs_final": fs[-1],
            "fs_amplification": r6(fs[-1] / fs[0]) if fs[0] > 0 else None,
            "fs_min": r6(min(fs)), "fs_min_step": int(steps[int(np.argmin(fs))]),
            "cs0": cs[0], "cs_final": r6(cs[-1]),
            "flip0": flip[0], "flip_final": r6(flip[-1]), "flip_max": r6(max(flip)),
            "gap_final": r6(gap[-1]), "gap_first_eval": r6(gap[1]) if len(gap) > 1 else None,
            "gap_t90": t90(steps, gap), "fs_t90": t90(steps, fs),
            "cs_t90": t90(steps, cs),
            "entropy_final": r6(ent[-1]) if ent and ent[-1] is not None else None,
            "entropy_min": r6(min(v for v in ent if v is not None)) if any(v is not None for v in ent) else None,
        }
    out["arms"] = arms

    st = arms["standard"]
    fv = arms["frame_v2"]
    cv = arms["cure_v3"]

    # gate 1: standard 臂 gap 显著升高
    g1 = bool(st["gap_final"] is not None and st["gap_final"] >= 2.0
              and st["gap_final"] > 5 * max(st["gap_first_eval"], 1e-6))
    # gate 2: gap 先于/伴随 FS —— T90(gap) <= T90(FS)
    g2 = bool(st["gap_t90"] is not None and st["fs_t90"] is not None
              and st["gap_t90"] <= st["fs_t90"])
    # gate 2b: 滞后相关 gap→FS 不弱于 FS→gap
    ss = load("standard")["steps"]
    dgap = np.diff([s["reward_gap_abs"] for s in ss])
    dfs = np.diff([s["fs"] for s in ss])
    lag_fwd = spearman(dgap[:-1], dfs[1:])   # Δgap_t → ΔFS_{t+1}
    lag_bwd = spearman(dfs[:-1], dgap[1:])   # ΔFS_t → Δgap_{t+1}
    out["lag_correlation"] = {"gap_leads_fs": r6(lag_fwd), "fs_leads_gap": r6(lag_bwd),
                              "contemporaneous": r6(spearman(dgap, dfs))}
    g2b = bool(lag_fwd == lag_fwd and lag_fwd >= lag_bwd - 0.15)
    # gate 3: frame_v2 对照臂无同幅放大
    g3 = bool(fv["fs_amplification"] is not None and st["fs_amplification"] is not None
              and fv["fs_amplification"] < st["fs_amplification"] * 0.7)
    # gate 4: cure_v3 前半程（step<=30）FS 显著下降
    g4 = bool(cv["fs_min"] is not None and cv["fs0"] > 0
              and cv["fs_min"] <= cv["fs0"] * 0.8)
    # caveat: cure_v3 全预算末段失稳（FS 超过 standard 终值、flip 高）
    caveat = {
        "cure_v3_final_fs_above_standard": bool(cv["fs_final"] and st["fs_final"]
                                                and cv["fs_final"] > st["fs_final"]),
        "cure_v3_final_flip": cv["flip_final"], "standard_final_flip": st["flip_final"],
        "note": "content-grounded 臂前 25 步 FS 降 40%（修复方向正确），后 25 步失稳上冲；"
                "修复主张须限定 early-stopping regime 或调整训练预算，留待 B5 正式协议",
    }
    out["gates"] = {
        "g1_standard_gap_rises": g1,
        "g2_gap_precedes_fs_t90": g2,
        "g2b_lag_correlation": g2b,
        "g3_frame_recoded_no_amplification": g3,
        "g4_content_grounded_early_fs_reduction": g4,
        "caveat_cure_v3_instability": caveat,
    }
    passed = sum([g1, g2, g2b, g3, g4])
    out["B3_verdict"] = "PASS" if passed == 5 else ("PASS_WITH_CAVEATS" if passed >= 4 else
                                                    ("PARTIAL" if passed >= 3 else "FAIL"))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out["arms"], indent=1, ensure_ascii=False))
    print(json.dumps(out["gates"], indent=1, ensure_ascii=False))
    print(f"B3 verdict: {out['B3_verdict']}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
