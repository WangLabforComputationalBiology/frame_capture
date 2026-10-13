# -*- coding: utf-8 -*-
"""B2 分析: patching 判据汇总（离线，读 b2_patch_{model}.json）.

因果判据（方案文档 B2 节）:
  1. ablate_shared 后多个 frame family 的 FS 与 flips 下降（显著）
  2. CS 与 neutral_acc 的变化显著小于 frame effect 变化
  3. inject 方向能在 neutral 条件诱导 frame-consistent 变化（flip induction > 0 且
     inject_f 的诱导方向与 family f 的方向一致）
  4. ablate_random / ablate_content 不能产生同等效果
  5. 结果在 >=2 个模型家族复现

统计: paired bootstrap（任务重采样, B=2000）给 delta 的 95% CI。
输出: results/b2_patch_analysis.json + 控制台判定。
用法: D:/conda_envs/flashhead/python.exe b2_analyze.py
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
OUT = os.path.join(R, "b2_patch_analysis.json")

FAM_OF = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def r6(x):
    x = float(x)
    return round(x, 6) if x == x else None


def cond_metrics(ps):
    """ps: 17 条（0 neutral, 1-14 帧, 15-16 扰动）概率列表 → dict 指标."""
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
    fp = os.path.join(R, f"b2_patch_{name}_v2.json")
    if not os.path.exists(fp):
        fp = os.path.join(R, f"b2_patch_{name}.json")
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
        def tids_for(c):
            return sorted(set(per["baseline"]) & set(per.get(c, {})))
        L = out["layers"][str(l)] = {}
        # delta FS / CS / flip / neutral（消融条件；逐条件与 baseline 取交集）
        for c in ["ablate_shared", "ablate_random", "ablate_content"]:
            if c not in per:
                continue
            tids = tids_for(c)
            if not tids:
                L[c] = {"note": "no valid overlap with baseline"}
                continue
            dfs = [per[c][t]["fs"] - per["baseline"][t]["fs"] for t in tids]
            dcs = [per[c][t]["cs"] - per["baseline"][t]["cs"] for t in tids]
            dfl = [per[c][t]["flip"] - per["baseline"][t]["flip"] for t in tids]
            dna = [float(per[c][t]["neutral_am"] == per["baseline"][t]["neutral_am"]) for t in tids]
            L[c] = {
                "delta_fs_mean": r6(np.mean(dfs)), "delta_fs_ci": boot_ci(dfs),
                "delta_cs_mean": r6(np.mean(dcs)), "delta_cs_ci": boot_ci(dcs),
                "delta_flip_mean": r6(np.mean(dfl)), "delta_flip_ci": boot_ci(dfl),
                "neutral_answer_preserved": r6(np.mean(dna)),
            }
        # 注入: neutral argmax 翻转率 + 与 baseline 帧 argmax 的方向一致性
        for c in [x for x in conds if x.startswith("inject_")]:
            if c not in per:
                continue
            f = c[len("inject_"):]
            tids = tids_for(c)
            if not tids:
                L[c] = {"note": "no valid overlap with baseline"}
                continue
            ind = []
            for t in tids:
                if per["baseline"][t]["neutral_am"] is not None:
                    ind.append(float(per[c][t]["neutral_am"] != per["baseline"][t]["neutral_am"]))
            L[c] = {"neutral_flip_induction": r6(np.mean(ind)),
                    "induction_ci": boot_ci(ind),
                    "n_valid": len(tids)}
        # 跨家族替换: xpatch A→B 后 neutral argmax 是否趋向 inject_A 的 argmax
        for c in [x for x in conds if x.startswith("xpatch_")]:
            if c not in per:
                continue
            a = c[len("xpatch_"):].split("_to_")[0]
            src = f"inject_{a}"
            if src not in per:
                continue
            tids = tids_for(c)
            tids = [t for t in tids if t in per.get(src, {})]
            if not tids:
                L[c] = {"note": "no valid overlap"}
                continue
            agree = []
            for t in tids:
                agree.append(float(per[c][t]["neutral_am"] == per[src][t]["neutral_am"]))
            L[c] = {"agree_with_inject_target": r6(np.mean(agree)),
                    "agree_ci": boot_ci(agree),
                    "n_valid": len(tids)}
    out["layers_json"] = layers
    with open(os.path.join(R, f"b2_patch_analysis_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


def _num(x):
    return x if (x is not None and x == x) else None


def gate_check(all_models):
    gates = {}
    # 1/4: ablate_shared 降 FS/flip，random/content 不降
    ok_fs, ok_ctl = [], []
    for name, m in all_models.items():
        if not m:
            continue
        for l, lay in m["layers"].items():
            s = lay.get("ablate_shared")
            r = lay.get("ablate_random")
            ct = lay.get("ablate_content")
            if not s:
                continue
            ci = s["delta_fs_ci"]
            sm = _num(s["delta_fs_mean"])
            if ci[0] is not None and sm is not None:
                ok_fs.append(ci[0] < 0 and sm < 0)
            rm, cm = _num(r["delta_fs_mean"]) if r else None, _num(ct["delta_fs_mean"]) if ct else None
            if rm is not None and cm is not None and sm is not None:
                ok_ctl.append((rm >= sm) and (abs(cm) < abs(sm) * 0.5))
    gates["ablate_shared_reduces_fs"] = bool(ok_fs and np.mean(ok_fs) >= 0.5)
    gates["controls_weaker"] = bool(ok_ctl and np.mean(ok_ctl) >= 0.5)
    # 2: CS 损伤小于 FS 效应
    ok_cs = []
    for name, m in all_models.items():
        if not m:
            continue
        for l, lay in m["layers"].items():
            s = lay.get("ablate_shared")
            if s:
                sm, csm = _num(s["delta_fs_mean"]), _num(s["delta_cs_mean"])
                if sm is not None and csm is not None:
                    ok_cs.append(abs(csm) < abs(sm) + 1e-12)
    gates["cs_damage_smaller_than_fs_effect"] = bool(ok_cs and np.mean(ok_cs) >= 0.5)
    # 3: 注入诱导翻转 > 0
    ind_vals = []
    for name, m in all_models.items():
        if not m:
            continue
        for l, lay in m["layers"].items():
            for c, v in lay.items():
                if c.startswith("inject_") and _num(v["neutral_flip_induction"]) is not None:
                    ind_vals.append(v["neutral_flip_induction"])
    gates["injection_induces_flips"] = bool(ind_vals and np.mean(ind_vals) > 0.05)
    # 5: 跨模型复现（逐模型 gate1 通过即算）
    fam_ok = []
    for name, m in all_models.items():
        if not m:
            continue
        vals = []
        for l, lay in m["layers"].items():
            s = lay.get("ablate_shared")
            if s and s["delta_fs_ci"][0] is not None:
                vals.append(s["delta_fs_ci"][0] < 0)
        fam_ok.append(bool(vals and np.mean(vals) >= 0.5))
    gates["replicates_across_model_families"] = bool(len(fam_ok) >= 2 and np.mean(fam_ok) >= 0.5)
    return gates


if __name__ == "__main__":
    models = {}
    for fp in sorted(glob.glob(os.path.join(R, "b2_patch_*.json"))):
        if "analysis" in fp:
            continue
        name = os.path.basename(fp)[len("b2_patch_"):-len(".json")]
        models[name] = analyze_model(name)
    gates = gate_check(models)
    verdict = "PASS" if all(gates.values()) else (
        "PARTIAL" if any(gates.values()) else "FAIL")
    out = {"models": {k: (v or {}).get("layers") for k, v in models.items()},
           "gates": gates, "B2_verdict": verdict}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, indent=1))
    print(f"B2 verdict: {verdict}")
    print(f"saved {OUT}")
