# -*- coding: utf-8 -*-
"""B5: content-grounded repair 正式评估 —— 三个模型 × 两个电池（GPU）.

比较（B5 成功标准，全部相对 standard-50 步 DPO）:
  sft（起点） / b3_standard（50 步 standard DPO） / b3_cure_v3_s25（25 步 early-stop 修复）
电池:
  A) 48 任务电池（17 渲染/任务: neutral + 14 帧 + 2 扰动）
  B) b0_heldout 24 新任务电池（30 渲染/任务: 14 std + 12 未见模板 + 2 nat + 2 扰动），
     该电池未参与任何指标选择（H4 域外测试集，锁死）
指标: FS / CS / flip(vs neutral argmax) / capability(neutral 与逐帧 canonical 准确率) /
      family fingerprint（6 家族 FS 向量）
判定 gate（MD B5 节）: FS↓、flip↓、CS 不显著下降、capability 不显著下降、
      held-out 上仍有效、fingerprint 未被抹除。paired bootstrap（任务重采样 B=2000）。
输出: results/b5_repair_eval.json；心跳 results/_vb_heartbeat.json
用法: D:/conda_envs/flashhead/python.exe b5_eval_repair.py
"""
import gc
import json
import os
import sys
from datetime import datetime

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from items_v3 import get_tasks48  # noqa: E402
from items_b0 import get_tasks_b0, _TASKS_B0  # noqa: E402
from t3_unified_e1 import choice_probs_dual  # noqa: E402

R = os.path.join(THIS, "results")
OUT = os.path.join(R, "b5_repair_eval.json")
HEARTBEAT = os.path.join(R, "_vb_heartbeat.json")
BASE = r"D:\models\qwen15-ladder\sft"
STD50 = r"D:\models\qwen15-ladder\b3_standard"
CURE_S25_ADAPTER = r"D:\models\qwen15-ladder\b3_cure_v3_s25_s25_adapter"
FAM_OF_STD14 = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAM_OF_NEW12 = {i: f for i, f in enumerate(
    ["rephrase"] * 2 + ["case"] * 2 + ["persona"] * 2 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]


def heartbeat(model, battery, tid, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B5", "script": "b5_eval_repair.py", "model": model,
                   "battery": battery, "task": tid, "done": done, "total": total,
                   "ts": datetime.now().isoformat()}, fh, ensure_ascii=False)


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def load_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    models = {}
    for tag, path in [("sft", BASE), ("standard50", STD50)]:
        m = AutoModelForCausalLM.from_pretrained(
            path, device_map="cuda:0", low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        models[tag] = m
        print(f"loaded {tag}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="cuda:0", low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    m = PeftModel.from_pretrained(base, CURE_S25_ADAPTER).eval()
    models["cure_s25"] = m
    print("loaded cure_s25 (sft + s25 adapter)", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    return models, tok


def battery48_renders():
    tasks = get_tasks48()
    out = {}
    for tid, t in tasks.items():
        out[tid] = {"renders": [t["content"]] + list(t["frames"]) + list(t["content_perturb"]),
                    "answer_letter": 0, "n_frames": 14, "fams": [FAM_OF_STD14[i] for i in range(14)],
                    "neutral_idx": 0, "frame_idx": list(range(1, 15)), "perturb_idx": [15, 16]}
    return out


def battery_b0_renders():
    tasks = get_tasks_b0()
    ans = {t[0]: t[3] for t in _TASKS_B0}
    out = {}
    for tid, t in tasks.items():
        allf = t["frames_std"] + t["frames_new"]
        out[tid] = {"renders": allf + t["perturb_prompts"],
                    "answer_letter": ord(ans[tid]) - 65,
                    "n_frames": len(allf),
                    "fams": [FAM_OF_STD14[i] for i in range(14)] +
                            [FAM_OF_NEW12[i] for i in range(12)],
                    "neutral_idx": None, "frame_idx": list(range(len(allf))),
                    "perturb_idx": [len(allf), len(allf) + 1]}
    return out


def run_battery(models, tok, battery, tag_battery, out):
    for mname, model in models.items():
        res = out.setdefault(mname, {}).get(tag_battery)
        if res is not None:
            print(f"[skip] {mname}/{tag_battery} already done", flush=True)
            continue
        rows = {}
        tids = sorted(battery)
        for k, tid in enumerate(tids):
            spec = battery[tid]
            ps, ok = [], []
            for r in spec["renders"]:
                d = choice_probs_dual(model, tok, r + "\nAnswer:")[0]
                ps.append(None if d is None else np.asarray(d, float))
                ok.append(d is not None)
            rows[tid] = {"probs": [None if p is None else p.tolist() for p in ps],
                         "answer_letter": spec["answer_letter"],
                         "n_frames": spec["n_frames"], "fams": spec["fams"],
                         "neutral_idx": spec["neutral_idx"],
                         "frame_idx": spec["frame_idx"], "perturb_idx": spec["perturb_idx"]}
            heartbeat(mname, tag_battery, tid, k + 1, len(tids))
        out[mname][tag_battery] = rows
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        os.replace(tmp, OUT)
        print(f"  {mname}/{tag_battery} done ({len(tids)} tasks)", flush=True)


def metrics(rows):
    """rows: {tid: spec+probs} → 聚合指标 + 逐任务值。"""
    per = {}
    for tid, r in rows.items():
        ps = [np.array(v, float) if v is not None else None for v in r["probs"]]
        fr = [ps[i] for i in r["frame_idx"] if ps[i] is not None]
        pe = [ps[i] for i in r["perturb_idx"] if ps[i] is not None]
        if len(fr) < 6:
            continue
        ref = np.mean(fr, axis=0)
        fs = float(np.mean([jsd(v, ref) for v in fr]))
        cs = float(np.mean([jsd(v, ref) for v in pe])) if pe else float("nan")
        am = [int(np.argmax(v)) for v in fr]
        counts = {}
        for a in am:
            counts[a] = counts.get(a, 0) + 1
        mode = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        flip = float(np.mean([a != mode for a in am]))
        acc_frames = float(np.mean([a == r["answer_letter"] for a in am]))
        acc_neutral = (float(int(np.argmax(ps[r["neutral_idx"]]) == r["answer_letter"]))
                       if r["neutral_idx"] is not None and ps[r["neutral_idx"]] is not None else np.nan)
        fam_fs = {}
        for k, i in enumerate(r["frame_idx"]):
            if ps[i] is None:
                continue
            fam_fs.setdefault(r["fams"][k], []).append(jsd(ps[i], ref))
        per[tid] = {"fs": fs, "cs": cs, "flip": flip,
                    "acc_frames": acc_frames, "acc_neutral": acc_neutral,
                    "fam_fs": {k: float(np.mean(v)) for k, v in fam_fs.items()}}
    agg = {
        "fs": float(np.mean([v["fs"] for v in per.values()])),
        "cs": float(np.nanmean([v["cs"] for v in per.values()])),
        "flip": float(np.mean([v["flip"] for v in per.values()])),
        "acc_frames": float(np.mean([v["acc_frames"] for v in per.values()])),
        "acc_neutral": float(np.nanmean([v["acc_neutral"] for v in per.values()])),
        "family_fs": {k: float(np.mean([v["fam_fs"][k] for v in per.values() if k in v["fam_fs"]]))
                      for k in FAMS},
    }
    return per, agg


def boot_ci_delta(deltas, B=2000, seed=7):
    rng = np.random.RandomState(seed)
    d = np.asarray([x for x in deltas if x == x], float)
    if len(d) == 0:
        return [None, None]
    bs = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]


def main():
    out = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            out = json.load(fh)
    if not out.get("cure_s25", {}).get("b0"):
        models, tok = load_models()
        b48 = battery48_renders()
        b0 = battery_b0_renders()
        run_battery(models, tok, b48, "battery48", out)
        run_battery(models, tok, b0, "battery_b0", out)
        del models
        gc.collect(); torch.cuda.empty_cache()

    # ---- 指标与判定 ----
    agg, per = {}, {}
    for mname in out:
        agg[mname], per[mname] = {}, {}
        for bat in ["battery48", "battery_b0"]:
            if bat in out[mname]:
                per[mname][bat], agg[mname][bat] = metrics(out[mname][bat])
    print(json.dumps(agg, indent=1, ensure_ascii=False)[:1500], flush=True)

    gates = {}
    for bat in ["battery48", "battery_b0"]:
        if bat not in agg.get("standard50", {}) or bat not in agg.get("cure_s25", {}):
            continue
        tids = sorted(set(per["standard50"][bat]) & set(per["cure_s25"][bat]))
        dfs = [per["cure_s25"][bat][t]["fs"] - per["standard50"][bat][t]["fs"] for t in tids]
        dfl = [per["cure_s25"][bat][t]["flip"] - per["standard50"][bat][t]["flip"] for t in tids]
        dcs = [per["cure_s25"][bat][t]["cs"] - per["standard50"][bat][t]["cs"] for t in tids]
        dac = [per["cure_s25"][bat][t]["acc_frames"] - per["standard50"][bat][t]["acc_frames"]
               for t in tids]
        gates[bat] = {
            "fs_delta": round(float(np.mean(dfs)), 6), "fs_ci": boot_ci_delta(dfs),
            "flip_delta": round(float(np.mean(dfl)), 6), "flip_ci": boot_ci_delta(dfl),
            "cs_delta": round(float(np.nanmean(dcs)), 6), "cs_ci": boot_ci_delta(dcs),
            "acc_delta": round(float(np.nanmean(dac)), 6), "acc_ci": boot_ci_delta(dac),
            "fs_reduced": bool(np.mean(dfs) < 0 and boot_ci_delta(dfs)[1] < 0),
            "flip_reduced": bool(np.mean(dfl) < 0),
            "cs_not_sig_lower": bool(boot_ci_delta(dcs)[0] is None or boot_ci_delta(dcs)[0] > -0.02),
            "acc_not_sig_lower": bool(boot_ci_delta(dac)[0] is None or boot_ci_delta(dac)[0] > -0.05),
        }
    fp_std = agg.get("standard50", {}).get("battery48", {}).get("family_fs", {})
    fp_cure = agg.get("cure_s25", {}).get("battery48", {}).get("family_fs", {})
    if fp_std and fp_cure:
        v_std = np.array([fp_std[k] for k in FAMS])
        v_cur = np.array([fp_cure[k] for k in FAMS])
        gates["fingerprint_preserved"] = {
            "family_fs_standard": fp_std, "family_fs_cure": fp_cure,
            "cosine": round(float(v_std @ v_cur /
                                  (np.linalg.norm(v_std) * np.linalg.norm(v_cur) + 1e-12)), 6),
            "not_erased": bool(float(v_std @ v_cur /
                                     (np.linalg.norm(v_std) * np.linalg.norm(v_cur) + 1e-12)) > 0.8),
        }
    out["aggregate"] = agg
    out["gates"] = gates
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, indent=1, ensure_ascii=False), flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
