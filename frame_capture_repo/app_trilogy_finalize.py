# -*- coding: utf-8 -*-
"""E4: 固化三部曲全部数字到 results/app_trilogy.json.

来源:
  箭一(处方): phi_unified48_*.json -> FS/CS/指纹 (t0_treatment_table口径)
  箭二(认证): 15检查点 48任务电池 -> 证书 PASS/FAIL + 地板准绳
  箭三(守门): mmlu_dual.json 每项(agree/acc) 重算 coverage-accuracy 曲线
能力守门:   gpqa_gate.json (acc per model)
全部数字可从 results/ 溯源, 供论文应用节直接引用。
"""
import json
import os
import sys

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")
sys.path.insert(0, THIS)
from t0_treatment_table import ckpt_metrics  # noqa: E402

FLOOR_BANDS = {
    # 族 -> [(模型名, 地板FS)] 用 base 模型标定
    "qwen": [("qwen15-base", None)],
    "llama32": [("llama32-3b-base", None)],
    "gemma": [("gemma3-4b-it", None)],
    "internlm2": [("internlm2-base", None)],
    "tulu": [("tulu3-8b-base", None)],
    "qwen25": [("qwen25-7b-base", None)],
    "llama31": [("llama31-8b-instruct", None)],
}


def fs_of(name):
    per = ckpt_metrics(name)
    return np.mean([v["FS"] for v in per.values()])


def main():
    tril = {}

    # ---- 箭一: 处方疗效 ----
    ax1_models = ["qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame_v2",
                  "llama32-sft", "llama32-dpo", "llama32-dpo_frame_v2",
                  "gemma3-4b-it", "gemma3-4b-dpo_frame_v2"]
    ax1 = {}
    for n in ax1_models:
        per = ckpt_metrics(n)
        ax1[n] = {
            "FS": round(np.mean([v["FS"] for v in per.values()]), 4),
            "CS": round(np.mean([v["CS"] for v in per.values()]), 4),
            "n_tasks": len(per),
        }
    def r_common(a, b):
        pa, pb = ckpt_metrics(a), ckpt_metrics(b)
        common = [t for t in pa if t in pb]
        return round(float(np.corrcoef([pa[t]["FS"] for t in common],
                                       [pb[t]["FS"] for t in common])[0, 1]), 3)
    tril["Ax1_prescription"] = {
        "desc": "frame-recoded DPO keeps FS at the frame-floor across families",
        "models": ax1,
        "fingerprint_notes": {
            "qwen15: r(frame_v2,sft)": r_common("qwen15-sft", "qwen15-dpo_frame_v2"),
            "llama32: r(frame_v2,sft)": r_common("llama32-sft", "llama32-dpo_frame_v2"),
            "gemma: r(frame_v2,sft)": r_common("gemma3-4b-it", "gemma3-4b-dpo_frame_v2"),
        },
    }

    # ---- 箭三: 守门 (覆盖-准确率曲线, 从 mmlu_dual per-item 重算) ----
    mmlu = json.load(open(os.path.join(RES, "mmlu_dual.json"), encoding="utf-8"))
    ax3 = {}
    for m, v in mmlu.items():
        xs, ys = [], []
        for cat in ("law", "health"):
            for it in v["per"][cat].values():
                xs.append(it["agree"])
                ys.append(it["acc"])
        xs = np.array(xs)
        ys = np.array(ys)
        m_hi = xs >= 0.999
        ax3[m] = {
            "n": int(len(ys)),
            "full_acc": round(float(ys.mean()), 4),
            "cov_at_full_agree": round(float(m_hi.mean()), 4),
            "acc_at_full_agree": (round(float(ys[m_hi].mean()), 4) if m_hi.any() else None),
            "lift_pt": round(float((ys[m_hi].mean() - ys.mean()) * 100 if m_hi.any() else 0), 2),
            "consistent_but_wrong": int(((m_hi) & (ys <= 0.001)).sum()),
        }
    tril["Ax3_gatekeeper"] = {
        "desc": "frame agreement as free confidence: only answer frame-stable items",
        "models": ax3,
    }

    # ---- 能力守门 ----
    gpqa = json.load(open(os.path.join(RES, "gpqa_gate.json"), encoding="utf-8"))
    tril["Capability_gate"] = {"GPQA_acc": {m: v["acc"] for m, v in gpqa.items()}}

    # ---- 箭二: 认证证书 (全检查点 FS vs 地板带) ----
    # 地板带: 六个族 base FS
    floors = {}
    for fam, mods in FLOOR_BANDS.items():
        for mn, _ in mods:
            try:
                floors[fam] = round(fs_of(mn), 4)
            except Exception as e:
                floors[fam] = None
                print(f"  [warn] floor {mn}: {e}", flush=True)
    cert = {}
    all_ckpts = ["qwen15-base", "qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame_v2",
                 "qwen15-rlvr", "llama32-3b-base", "llama32-sft", "llama32-dpo",
                 "llama32-dpo_frame_v2", "gemma3-4b-it", "gemma3-4b-dpo_frame_v2",
                 "internlm2-base", "internlm2-chat", "qwen25-7b-base", "qwen25-7b-instruct",
                 "tulu3-8b-base", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b-rl",
                 "llama31-8b-instruct"]
    for n in all_ckpts:
        try:
            fs = fs_of(n)
            cert[n] = round(fs, 4)
        except Exception:
            pass
    tril["Ax2_certification"] = {
        "desc": "deployment audit: FS battery + floor band + fingerprint + capability gate",
        "family_floors": floors,
        "ckpt_FS": cert,
    }

    out = os.path.join(RES, "app_trilogy.json")
    json.dump(tril, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out, flush=True)
    # 简报
    print("\n=== Ax1 处方 FS ===")
    for m in ["qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame_v2",
              "llama32-sft", "llama32-dpo", "llama32-dpo_frame_v2",
              "gemma3-4b-it", "gemma3-4b-dpo_frame_v2"]:
        print(f"  {m:26} FS={ax1[m]['FS']}")
    print("\n=== Ax3 守门 ===")
    for m in ["qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame_v2",
              "llama32-sft", "llama32-dpo", "llama32-dpo_frame_v2"]:
        a = ax3[m]
        print(f"  {m:24} full={a['full_acc']:.3f} cov@1={a['cov_at_full_agree']:.3f} "
              f"acc@1={a['acc_at_full_agree']} lift={a['lift_pt']:+.1f}pt cw={a['consistent_but_wrong']}")
    print("\n=== GPQA 能力 ===")
    for m, d in gpqa.items():
        print(f"  {m:26} acc={d['acc']:.3f}")


if __name__ == "__main__":
    main()