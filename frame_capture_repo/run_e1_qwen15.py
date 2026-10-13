# -*- coding: utf-8 -*-
"""E1 第二家族：Qwen2.5-1.5B 自训阶梯（base→SFT→DPO→RLVR）。

复用 run_e1.py 协议，只换 MODEL_DIRS 与输出目录。
训练检查点位于 D:\models\qwen15-ladder\{sft,dpo,rlvr}（LoRA 合并后的完整权重）。

用法：
    D:/conda_envs/flashhead/python.exe run_e1_qwen15.py [--stages base]
"""
import argparse
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from run_e1 import load_model, compute_phi  # noqa: E402
from items import get_tasks  # noqa: E402

import numpy as np  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

MODEL_DIRS = {
    "base": r"D:\models\Qwen2.5-1.5B",
    "sft": r"D:\models\qwen15-ladder\sft",
    "dpo": r"D:\models\qwen15-ladder\dpo",
    "dpo_frame": r"D:\models\qwen15-ladder\dpo_frame",
    "rlvr": r"D:\models\qwen15-ladder\rlvr",
}
OUT_DIR = os.path.join(THIS, "results", "qwen15")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["base"])
    ap.add_argument("--dump-frames", action="store_true")
    args = ap.parse_args()

    tasks = get_tasks()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    frame_data = {}
    for stage in args.stages:
        if stage not in MODEL_DIRS:
            print(f"[skip] unknown stage {stage}")
            continue
        print(f"\n=== {stage} ===", flush=True)
        t0 = time.time()
        model, tok = load_model(MODEL_DIRS[stage])
        stage_res = {}
        stage_frames = {}
        for tid, task in tasks.items():
            r = compute_phi(model, tok, task)
            stage_res[tid] = r
            print(f"  {tid}: phi={r['phi']:.3f}  FS={r['FS']:.4f}  CS={r['CS']:.4f}", flush=True)
            if args.dump_frames:
                all_f = r["frame_dists"]
                stage_frames[tid] = {
                    "frames": all_f[:-4],
                    "heldout_frames": all_f[-4:],
                    "answer": r["answer"],
                    "content": task["content"],
                }
        results[stage] = stage_res
        if args.dump_frames:
            frame_data[stage] = stage_frames
            fo = os.path.join(OUT_DIR, f"frame_level_{stage}.json")
            with open(fo, "w", encoding="utf-8") as f:
                json.dump(stage_frames, f, ensure_ascii=False, indent=2)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)

        with open(os.path.join(OUT_DIR, "phi_ladder_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        import gc
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        print("  [VRAM released]", flush=True)

    summary = {}
    for stage, st in results.items():
        phis = [st[t]["phi"] for t in tasks]
        summary[stage] = {
            "phi_mean": float(np.mean(phis)),
            "phi_sem": float(np.std(phis) / np.sqrt(len(phis))),
            "fs_mean": float(np.mean([st[t]["FS"] for t in tasks])),
            "cs_mean": float(np.mean([st[t]["CS"] for t in tasks])),
        }
        print(f"\n{stage}: mean phi = {summary[stage]['phi_mean']:.3f} (SEM {summary[stage]['phi_sem']:.3f})")

    with open(os.path.join(OUT_DIR, "phi_ladder.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'phi_ladder.json')}")


if __name__ == "__main__":
    main()
