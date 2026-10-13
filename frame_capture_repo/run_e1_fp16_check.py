# -*- coding: utf-8 -*-
"""4-bit 校准对照：Qwen2.5-1.5B base 以未量化 fp16 重跑 E1 协议，
与 4-bit 结果（phi_ladder_full5.json 的 base 腿）对账。"""
import json
import os
import sys

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from run_e1 import compute_phi  # noqa: E402
from items import get_tasks  # noqa: E402

MODEL_DIR = r"D:\models\Qwen2.5-1.5B"
OUT = os.path.join(THIS, "results", "qwen15", "phi_base_fp16.json")


def load_fp16(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def main():
    tasks = get_tasks()
    model, tok = load_fp16(MODEL_DIR)
    res = {}
    for tid, task in tasks.items():
        r = compute_phi(model, tok, task)
        res[tid] = {"FS": r["FS"], "CS": r["CS"], "phi": r["phi"]}
        print(f"  {tid}: phi={r['phi']:.3f}  FS={r['FS']:.4f}  CS={r['CS']:.4f}", flush=True)
    phis = [v["phi"] for v in res.values()]
    summary = {
        "phi_mean": float(np.mean(phis)),
        "phi_sem": float(np.std(phis) / np.sqrt(len(phis))),
        "fs_mean": float(np.mean([v["FS"] for v in res.values()])),
        "cs_mean": float(np.mean([v["CS"] for v in res.values()])),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": res}, f, ensure_ascii=False, indent=2)
    print("\nfp16 base:", summary)


if __name__ == "__main__":
    main()
