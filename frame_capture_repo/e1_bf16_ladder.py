# -*- coding: utf-8 -*-
"""E1: bf16 全梯子复测 —— 12 个 ≤4B 模型在 bf16 下测量 48 任务.

模型列表（≤4B，可装入 10GB 单卡 bf16）:
  - Qwen2.5-1.5B (3GB) + SFT/DPO/DPO_frame/RLVR
  - Llama-3.2-3B (6GB) + SFT/DPO
  - InternLM2-1.8B (3.6GB) + chat
  - Gemma-3-4B-it (8GB, 已 bf16 运行)

协议: 与 t3_unified_e1.py 相同（48 任务 × 14 帧 × 2 扰动）.
输出: results/phi_unified48_{name}_bf16.json

用法:
    D:/conda_envs/flashhead/python.exe e1_bf16_ladder.py qwen15-base
    D:/conda_envs/flashhead/python.exe e1_bf16_ladder.py --all
    D:/conda_envs/flashhead/python.exe e1_bf16_ladder.py --roi 8  # 快速预检
"""
import gc
import json
import os
import sys

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from items_v3 import get_tasks48
import _internlm2_tok as im2tok

FORMS = [(c, f" {c}", f"{c}.", f" {c}.", f"{c})", f" {c})", f"{c}:", f" {c}:") for c in "ABC"]


def jsd(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = np.maximum(0.5 * (p + q), 1e-15)
    eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def load_model_bf16(path, internlm2=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if internlm2:
        # InternLM2 base 需要特殊处理
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="cuda:0", low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, trust_remote_code=True)
        tok = im2tok.build_tokenizer(model_dir=os.path.realpath(path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="cuda:0", low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model.eval()
    return model, tok


def choice_probs_first(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to("cuda") for k, v in enc.items()})
    if not hasattr(out, "logits") or out.logits is None:
        return None
    logits = out.logits[0, -1, :].float().cpu().numpy()
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    probs = {}
    for ci, c in enumerate("ABC"):
        first_ids = set()
        for form in FORMS[ci]:
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                first_ids.add(ids[0])
        probs[c] = float(sum(p[i] for i in first_ids))
    tot = sum(probs.values())
    if tot <= 0.5:
        return None
    return np.array([probs[c] / tot for c in "ABC"])


# ≤4B 模型（可在 10GB 单卡 bf16 下运行）
BF16_MODELS = [
    ("qwen15-base", False, r"D:\models\Qwen2.5-1.5B"),
    ("qwen15-sft", False, r"D:\models\qwen15-ladder\sft"),
    ("qwen15-dpo", False, r"D:\models\qwen15-ladder\dpo"),
    ("qwen15-dpo_frame", False, r"D:\models\qwen15-ladder\dpo_frame"),
    ("qwen15-rlvr", False, r"D:\models\qwen15-ladder\rlvr"),
    ("llama32-3b-base", False, r"D:\models\Llama-3.2-3B-base"),
    ("llama32-3b-instruct", False, r"D:\models\Llama-3.2-3B-Instruct"),
    ("llama32-sft", False, r"D:\models\llama32-ladder\sft"),
    ("llama32-dpo", False, r"D:\models\llama32-ladder\dpo"),
    ("internlm2-base", True, r"D:\models\internlm2_1_8b_base"),
    ("internlm2-chat", True, r"D:\models\internlm2_1_8b_chat"),
    # Gemma-3-4B-it 已 bf16，不在 4-bit 影响范围内，跳过
]


def run(name, is_internlm2, path, roi=None):
    out_path = os.path.join(THIS, "results", f"phi_unified48_{name}_bf16.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists", flush=True)
        return

    print(f"\n=== {name} (bf16, internlm2={is_internlm2}) ===", flush=True)
    model, tok = load_model_bf16(path, internlm2=is_internlm2)
    print(f"loaded, tok={type(tok).__name__}", flush=True)

    tasks = get_tasks48()
    if roi:
        tasks = {k: tasks[k] for k in list(tasks)[:roi]}

    detail = {}
    for tid, task in tasks.items():
        fr = []
        for f in task["frames"]:
            d = choice_probs_first(model, tok, f + "\nAnswer:")
            fr.append(None if d is None else d.tolist())
        pe = []
        for cp in task["content_perturb"]:
            d = choice_probs_first(model, tok, cp + "\nAnswer:")
            pe.append(None if d is None else d.tolist())
        detail[tid] = {"frames_first": fr, "perturbs_first": pe}
        nv = sum(1 for x in fr if x is not None)
        npv = sum(1 for x in pe if x is not None)
        print(f"  {tid}: frames_valid={nv}/14 perturb_valid={npv}/2", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "protocol": "bf16-v1", "detail": detail},
                  f, ensure_ascii=False, indent=1)
    print(f"saved {out_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None, help="model name to run")
    ap.add_argument("--all", action="store_true", help="run all bf16 models")
    ap.add_argument("--roi", type=int, default=None, help="quick subset")
    args = ap.parse_args()

    if args.all:
        targets = BF16_MODELS
    elif args.model:
        matches = [m for m in BF16_MODELS if m[0] == args.model]
        if not matches:
            print(f"unknown model {args.model}. Available: {[m[0] for m in BF16_MODELS]}")
            sys.exit(1)
        targets = matches
    else:
        print("specify a model or --all")
        sys.exit(1)

    for name, is_internlm2, path in targets:
        run(name, is_internlm2, path, roi=args.roi)