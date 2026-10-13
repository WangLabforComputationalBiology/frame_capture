# -*- coding: utf-8 -*-
"""E0a: 管道噪声零点 —— byte-identical 重复推理 10× 测 JSD 噪声 floor.

方法: 取 tasks_v3 的 48 个 content prompt（无 frame），
对每个 prompt 做 10× 重复推理（option-level dist），
所有 45 个上三角 JSD 的均值 = 该任务的管道噪声；
全 48 任务均值的均值 = 管道噪声零点。

用法:
    D:/conda_envs/flashhead/python.exe e0a_pipeline_noise.py qwen15-base
    D:/conda_envs/flashhead/python.exe e0a_pipeline_noise.py qwen15-base --roi 8  # 仅 8 任务快速预检
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


def load_model(kind, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if kind == "std4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=False,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    elif kind == "gemma_bf16":
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="cuda:0", low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, trust_remote_code=True)
    elif kind == "internlm2":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=False,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        raise ValueError(f"unknown kind: {kind}")
    model.eval()
    return model


def load_tok(kind, path):
    from transformers import AutoTokenizer
    if kind == "internlm2":
        return im2tok.build_tokenizer(model_dir=os.path.realpath(path))
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def choice_probs(model, tok, prompt):
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


MODEL_MAP = {
    "qwen15-base": ("std4bit", r"D:\models\Qwen2.5-1.5B"),
    "qwen15-sft": ("std4bit", r"D:\models\qwen15-ladder\sft"),
    "qwen15-dpo": ("std4bit", r"D:\models\qwen15-ladder\dpo"),
    "qwen15-dpo_frame": ("std4bit", r"D:\models\qwen15-ladder\dpo_frame"),
    "qwen15-rlvr": ("std4bit", r"D:\models\qwen15-ladder\rlvr"),
    "llama32-3b-base": ("std4bit", r"D:\models\Llama-3.2-3B-base"),
    "llama32-3b-instruct": ("std4bit", r"D:\models\Llama-3.2-3B-Instruct"),
    "llama32-sft": ("std4bit", r"D:\models\llama32-ladder\sft"),
    "llama32-dpo": ("std4bit", r"D:\models\llama32-ladder\dpo"),
    "internlm2-base": ("internlm2", r"D:\models\internlm2_1_8b_base"),
    "internlm2-chat": ("internlm2", r"D:\models\internlm2_1_8b_chat"),
    "gemma3-4b-it": ("gemma_bf16", r"D:\models\Gemma-3-4B-it"),
    "qwen25-7b-base": ("std4bit", r"D:\models\Qwen2.5-7B-base"),
    "qwen25-7b-instruct": ("std4bit", r"D:\models\Qwen2.5-7B-Instruct"),
}

N_REPEAT = 10


def run(name, roi=None):
    if name not in MODEL_MAP:
        print(f"unknown model {name}")
        return
    kind, path = MODEL_MAP[name]
    out_path = os.path.join(THIS, "results", f"e0a_noise_{name}.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists", flush=True)
        return

    print(f"loading {name} ({kind}: {path}) ...", flush=True)
    model = load_model(kind, path)
    tok = load_tok(kind, path)
    print(f"loaded, tok={type(tok).__name__}", flush=True)

    tasks = get_tasks48()
    if roi:
        tasks = {k: tasks[k] for k in list(tasks)[:roi]}

    results = {}
    for tid, task in tasks.items():
        prompt = task["content"] + "\nAnswer:"
        reps = []
        for rep in range(N_REPEAT):
            d = choice_probs(model, tok, prompt)
            if d is None:
                reps.append(None)
            else:
                reps.append(d)
        valid = [r for r in reps if r is not None]
        nv = len(valid)
        jsds = []
        if nv >= 2:
            for i in range(nv):
                for j in range(i + 1, nv):
                    jsds.append(jsd(valid[i], valid[j]))
        mu = float(np.mean(jsds)) if jsds else None
        sd = float(np.std(jsds)) if jsds else None
        results[tid] = {
            "n_valid": nv,
            "mean_jsd": mu,
            "std_jsd": sd,
            "n_pairs": len(jsds),
            "all_probs": [r.tolist() if r is not None else None for r in reps],
        }
        print(f"  {tid}: n_valid={nv}/{N_REPEAT} mean_jsd={mu:.6f} std={sd:.6f}",
              flush=True)

    all_means = [v["mean_jsd"] for v in results.values() if v["mean_jsd"] is not None]
    summary = {
        "model": name,
        "n_tasks": len(results),
        "n_repeat": N_REPEAT,
        "overall_mean_jsd": float(np.mean(all_means)) if all_means else None,
        "overall_median_jsd": float(np.median(all_means)) if all_means else None,
        "overall_std_jsd": float(np.std(all_means)) if all_means else None,
        "min_mean_jsd": float(min(all_means)) if all_means else None,
        "max_mean_jsd": float(max(all_means)) if all_means else None,
        "per_task": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved {out_path}", flush=True)
    print(f"=== OVERALL pipeline noise floor: mean JSD = {summary['overall_mean_jsd']:.6f} bits ===", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODEL_MAP))
    ap.add_argument("--roi", type=int, default=None, help="subset size for quick check")
    args = ap.parse_args()
    run(args.model, roi=args.roi)