# -*- coding: utf-8 -*-
"""B0c: held-out 电池 runner —— 24 新任务 × (14 std + 12 new + 2 nat + 2 perturb)
× 3 个等价重复采样配置（seed 0=canonical，seed 1/2=固定选项排列），bf16 原生权重。

协议与 t5_unified48.py 一致: 每条 prompt + "\nAnswer:"，choice_probs_dual 取
first-agg 3 选项分布（tot<=0.5 → None）。选项排列只重排 A/B/C 的内容位置，
FS/CS 分析不依赖答案身份，故构成"等价重复采样配置"（B0 判据的多 seed 项）。
每个 (seed, task) 的排列由 rng(1000*seed+7) 一次性生成，续跑时排列不变。

可断点续跑: results/b0_heldout_{name}.json 每 (model, seed, task) 原子落盘一次，
重跑时已有完整条目自动跳过，模型全部完成则整模型跳过。
心跳写 results/_vb_heartbeat.json（看门狗死活判定依据）。

用法:
    D:/conda_envs/flashhead/python.exe b0_heldout_run.py qwen15-base qwen15-sft qwen15-dpo
（无参数 = 全部 6 个模型；llama32 3B bf16 约 6.4GB 可装入 10GB 卡）
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
from items import make_frames, make_perturbs  # noqa: E402
from items_b0 import get_tasks_b0, _new_frames, _natural_frames  # noqa: E402
from t3_unified_e1 import choice_probs_dual  # noqa: E402

MODELS = {
    "qwen15-base":     r"D:\models\Qwen2.5-1.5B",
    "qwen15-sft":      r"D:\models\qwen15-ladder\sft",
    "qwen15-dpo":      r"D:\models\qwen15-ladder\dpo",
    "llama32-3b-base": r"D:\models\Llama-3.2-3B-base",
    "llama32-sft":     r"D:\models\llama32-ladder\sft",
    "llama32-dpo":     r"D:\models\llama32-ladder\dpo",
}
SEEDS = [0, 1, 2]
N_STD, N_NEW, N_NAT, N_PERT = 14, 12, 2, 2
HEARTBEAT = os.path.join(THIS, "results", "_vb_heartbeat.json")


def heartbeat(model, seed, tid, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B0c", "script": "b0_heldout_run.py", "model": model,
                   "seed": seed, "task": tid, "done": done, "total": total,
                   "ts": datetime.now().isoformat()}, fh, ensure_ascii=False)


def load_bf16(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cuda:0", low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return model, tok


def task_done(entry):
    return (isinstance(entry, dict)
            and len(entry.get("frames_std") or []) == N_STD
            and len(entry.get("frames_new") or []) == N_NEW
            and len(entry.get("frames_nat") or []) == N_NAT
            and len(entry.get("perturbs") or []) == N_PERT)


def render(tid, perm, tasks_raw, task_cache):
    """返回 (frames_std[14], frames_new[12], frames_nat[2], perturbs[2])。"""
    if perm is None:
        tc = task_cache[tid]
        return tc["frames_std"], tc["frames_new"], tc["frames_nat"], tc["perturb_prompts"]
    q, options, answer, perturbs = tasks_raw[tid]
    opts = [options[i] for i in perm]
    frames_std = make_frames(q, opts)
    frames_new = _new_frames(q, opts)
    frames_nat = _natural_frames(q, opts)
    pert = make_perturbs(q, opts, answer,
                         [(pc, [po[i] for i in perm], pa) for pc, po, pa in perturbs])
    return frames_std, frames_new, frames_nat, pert


def run(name, tasks_raw, task_cache, readout="bf16"):
    suffix = "" if readout == "bf16" else f"_{readout}"
    out_path = os.path.join(THIS, "results", f"b0_heldout_{name}{suffix}.json")
    out = {"model": name, "protocol": "b0-heldout-v1", "readout": readout,
           "seeds": {str(s): {} for s in SEEDS}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        out["seeds"] = prev.get("seeds", out["seeds"])
        if all(all(task_done(out["seeds"][str(s)].get(tid)) for tid in tasks_raw)
               for s in SEEDS):
            print(f"[skip] {name} all seeds/tasks done", flush=True)
            return
    print(f"\n=== {name} ({readout}: {MODELS[name]}) ===", flush=True)
    if readout == "nf4":
        from t3_unified_e1 import load_model
        model, tok = load_model("std4bit", MODELS[name])
    else:
        model, tok = load_bf16(MODELS[name])
    print(f"loaded, tok={type(tok).__name__}", flush=True)

    total = len(SEEDS) * len(tasks_raw)
    done = sum(1 for s in SEEDS for tid in tasks_raw
               if task_done(out["seeds"][str(s)].get(tid)))
    for seed in SEEDS:
        sd = out["seeds"][str(seed)]
        rng = np.random.RandomState(1000 * seed + 7)
        perm_of = {tid: (None if seed == 0 else
                         tuple(int(v) for v in rng.permutation(3)))
                   for tid in tasks_raw}
        for tid in tasks_raw:
            if task_done(sd.get(tid)):
                continue
            frames_std, frames_new, frames_nat, pert = render(
                tid, perm_of[tid], tasks_raw, task_cache)
            try:
                d_f = [choice_probs_dual(model, tok, f + "\nAnswer:")[0] for f in frames_std]
                d_n = [choice_probs_dual(model, tok, f + "\nAnswer:")[0] for f in frames_new]
                d_t = [choice_probs_dual(model, tok, f + "\nAnswer:")[0] for f in frames_nat]
                d_p = [choice_probs_dual(model, tok, p + "\nAnswer:")[0] for p in pert]
                sd[tid] = {
                    "frames_std": [None if x is None else x.tolist() for x in d_f],
                    "frames_new": [None if x is None else x.tolist() for x in d_n],
                    "frames_nat": [None if x is None else x.tolist() for x in d_t],
                    "perturbs": [None if x is None else x.tolist() for x in d_p],
                    "perm": list(perm_of[tid]) if perm_of[tid] is not None else None,
                }
            except Exception as ex:
                print(f"  [err] seed{seed} {tid}: {ex}", flush=True)
                sd[tid] = {"frames_std": [None] * N_STD, "frames_new": [None] * N_NEW,
                           "frames_nat": [None] * N_NAT, "perturbs": [None] * N_PERT,
                           "perm": None, "error": str(ex)}
            done += 1
            valid = sum(1 for x in sd[tid]["frames_std"] + sd[tid]["frames_new"]
                        if x is not None)
            print(f"  seed{seed} {tid}: valid={valid}/{N_STD + N_NEW}", flush=True)
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, out_path)
            heartbeat(name, seed, tid, done, total)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    from items_b0 import _TASKS_B0
    tasks_raw = {tid: (c, o, a, p) for tid, c, o, a, p in _TASKS_B0}
    task_cache = get_tasks_b0()
    readout = "bf16"
    if "--nf4" in sys.argv:
        readout = "nf4"
        sys.argv.remove("--nf4")
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        if n not in MODELS:
            print(f"[unknown model] {n}", flush=True)
            continue
        run(n, tasks_raw, task_cache, readout=readout)
    print("\nALL DONE", flush=True)
