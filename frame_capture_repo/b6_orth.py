# -*- coding: utf-8 -*-
"""B6: Step 2 正交化消融 —— 回答"表示层可分不可分".

问题（FC_REFILL_PLAN.md §5 Step 2）:
  原做法 ablate_shared 删掉整个 14 帧方向子空间 → ΔFS ↓ 但 ΔCS 也一起塌（−0.25 ~ −0.45）。
  新做法 只删 u_f 中与 u_c 正交的那一部分:  u_f⊥ = u_f − Π_{u_c}(u_f)
  判据: ΔCS → ≈0 同时 ΔFS 保持负  ⇒ 表示层可分, Step 3 有理论许可。

条件:
  baseline            无干预
  ablate_shared       P = 全部 14 帧方向(其余任务) top-3 主子空间             [复现 v2]
  ablate_orth         P = 帧方向正交化到内容子空间后 top-3 主子空间           [新]
  ablate_orth_matched P = ablate_shared 的同组方向投影掉内容分量(Gram-Schmidt) [新,堵秩选择混淆]
  ablate_orthrandom   P = 与 ablate_orth 同秩的随机正交基                     [新对照]
  ablate_content      P = 内容方向(其余任务) top-3 主子空间                    [复现 v2]

层: 与 b2_patch v2 一致, l_hi = round(0.875*(L-1)), l_mid = round(0.75*(L-1)).
方向估计用 leave-one-task-out (LOTO)。渲染 17 条 (neutral + 14 帧 + 2 内容扰动)。
输出: results/b6_orth_{model}.json（断点续跑；v2 起含 matched 条件与 SVD 缓存）
用法: D:/conda_envs/flashhead/python.exe b6_orth.py [model ...]
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
from t3_unified_e1 import choice_probs_dual  # noqa: E402
from b6_math import (load_hidden_dirs, topk_subspace, content_subspace,  # noqa: E402
                     orth_to_content, orth_matched, random_orthonormal, K_FRAME)

MODELS = {
    "qwen15-sft":  r"D:\models\qwen15-ladder\sft",
    "qwen15-dpo":  r"D:\models\qwen15-ladder\dpo",
    "llama32-sft": r"D:\models\llama32-ladder\sft",
    "llama32-dpo": r"D:\models\llama32-ladder\dpo",
}
HEARTBEAT = os.path.join(THIS, "results", "_vb_heartbeat.json")
CONDITIONS = ["baseline", "ablate_shared", "ablate_orth",
              "ablate_orth_matched", "ablate_orthrandom", "ablate_content"]
SEED_ORTHRND = 20260919


def heartbeat(model, cond, layer, tid, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B6", "script": "b6_orth.py", "model": model,
                   "cond": cond, "layer": layer, "task": tid,
                   "done": done, "total": total,
                   "ts": datetime.now().isoformat()}, fh, ensure_ascii=False)


def load_bf16(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cuda:0", low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return model, tok


def prep_orth(loto):
    """每层每任务一次性算好所有消融矩阵（SVD 缓存，避免逐条件重复算）.

    返回 {tid: {'shared': (k,dim), 'content': (r,dim), 'orth': (k,dim),
                'orthk': int, 'matched': (k',dim)}}
    """
    tids = sorted(loto)
    prep = {}
    for t in tids:
        others = [x for x in tids if x != t]
        C = content_subspace(loto, others)
        M = np.concatenate([loto[x]["frames"] for x in others], axis=0)
        Psh = topk_subspace(M, K_FRAME)
        Morth = np.concatenate([orth_to_content(loto[x]["frames"], C)
                                for x in others], axis=0)
        Porth = topk_subspace(Morth, K_FRAME)
        prep[t] = {"shared": Psh, "content": C, "orth": Porth,
                   "orthk": int(Porth.shape[0]), "matched": orth_matched(Psh, C)}
    return prep


def load_or_build_prep(name, l, loto):
    """prep 磁盘缓存：崩进程/续跑时不再重算 20 分钟的 SVD。"""
    fields = ["shared", "content", "orth", "matched"]
    cache = os.path.join(THIS, "results", f"b6_prep_{name}_L{l}.npz")
    tids = sorted(loto)
    if os.path.exists(cache):
        try:
            z = np.load(cache)
            prep = {t: {f: z[f"{t}::{f}"] for f in fields} for t in tids}
            for t in tids:
                prep[t]["orthk"] = int(prep[t]["orth"].shape[0])
            print(f"  prep cache hit: {cache}", flush=True)
            return prep
        except Exception as ex:
            print(f"  [prep cache corrupt, rebuild] {ex}", flush=True)
    prep = prep_orth(loto)
    tmp = cache + ".tmp.npz"
    np.savez_compressed(
        tmp, **{f"{t}::{f}": prep[t][f] for t in tids for f in fields})
    os.replace(tmp, cache)
    print(f"  prep built & cached: {cache}", flush=True)
    return prep


def save_json(out, out_path):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    os.replace(tmp, out_path)


def make_intervention(cond, prep_t, dim):
    if cond == "baseline":
        return None
    if cond == "ablate_shared":
        return _hook_project(prep_t["shared"])
    if cond == "ablate_content":
        return _hook_project(prep_t["content"])
    if cond == "ablate_orth":
        return _hook_project(prep_t["orth"])
    if cond == "ablate_orth_matched":
        return _hook_project(prep_t["matched"])
    if cond == "ablate_orthrandom":
        return _hook_project(random_orthonormal(dim, prep_t["orthk"], SEED_ORTHRND))
    raise ValueError(cond)


def _hook_project(P):
    tP = torch.tensor(P, dtype=torch.float32, device="cuda")

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        orig_dtype = hs.dtype
        h = hs.float()
        if tP.shape[0] > 0:
            h = h - (h @ tP.T) @ tP
        hs2 = h.to(orig_dtype)
        if isinstance(output, tuple):
            return (hs2,) + tuple(output[1:])
        return hs2
    return hook


def run(name):
    out_path = os.path.join(THIS, "results", f"b6_orth_{name}.json")
    out = {"model": name, "protocol": "b6-orthogonalized-ablation-v2", "results": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            out = json.load(fh)
    tasks = get_tasks48()
    all_tids = sorted(tasks)
    print(f"\n=== B6 {name} (v2: +matched, SVD cache) ===", flush=True)
    model, tok = load_bf16(MODELS[name])
    n_layers = len(model.model.layers)
    l_hi, l_mid = round(0.875 * (n_layers - 1)), round(0.75 * (n_layers - 1))
    layers = sorted({l_hi, l_mid})
    print(f"layers={layers} of {n_layers}", flush=True)

    total = len(layers) * len(CONDITIONS) * len(all_tids)
    done_saved = sum(len(v) for v in out["results"].values())
    done = done_saved
    for l in layers:
        root = os.path.join(THIS, "results", f"b1_hidden_{name}")
        loto = load_hidden_dirs(root, l + 1)
        prep = load_or_build_prep(name, l, loto)
        for cond in CONDITIONS:
            key = f"L{l}:{cond}"
            res = out["results"].get(key, {})
            todo = [tid for tid in all_tids if tid not in res]
            if not todo:
                continue
            print(f"  {key}: {len(todo)} tasks to run", flush=True)
            for tid in todo:
                task = tasks[tid]
                dim = prep[tid]["content"].shape[1]
                hook_fn = make_intervention(cond, prep[tid], dim)
                hnd = model.model.layers[l].register_forward_hook(hook_fn) if hook_fn else None
                try:
                    renders = ([task["content"]] + list(task["frames"])
                               + list(task["content_perturb"]))
                    ps = []
                    for r in renders:
                        d = choice_probs_dual(model, tok, r + "\nAnswer:")[0]
                        ps.append(None if d is None else d.tolist())
                    res[tid] = ps
                except Exception as ex:
                    print(f"  [err] {key} {tid}: {ex}", flush=True)
                    res[tid] = [None] * 17
                finally:
                    if hnd:
                        hnd.remove()
                done += 1
                if done % 8 == 0:
                    heartbeat(name, cond, l, tid, done, total)
                    out["results"][key] = res
                    save_json(out, out_path)   # 每 8 任务落盘，崩了最多丢 8 个
            out["results"][key] = res
            save_json(out, out_path)
            print(f"  {key} done ({done}/{total} cumulative)", flush=True)
    heartbeat(name, "ALL", 0, "-", done, total)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        if n not in MODELS:
            print(f"[unknown] {n}", flush=True)
            continue
        run(n)
    print("B6 ORTH DONE", flush=True)
