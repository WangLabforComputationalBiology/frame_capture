# -*- coding: utf-8 -*-
"""B2: 跨家族 activation patching —— 消融 / 注入 / 跨家族替换（bf16，可断点续跑）.

前置: b1_collect.py 已产出 results/b1_hidden_{model}/{tid}.npy。
方向估计用 leave-one-task-out（LOTO）防泄漏: 对任务 t 的干预，
共享子空间/家族方向只用其余 47 个任务的 hidden states 估计。

干预（在指定层的 decoder layer 输出 = residual stream 上实施）:
  baseline          无干预
  ablate_shared     h' = h - P P^T h，P = 全部 14 帧方向（其余任务）top-3 主子空间
  ablate_random     同规模随机正交子空间（固定种子）
  ablate_content    P = 内容方向（2 改答案扰动）top-3 主子空间（对照: 不应伤帧效应）
  inject_<fam>      h' = h + alpha * v_f，v_f = 家族 f 平均方向（其余任务，L2 归一），
                    alpha = 其余任务方向范数中位数（匹配天然幅度）
  xpatch_<A>_to_<B> h' = h + v_A - v_B（把 A 家族方向加进来、B 家族方向去掉）

层: l_hi = round(0.875*(L-1)) 主层 + l_mid = round(0.75*(L-1)) 次层。
渲染: 每任务 17 条（neutral + 14 帧 + 2 扰动），choice_probs_dual first-agg。

指标（每模型×层×条件）: fs, cs, flip_rate（对 baseline argmax 众数）,
neutral_acc（argmax==A，48 任务 canonical 答案 A）, flip_induction（inject 使
neutral argmax 翻转的比例）, xpatch_transfer（xpatch 后 argmax 与
inject_<A> 后 argmax 的一致率）。
输出: results/b2_patch_{model}.json；心跳 results/_vb_heartbeat.json。
用法: D:/conda_envs/flashhead/python.exe b2_patch.py [model ...]（默认 4 个 ladder 模型）
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

MODELS = {
    "qwen15-sft":  r"D:\models\qwen15-ladder\sft",
    "qwen15-dpo":  r"D:\models\qwen15-ladder\dpo",
    "llama32-sft": r"D:\models\llama32-ladder\sft",
    "llama32-dpo": r"D:\models\llama32-ladder\dpo",
}
HEARTBEAT = os.path.join(THIS, "results", "_vb_heartbeat.json")
FAM_OF = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]
XPATCH = [("authority", "persona"), ("persona", "authority"), ("peer", "verbal")]
CONDITIONS = (["baseline", "ablate_shared", "ablate_random", "ablate_content"]
              + [f"inject_{f}" for f in FAMS]
              + [f"xpatch_{a}_to_{b}" for a, b in XPATCH])


def heartbeat(model, cond, layer, tid, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B2v2", "script": "b2_patch.py", "model": model,
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


def load_hidden_dirs(name, layer_stack_idx=-1):
    """返回 {tid: {'fam': {f: dir}, 'content': dir, 'frames': (14,dim), 'neutral': dir}}

    layer_stack_idx: b1_hidden npy 的层索引（hidden_states 编号；干预 decoder 层 l
    对应 hidden_states[l+1]）。默认 -1 = 最后一层（v1 行为，有层错配缺陷）。
    """
    d = {}
    root = os.path.join(THIS, "results", f"b1_hidden_{name}")
    for fp in sorted(os.listdir(root)):
        if not fp.endswith(".npy"):
            continue
        tid = fp[:-4]
        arr = np.load(os.path.join(root, fp))[:, layer_stack_idx, :].astype(np.float64)
        neutral = arr[0]
        frames = arr[1:15] - neutral
        fam = {f: frames[[j for j in range(14) if FAM_OF[j] == f]].mean(axis=0)
               for f in FAMS}
        content = (arr[15:17] - neutral).mean(axis=0)
        d[tid] = {"fam": fam, "content": content, "frames": frames, "neutral": neutral}
    return d


def topk_subspace(mat, k=3):
    """mat: (n, dim) 行为方向；返回 (k, dim) 主子空间基（列中心化）。"""
    X = mat - mat.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


def random_orthonormal(dim, k, seed):
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(dim, k))
    return Q.T


def make_intervention(cond, loto_dirs, target_tid, layer):
    """返回 hook 函数（修改 decoder layer 输出 hidden_states）或 None。"""
    others = [t for t in loto_dirs if t != target_tid]
    dim = loto_dirs[target_tid]["neutral"].shape[0]
    if cond == "baseline":
        return None
    if cond == "ablate_shared":
        M = np.concatenate([loto_dirs[t]["frames"] for t in others], axis=0)
        P = topk_subspace(M, 3)
    elif cond == "ablate_random":
        P = random_orthonormal(dim, 3, seed=20260911)
    elif cond == "ablate_content":
        M = np.stack([loto_dirs[t]["content"] for t in others])
        P = topk_subspace(M, 3)
    elif cond.startswith("inject_"):
        f = cond[len("inject_"):]
        V = np.stack([loto_dirs[t]["fam"][f] for t in others])
        v = V.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        alpha = float(np.median([np.linalg.norm(loto_dirs[t]["fam"][f]) for t in others]))
        P, add = v[None, :], alpha * v
        return _hook_project_add(P, add)
    elif cond.startswith("xpatch_"):
        body = cond[len("xpatch_"):]
        a, b = body.split("_to_")
        V_a = np.stack([loto_dirs[t]["fam"][a] for t in others])
        V_b = np.stack([loto_dirs[t]["fam"][b] for t in others])
        va = V_a.mean(axis=0); va /= (np.linalg.norm(va) + 1e-12)
        vb = V_b.mean(axis=0); vb /= (np.linalg.norm(vb) + 1e-12)
        alpha_a = float(np.median([np.linalg.norm(loto_dirs[t]["fam"][a]) for t in others]))
        alpha_b = float(np.median([np.linalg.norm(loto_dirs[t]["fam"][b]) for t in others]))
        return _hook_project_add(np.zeros((0, dim)), alpha_a * va - alpha_b * vb)
    else:
        raise ValueError(cond)
    return _hook_project_add(P, np.zeros(dim))


def _hook_project_add(P, add_vec):
    tP = torch.tensor(P, dtype=torch.float32, device="cuda")
    tadd = torch.tensor(add_vec, dtype=torch.float32, device="cuda")

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        orig_dtype = hs.dtype
        h = hs.float()
        if tP.shape[0] > 0:
            proj = (h @ tP.T) @ tP
            h = h - proj
        h = h + tadd
        hs2 = h.to(orig_dtype)
        if isinstance(output, tuple):
            return (hs2,) + tuple(output[1:])
        return hs2
    return hook


def run(name):
    out_path = os.path.join(THIS, "results", f"b2_patch_{name}_v2.json")
    out = {"model": name, "protocol": "b2-patch-v2-layermatched", "results": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            out = json.load(fh)
    tasks = get_tasks48()
    all_tids = sorted(tasks)
    print(f"\n=== {name} (v2: layer-matched directions) ===", flush=True)
    model, tok = load_bf16(MODELS[name])
    n_layers = len(model.model.layers)
    l_hi, l_mid = round(0.875 * (n_layers - 1)), round(0.75 * (n_layers - 1))
    layers = sorted({l_hi, l_mid})
    print(f"layers={layers} of {n_layers}", flush=True)

    done = out.get("_done_count", 0)
    total = len(layers) * len(CONDITIONS) * len(all_tids)
    for l in layers:
        # v2 核心修复: 方向在干预同层估计（hidden_states[l+1]）
        loto = load_hidden_dirs(name, l + 1)
        for cond in CONDITIONS:
            key = f"L{l}:{cond}"
            res = out["results"].get(key, {})
            for tid in all_tids:
                if tid in res:
                    continue
                task = tasks[tid]
                hook_fn = make_intervention(cond, loto, tid, l)
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
            out["_done_count"] = done
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False)
            os.replace(tmp, out_path)
            print(f"  {key} done ({done}/{total})", flush=True)
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
    print("B2 PATCH DONE", flush=True)
