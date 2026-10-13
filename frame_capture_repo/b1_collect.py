# -*- coding: utf-8 -*-
"""B1 采集: 跨 frame family 表示分析 —— hidden state 采集（bf16，可断点续跑）.

协议: 48 任务电池（items_v3），每任务 19 条渲染，各接 "\nAnswer:":
  idx 0    = neutral（裸内容+选项）
  idx 1-14 = 14 个 std 帧（家族索引与 items.make_frames 一致）
  idx 15-16 = 2 个改答案内容扰动（content direction 用）
  idx 17-18 = 2 个 nonsense 词缀控制（无意义前缀）
每条记录全部层（hidden_states, L+1 个）的 last-token hidden state，fp16。
输出: results/b1_hidden_{name}/{tid}.npy  (19, L+1, dim)

心跳: results/_vb_heartbeat.json。用法:
    D:/conda_envs/flashhead/python.exe b1_collect.py qwen15-sft qwen15-dpo llama32-sft llama32-dpo
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

MODELS = {
    "qwen15-sft":  r"D:\models\qwen15-ladder\sft",
    "qwen15-dpo":  r"D:\models\qwen15-ladder\dpo",
    "llama32-sft": r"D:\models\llama32-ladder\sft",
    "llama32-dpo": r"D:\models\llama32-ladder\dpo",
    "qwen15-base": r"D:\models\Qwen2.5-1.5B",
    "llama32-3b-base": r"D:\models\Llama-3.2-3B-base",
}
HEARTBEAT = os.path.join(THIS, "results", "_vb_heartbeat.json")

NONSENSE = ["Zzz quibble flurb wibble. ", "Kk2# ff9 -- blorp smood. "]


def heartbeat(model, tid, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B1", "script": "b1_collect.py", "model": model,
                   "task": tid, "done": done, "total": total,
                   "ts": datetime.now().isoformat()}, fh, ensure_ascii=False)


def load_bf16(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cuda:0", low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return model, tok


def run(name):
    out_dir = os.path.join(THIS, "results", f"b1_hidden_{name}")
    os.makedirs(out_dir, exist_ok=True)
    tasks = get_tasks48()
    todo = [tid for tid in tasks if not os.path.exists(os.path.join(out_dir, f"{tid}.npy"))]
    print(f"\n=== {name}: {len(todo)}/{len(tasks)} tasks to collect ===", flush=True)
    if not todo:
        return
    model, tok = load_bf16(MODELS[name])
    done = len(tasks) - len(todo)
    total = len(tasks)
    with torch.no_grad():
        for tid, task in tasks.items():
            fp = os.path.join(out_dir, f"{tid}.npy")
            if os.path.exists(fp):
                continue
            neutral = task["content"]
            renders = ([neutral]
                       + list(task["frames"])
                       + list(task["content_perturb"])
                       + [ns + neutral for ns in NONSENSE])
            hs_list = []
            for r in renders:
                enc = tok(r + "\nAnswer:", return_tensors="pt")
                out = model(**{k: v.to("cuda") for k, v in enc.items()},
                            output_hidden_states=True)
                hs = out.hidden_states  # tuple(L+1) of (1, seq, dim)
                last = torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy()
                hs_list.append(last.astype(np.float16))
                del out, hs, enc
            stack = np.stack(hs_list)  # (19, L+1, dim)
            np.save(fp, stack)
            done += 1
            print(f"  {tid}: {stack.shape} ({done}/{total})", flush=True)
            heartbeat(name, tid, done, total)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"saved dir {out_dir}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        if n not in MODELS:
            print(f"[unknown] {n}", flush=True)
            continue
        run(n)
    print("B1 COLLECT DONE", flush=True)
