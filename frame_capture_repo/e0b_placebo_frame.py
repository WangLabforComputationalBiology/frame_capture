# -*- coding: utf-8 -*-
"""E0b: 安慰剂 frame —— 跨任务 frame 置换 + 选项顺序重映射.

安慰剂设计:
  1. 跨任务 frame 置换: 对每个任务，用其他任务的 frame 分布替换其帧，
     保留 frame 数量不变。这打破 task-specific 的 frame-content 关系，
     但保留全局分布特性。
  2. 选项顺序重映射: 随机置换 A/B/C 标签后重算 FS，检验 FS 是否
     依赖于选项字母标记而非实际概率分配。

用法:
    D:/conda_envs/flashhead/python.exe e0b_placebo_frame.py
"""
import json
import os
import sys

sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")

NAMES = ["qwen15-base", "qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame",
         "qwen15-rlvr", "llama32-3b-base", "llama32-3b-instruct",
         "llama32-sft", "llama32-dpo",
         "internlm2-base", "internlm2-chat", "qwen25-7b-instruct",
         "gemma3-4b-it", "qwen25-7b-base"]

RNG = np.random.default_rng(42)


def jsd(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = np.maximum(0.5 * (p + q), 1e-15)
    eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def load_one(name):
    fp = os.path.join(R, f"phi_unified48_{name}.json")
    if not os.path.exists(fp):
        return None
    return json.load(open(fp, encoding="utf-8"))["detail"]


def main():
    out = {}
    for name in NAMES:
        detail = load_one(name)
        if detail is None:
            print(f"[skip] {name} not found")
            continue

        # 收集所有任务的帧分布字典
        task_frames = {}
        for tid, t in detail.items():
            fr = [np.array(x) for x in t["frames_first"] if x is not None]
            if len(fr) >= 2:
                task_frames[tid] = fr

        tids = list(task_frames.keys())
        n_task = len(tids)
        if n_task < 5:
            print(f"  {name}: only {n_task} tasks, skip")
            continue

        # 真实 FS（论文方法）
        real_fs = []
        for tid in tids:
            fr = task_frames[tid]
            ref = np.mean(fr, axis=0)
            real_fs.append(float(np.mean([jsd(f, ref) for f in fr])))
        real_fs_mean = float(np.mean(real_fs))

        # 安慰剂 1: 跨任务 frame 置换
        n_perm = 1000
        placebo_xfer = []
        for _ in range(n_perm):
            task_fs = []
            for tid in tids:
                n = len(task_frames[tid])
                # 随机从其他任务抽取 n 个帧分布
                pool = [v for k, v in task_frames.items() if k != tid]
                all_frames = [f for sub in pool for f in sub]
                if len(all_frames) < n:
                    continue
                chosen = [all_frames[i] for i in RNG.choice(len(all_frames), n, replace=True)]
                ref = np.mean(chosen, axis=0)
                task_fs.append(float(np.mean([jsd(f, ref) for f in chosen])))
            if task_fs:
                placebo_xfer.append(np.mean(task_fs))

        # 安慰剂 2: 选项顺序重映射（随机置换 A/B/C）
        from itertools import permutations
        all_perms = list(permutations([0, 1, 2]))
        placebo_swap = []
        for _ in range(n_perm):
            task_fs = []
            for tid in tids:
                fr = task_frames[tid]
                pi = all_perms[RNG.integers(len(all_perms))]
                fr_swapped = [np.array([x[pi[0]], x[pi[1]], x[pi[2]]]) for x in fr]
                ref = np.mean(fr_swapped, axis=0)
                task_fs.append(float(np.mean([jsd(f, ref) for f in fr_swapped])))
            placebo_swap.append(np.mean(task_fs))

        # 安慰剂 3: Dirichlet 噪声（从帧均值出发的随机采样）
        placebo_dir = []
        for _ in range(n_perm):
            task_fs = []
            for tid in tids:
                fr = task_frames[tid]
                ref = np.mean(fr, axis=0)
                n = len(fr)
                noisy = []
                for _ in range(n):
                    noisy.append(RNG.dirichlet(ref * 100 + 1))
                ref_n = np.mean(noisy, axis=0)
                task_fs.append(float(np.mean([jsd(f, ref_n) for f in noisy])))
            if task_fs:
                placebo_dir.append(np.mean(task_fs))

        out[name] = {
            "n_tasks": n_task,
            "real_fs_mean": real_fs_mean,
            "placebo_cross_task": {
                "mean": float(np.mean(placebo_xfer)),
                "std": float(np.std(placebo_xfer)),
                "pct99": float(np.percentile(placebo_xfer, 99)),
            },
            "placebo_option_swap": {
                "mean": float(np.mean(placebo_swap)),
                "std": float(np.std(placebo_swap)),
                "pct99": float(np.percentile(placebo_swap, 99)),
            },
            "placebo_dirichlet": {
                "mean": float(np.mean(placebo_dir)),
                "std": float(np.std(placebo_dir)),
                "pct99": float(np.percentile(placebo_dir, 99)),
            },
        }
        print(f"  {name:24} real={real_fs_mean:.5f}  "
              f"xfer={out[name]['placebo_cross_task']['mean']:.5f}  "
              f"swap={out[name]['placebo_option_swap']['mean']:.5f}  "
              f"dir={out[name]['placebo_dirichlet']['mean']:.5f}", flush=True)

    out_path = os.path.join(R, "e0b_placebo.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {out_path}")

    # 判据
    print("\n=== E0b 判据 ===")
    for name, v in out.items():
        ok = v["real_fs_mean"] > v["placebo_cross_task"]["pct99"]
        print(f"  {name:24} {'OK: real > xfer p99' if ok else 'WARN: real <= xfer p99'}")
    print("\n判据: 全部 14 检查点 real FS > 跨任务安慰剂 99% 分位 → 管道伪影攻击失败。")


if __name__ == "__main__":
    main()