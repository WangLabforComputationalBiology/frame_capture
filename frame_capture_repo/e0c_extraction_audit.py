# -*- coding: utf-8 -*-
"""E0c: 提取审计 —— per-family per-checkpoint 选项质量通过率.

方法: 从现有 48 任务原始数据读出每个 frame 变体的选项质量 tot，
统计 per-family（rephrase/case/persona/authority/peer/verbal）的
通过率（tot > 0.5）和 pass 分布变异系数。

如果某些 frame family 的系统性通过率偏低或变异大，
则风格漂移被计入 FS 的风险升高。

用法:
    D:/conda_envs/flashhead/python.exe e0c_extraction_audit.py
"""
import json
import os
import sys

sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")

FAM_OF = {0: "rephrase", 1: "rephrase", 2: "rephrase", 3: "case", 4: "case",
          5: "persona", 6: "persona", 7: "persona", 8: "authority", 9: "authority",
          10: "peer", 11: "peer", 12: "verbal", 13: "verbal"}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]

NAMES = ["qwen15-base", "qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame",
         "qwen15-rlvr", "llama32-3b-base", "llama32-3b-instruct",
         "llama32-sft", "llama32-dpo",
         "internlm2-base", "internlm2-chat", "qwen25-7b-instruct",
         "gemma3-4b-it", "qwen25-7b-base"]


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

        # 统计每个 frame family 的通过率
        fam_pass = {f: [] for f in FAMS}
        fam_total = {f: 0 for f in FAMS}
        fam_pass_count = {f: 0 for f in FAMS}

        for tid, t in detail.items():
            for idx, dist_first in enumerate(t["frames_first"]):
                fam = FAM_OF.get(idx)
                if fam is None:
                    continue
                fam_total[fam] += 1
                if dist_first is not None:
                    # 计算 tot（需从原始分布反向，但已有 first-agg 分布）
                    d = np.array(dist_first)
                    # 如果分布非空且 tot > 0.5 → 通过
                    tot = d.sum()  # first-agg 已经是归一化前的聚合
                    # 实际上 first-agg 已归一化，所以 tot = 1
                    # 但原始提取条件是在 tot > 0.5 前归一化
                    # 我们无法从归一化分布反推原始 tot
                    # 改用另一种方法：检查是否存在 None
                    fam_pass_count[fam] += 1

        # 修正：None 意味着该帧变体未通过 tot > 0.5 过滤
        # 直接统计每个 family 的 None 率
        fam_none = {f: [] for f in FAMS}
        fam_all = {f: [] for f in FAMS}
        for tid, t in detail.items():
            for idx, dist_first in enumerate(t["frames_first"]):
                fam = FAM_OF.get(idx)
                if fam is None:
                    continue
                fam_all[fam].append(1)
                fam_none[fam].append(1 if dist_first is None else 0)

        print(f"\n  {name}:")
        for fam in FAMS:
            n_all = len(fam_all[fam])
            n_none = sum(fam_none[fam])
            pass_rate = 1 - n_none / n_all if n_all > 0 else 0
            print(f"    {fam:12} pass_rate={pass_rate:.4f}  (n_all={n_all}, n_none={n_none})")

        # 也检查 perturb 的通过率
        p_pass = sum(1 for t in detail.values()
                     for p in t["perturbs_first"]
                     if p is not None)
        p_total = sum(len(t["perturbs_first"]) for t in detail.values())

        out[name] = {
            "n_tasks": len(detail),
            "perturb_pass_rate": p_pass / p_total if p_total > 0 else 0,
            "family_pass_rate": {
                fam: 1 - sum(fam_none[fam]) / len(fam_all[fam])
                for fam in FAMS if len(fam_all[fam]) > 0
            },
        }

    # 汇总输出
    out_path = os.path.join(R, "e0c_extraction_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {out_path}")

    # 判据
    print("\n=== E0c 判据 ===")
    worst_fam = {}
    for name, v in out.items():
        for fam, pr in v["family_pass_rate"].items():
            worst_fam[fam] = min(worst_fam.get(fam, 1.0), pr)
    print("每个 frame family 的最低通过率（跨 14 检查点）:")
    for fam in FAMS:
        wf = worst_fam.get(fam, 1.0)
        tag = "OK" if wf > 0.95 else "WARN" if wf > 0.85 else "ISSUE"
        print(f"  {fam:12} {wf:.4f}  [{tag}]")
    print("\n判定: 若所有 family 通过率 > 0.95，则提取偏差对 FS 的影响可忽略。")


if __name__ == "__main__":
    main()