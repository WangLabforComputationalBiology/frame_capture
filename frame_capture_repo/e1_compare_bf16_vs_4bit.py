# -*- coding: utf-8 -*-
"""E1 验证: bf16 vs 4-bit 对比分析.

对每个有 bf16 结果的模型，加载对应的 4-bit (std) 结果，
计算 per-task FS 的 Spearman 相关系数和平均绝对偏差。
"""
import json
import os
import sys
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")

# 有 bf16 结果的模型
BF16_NAMES = [
    "qwen15-base", "qwen15-sft", "qwen15-dpo", "qwen15-dpo_frame", "qwen15-rlvr",
    "llama32-3b-base", "llama32-3b-instruct", "llama32-sft", "llama32-dpo",
    "internlm2-base", "internlm2-chat",
]


def jsd(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = np.maximum(0.5 * (p + q), 1e-15)
    eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def fs_from_detail(detail):
    """从 detail 字典计算 per-task FS."""
    fs = {}
    for tid, t in detail.items():
        fr = [np.array(x) for x in t["frames_first"] if x is not None]
        if len(fr) < 2:
            continue
        ref = np.mean(fr, axis=0)
        fs[tid] = float(np.mean([jsd(f, ref) for f in fr]))
    return fs


def main():
    print(f"{'Model':24} {'4-bit FS':>9} {'bf16 FS':>9} {'Δ FS':>8} {'ρ':>6} {'tasks':>6}")
    print("-" * 66)

    all_rho = []
    all_mad = []  # mean absolute deviation
    all_rmse = []

    for name in BF16_NAMES:
        # 4-bit (std)
        fp_std = os.path.join(R, f"phi_unified48_{name}.json")
        if not os.path.exists(fp_std):
            print(f"  {name:24} [skip: 4-bit not found]")
            continue
        std = json.load(open(fp_std, encoding="utf-8"))

        # bf16
        fp_bf16 = os.path.join(R, f"phi_unified48_{name}_bf16.json")
        if not os.path.exists(fp_bf16):
            print(f"  {name:24} [skip: bf16 not found]")
            continue
        bf16 = json.load(open(fp_bf16, encoding="utf-8"))

        fs_std = fs_from_detail(std["detail"])
        fs_bf16 = fs_from_detail(bf16["detail"])

        common = sorted(set(fs_std.keys()) & set(fs_bf16.keys()))
        if not common:
            continue

        v_std = np.array([fs_std[t] for t in common])
        v_bf16 = np.array([fs_bf16[t] for t in common])

        rho = np.corrcoef(v_std, v_bf16)[0, 1]
        mad = float(np.mean(np.abs(v_std - v_bf16)))
        rmse = float(np.sqrt(np.mean((v_std - v_bf16) ** 2)))
        mu_std = float(np.mean(v_std))
        mu_bf16 = float(np.mean(v_bf16))

        all_rho.append(rho)
        all_mad.append(mad)
        all_rmse.append(rmse)

        print(f"  {name:24} {mu_std:9.5f} {mu_bf16:9.5f} {mu_bf16-mu_std:+8.5f} {rho:6.4f} {len(common):6d}")

    print("-" * 66)
    print(f"  {'平均':24} {'':9} {'':9} {'':8} {np.mean(all_rho):6.4f} {'':6}")
    print(f"  {'平均 MAD':24} {'':9} {'':9} {np.mean(all_mad):8.5f} {'':6}")
    print(f"  {'平均 RMSE':24} {'':9} {'':9} {np.mean(all_rmse):8.5f} {'':6}")

    # 判据
    min_rho = min(all_rho)
    max_mad = max(all_mad)
    print(f"\n=== E1 判据 ===")
    print(f"  最低 ρ = {min_rho:.4f}  {'OK: ρ > 0.95' if min_rho > 0.95 else 'WARN: ρ <= 0.95'}")
    print(f"  最大 MAD = {max_mad:.5f}  {'OK: MAD < 0.002' if max_mad < 0.002 else 'WARN: MAD >= 0.002'}")
    verdict = "A1 量化攻击失效：bf16 与 4-bit 结果高度一致" if min_rho > 0.95 and max_mad < 0.002 else "需要进一步分析差异"
    print(f"  判定: {verdict}")


if __name__ == "__main__":
    main()