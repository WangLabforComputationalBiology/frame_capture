# -*- coding: utf-8 -*-
"""E1: φ 阶梯——同一模型族沿 base→SFT→DPO→RLVR 的框架俘获率。

方法（对应框架全稿 §6.1 logit 版 + §6.2 表面幅度匹配）：
- 4-bit 加载（bitsandbytes，RTX 3080 10.7GB 可容 8B）
- 每档每任务：对每个框架变体取答案选项上的 softmax 边际 → π(·|x,m)
- FS = 帧平均 JSD(π(·|τx) ‖ π̄(·|x))，CS = 内容扰动版同法
- φ = FS/(FS+CS)
- 输出阶梯图（φ vs 对齐阶段）+ JSON

用法：
    PYTHONNOUSERSITE=1 D:/conda_envs/flashhead/python.exe run_e1.py [--dry-run]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import jensenshannon

MODEL_DIRS = {
    "base": r"D:\models\tulu3-8b\base",
    "sft": r"D:\models\tulu3-8b\sft",
    "dpo": r"D:\models\tulu3-8b\dpo",
    "rl": r"D:\models\tulu3-8b\rl",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ANSWER_TOKENS = {"A": "A", "B": "B", "C": "C"}  # 答案 token（大写，大小写变体经 upper() 归一）

# 选项 token 可能带标点（"A." / "A："），取首个非空白字符归一
def norm_ans(s):
    s = (s or "").strip().upper()
    if not s:
        return None
    if s[0] in "ABC":
        return s[0]
    return None


def load_model(model_path, dry_run=False):
    """4-bit 加载模型+分词器（transformers 4.48+ / 5.x 均兼容的接口）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dry_run:
        print(f"[dry-run] would load {model_path}")
        return None, None

    tok = AutoTokenizer.from_pretrained(model_path)
    # 4-bit 配置
    import torch
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def get_choice_logits(model, tok, prompt, choice_tokens=("A", "B", "C")):
    """对 prompt 取各选项的 softmax 概率（覆盖所有常见 token 形式）。

    关键：对齐模型会输出 "A." / " A" / "ĠA" / "A:" 等不同形式，
    单 token 匹配会失真。这里对每个选项取一组候选 token 的概率和。
    """
    import torch

    def candidate_ids(c):
        """选项 c 的所有常见 token 形式。"""
        cands = set()
        for s in (c, f" {c}", f"\n{c}", f"{c}.", f" {c}.", f"{c}:", f" {c}:", f"{c})", f" {c})"):
            ids = tok.encode(s, add_special_tokens=False)
            if ids:
                cands.add(ids[-1])  # 每个形式的最后一个 token
        # 直接字符串 token（如 "ĠA" 已被 f" {c}" 覆盖）
        cands.add(tok.convert_tokens_to_ids(c))
        return [x for x in cands if x is not None]

    choice_ids = {c: candidate_ids(c) for c in choice_tokens}
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to(model.device) for k, v in enc.items()})
    logits = out.logits[0, -1, :]  # 最后位置
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    dist = {}
    for c in choice_tokens:
        dist[c] = float(sum(probs[i] for i in choice_ids[c]))
    # 归一化（选项概率和）
    tot = sum(dist.values())
    if tot > 0:
        dist = {k: v / tot for k, v in dist.items()}
    return dist


def jsd(p, q):
    """Jensen-Shannon 散度（在共享键集上）。"""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    # 平滑防除零
    a = a + 1e-9
    b = b + 1e-9
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    kl1 = np.sum(a * np.log(a / m))
    kl2 = np.sum(b * np.log(b / m))
    return float(0.5 * (kl1 + kl2))


def compute_phi(model, tok, task, dry_run=False):
    """对单个任务计算 FS/CS/φ + 逐框架分布（供 N7）。"""
    content = task["content"]
    answer = task["answer"]
    frames = task["frames"]
    perturbs = task["content_perturb"]

    # 参考行为：帧平均（用原始 content 作为参考，等价于 E_m π(·|x,m) 近似）
    ref = get_choice_logits(model, tok, content) if not dry_run else {"A": 0.5, "B": 0.25, "C": 0.25}

    # FS：对每个框架变体（记录每框架 dist 供 N7）
    fs_list = []
    frame_dists = []
    for f in frames:
        p = get_choice_logits(model, tok, f) if not dry_run else ref
        frame_dists.append({"prompt": f, "dist": p})
        fs_list.append(jsd(p, ref))
    FS = float(np.mean(fs_list))

    # CS：内容扰动（表面幅度匹配：编辑距离相近档）
    cs_list = []
    perturb_dists = []
    for c in perturbs:
        p = get_choice_logits(model, tok, c) if not dry_run else ref
        perturb_dists.append({"prompt": c, "dist": p})
        cs_list.append(jsd(p, ref))
    CS = float(np.mean(cs_list))

    phi = FS / (FS + CS) if (FS + CS) > 1e-9 else 0.5
    return {"FS": FS, "CS": CS, "phi": phi, "n_frames": len(frames), "n_perturbs": len(perturbs),
            "frame_dists": frame_dists, "perturb_dists": perturb_dists, "answer": answer}


def _save_incremental(results, args):
    """每档完成立即保存当前结果到 JSON（崩溃保护）。"""
    import json
    out = os.path.join(OUT_DIR, "phi_ladder_partial.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只跑流程")
    ap.add_argument("--stages", nargs="+", default=list(MODEL_DIRS.keys()),
                    help="要跑的阶段（默认全部 base sft dpo rl）")
    ap.add_argument("--dump-frames", action="store_true",
                    help="同时导出框架级数据（供 N7 样本外预测）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from items import get_tasks

    tasks = get_tasks()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    frame_data = {}  # N7 用：{stage: {task: {frames, heldout_frames, answer}}}
    for stage in args.stages:
        if stage not in MODEL_DIRS:
            print(f"[skip] unknown stage {stage}")
            continue
        print(f"\n=== {stage} ===")
        t0 = time.time()
        model, tok = load_model(MODEL_DIRS[stage], dry_run=args.dry_run)
        stage_res = {}
        stage_frames = {}
        for tid, task in tasks.items():
            r = compute_phi(model, tok, task, dry_run=args.dry_run)
            stage_res[tid] = r
            print(f"  {tid}: φ={r['phi']:.3f}  FS={r['FS']:.4f}  CS={r['CS']:.4f}")
            # N7 数据：框架 → train/heldout 切分（留 4 个框架做 held-out）
            if args.dump_frames and not args.dry_run:
                all_f = r["frame_dists"]
                held = all_f[-4:]
                train = all_f[:-4]
                stage_frames[tid] = {
                    "frames": train,
                    "heldout_frames": held,
                    "answer": r["answer"],
                    "content": task["content"],
                }
        results[stage] = stage_res
        if args.dump_frames and not args.dry_run:
            frame_data[stage] = stage_frames
            # 框架级数据即时保存（每档完成后 N7 可单独跑）
            fo = os.path.join(OUT_DIR, f"frame_level_{stage}.json")
            with open(fo, "w", encoding="utf-8") as f:
                json.dump(stage_frames, f, ensure_ascii=False, indent=2)
            print(f"  [框架级数据 -> {fo}]")
        print(f"  [{time.time()-t0:.0f}s]")
        # 增量保存（每档完成立即写盘，崩溃不丢数据）
        _save_incremental(results, args)
        if not args.dry_run:
            import gc
            del model
            gc.collect()
            if hasattr(__import__('torch'), 'cuda'):
                import torch
                torch.cuda.empty_cache()
            print(f"  [显存已释放]")

    # 汇总
    summary = {}
    for stage, st in results.items():
        phis = [st[t]["phi"] for t in tasks]
        summary[stage] = {
            "phi_mean": float(np.mean(phis)),
            "phi_per_task": {t: st[t]["phi"] for t in tasks},
            "fs_mean": float(np.mean([st[t]["FS"] for t in tasks])),
            "cs_mean": float(np.mean([st[t]["CS"] for t in tasks])),
        }
        print(f"\n{stage}: mean φ = {summary[stage]['phi_mean']:.3f}")

    out = os.path.join(OUT_DIR, "phi_ladder.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")

    # 画图（无 matplotlib 则跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        stages = list(summary.keys())
        phis = [summary[s]["phi_mean"] for s in stages]
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(stages)), phis, "o-", linewidth=2.5, markersize=10)
        plt.xticks(range(len(stages)), stages)
        plt.ylim(0, 1)
        plt.ylabel("Frame Capture φ (mean over tasks)")
        plt.xlabel("Alignment stage")
        plt.title("E1: φ along base → SFT → DPO → RLVR (Tulu-3-8B)")
        plt.grid(alpha=0.3)
        fig = os.path.join(OUT_DIR, "phi_ladder.png")
        plt.savefig(fig, dpi=150, bbox_inches="tight")
        print(f"Figure -> {fig}")
    except ImportError as e:
        print(f"[warn] matplotlib not available: {e}")


if __name__ == "__main__":
    main()
