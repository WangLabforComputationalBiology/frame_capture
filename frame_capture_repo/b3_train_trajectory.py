# -*- coding: utf-8 -*-
"""B3: DPO 训练轨迹与 reward-frame coupling（手写 DPO + 逐步轨迹记录）.

基于 train_dpo_manual.py 的手写 DPO（LoRA on qwen15-ladder/sft，β=0.1，lr=3e-5），
增加每 EVAL_EVERY 个 optimizer step 的轨迹评测:
  1. 电池子集（12 固定任务 × neutral+14帧+2扰动，choice_probs_dual 口径）:
     FS / CS / phi / flip(vs neutral argmax) / 平均选择熵
  2. frame-conditioned implicit reward gap（无显式 RM，用 DPO 隐式 reward
     β[(logπ_θ(c)-logπ_θ(r)) - (logπ_ref(c)-logπ_ref(r))]）:
     对每个电池任务 c="The answer is A." r="The answer is B."，
     gap_signed = mean_{task,frame}[margin(frame) - margin(neutral)]，
     gap_abs = mean |·| —— reward 对 frame 的敏感度轨迹
  3. 训练批次 D_th（chosen/rejected 隐式 reward margin）逐步均值

臂（--tag）: standard(dpo_train_fast) / frame_v2(dpo_frame_v2_train, 双prompt同答案)
             / cure_v3(dpo_cure_v3_train)
产物: results/b3_traj_{tag}.json（逐步追加，原子写）；
      最终 merged 模型存 D:\\models\\qwen15-ladder\\b3_{tag}\\（新目录，不覆盖）
心跳: results/_vb_heartbeat.json
用法: D:/conda_envs/flashhead/python.exe b3_train_trajectory.py --tag standard
"""
import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from items_v3 import get_tasks48  # noqa: E402
from t3_unified_e1 import choice_probs_dual  # noqa: E402

BASE = r"D:\models\qwen15-ladder\sft"
DATA_BY_TAG = {
    "standard": os.path.join(THIS, "data", "dpo_train_fast.jsonl"),
    "frame_v2": os.path.join(THIS, "data", "dpo_frame_v2_train.jsonl"),
    "cure_v3": os.path.join(THIS, "data", "dpo_cure_v3_train.jsonl"),
}
BETA = 0.1
LR = 3e-5
BATCH = 8
MAX_PAIRS = 200
EVAL_EVERY = 5
EVAL_TASKS = 12          # 48 任务中每隔 4 个取 1 个
PROBE_N = 32             # 训练切片之外的 held-out probe（standard 数据末尾）
HEARTBEAT = os.path.join(THIS, "results", "_vb_heartbeat.json")


def logp_of(completion_logits, completion_ids):
    lp = torch.log_softmax(completion_logits.float(), dim=-1)
    ar = torch.arange(completion_logits.shape[0], device=completion_logits.device)
    return lp[ar, completion_ids].sum()


def split_logits(model, tok, prompt, completion, no_grad=True):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(completion, add_special_tokens=False)
    full = p_ids + c_ids
    ids = torch.tensor([full], device=model.device)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out = model(ids, use_cache=False).logits
    start = len(p_ids) - 1
    seg = out[0, start:(start + len(c_ids))]
    return seg, torch.tensor(c_ids, device=model.device)


def margin(model, tok, prompt, chosen, rejected):
    """implicit reward margin 的原始量 D = logp(chosen|prompt) - logp(rejected|prompt)（无梯度）。"""
    segC, cids = split_logits(model, tok, prompt, chosen)
    segR, rids = split_logits(model, tok, prompt, rejected)
    return (logp_of(segC, cids) - logp_of(segR, rids)).item()


def single_logp(model, tok, prompt, completion):
    seg, ids = split_logits(model, tok, prompt, completion)
    return logp_of(seg, ids).item()


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def heartbeat(tag, step, note, done, total):
    with open(HEARTBEAT, "w", encoding="utf-8") as fh:
        json.dump({"phase": "B3", "script": "b3_train_trajectory.py", "tag": tag,
                   "step": step, "note": note, "done": done, "total": total,
                   "ts": datetime.now().isoformat()}, fh, ensure_ascii=False)


def load_pairs(tag):
    data = DATA_BY_TAG[tag]
    recs = []
    with open(data, encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    train = recs[:MAX_PAIRS]
    probe = recs[MAX_PAIRS:MAX_PAIRS + PROBE_N] if tag == "standard" else []
    return train, probe


def pair_fields(tag, r):
    """统一返回 (promptC, promptR, chosen, rejected)。"""
    if tag == "frame_v2":
        return r["prompt_chosen"], r["prompt_rejected"], r["chosen"], r["rejected"]
    return r["prompt"], r["prompt"], r["chosen"], r["rejected"]


def eval_state(policy, ref, tok, tasks, tag):
    """返回轨迹评测 dict（电池指标 + frame reward gap + probe margin）。"""
    tids = sorted(tasks)[::4][:EVAL_TASKS]
    fs_l, cs_l, flip_l, ent_l = [], [], [], []
    gap_signed, gap_abs = [], []
    neu_am_all = []
    for tid in tids:
        task = tasks[tid]
        renders = [task["content"]] + list(task["frames"])
        ps, am = [], []
        for r in renders:
            d = choice_probs_dual(policy, tok, r + "\nAnswer:")[0]
            if d is None:
                continue
            ps.append(np.asarray(d, float))
            am.append(int(np.argmax(d)))
        if len(ps) < 6:
            continue
        ref_d = np.mean(ps, axis=0)
        fs_l.append(float(np.mean([jsd(v, ref_d) for v in ps])))
        pe = [np.asarray(choice_probs_dual(policy, tok, cp + "\nAnswer:")[0], float)
              for cp in task["content_perturb"]]
        pe = [v for v in pe if v is not None]
        if pe:
            cs_l.append(float(np.mean([jsd(v, ref_d) for v in pe])))
        neutral_am = am[0]
        neu_am_all.append(neutral_am)
        flip_l.append(float(np.mean([a != neutral_am for a in am[1:]])))
        ent_l.append(float(np.mean([-(v * np.log(np.maximum(v, 1e-12))).sum() for v in ps])))
        # frame-conditioned implicit reward gap（训练格式补全）
        c_txt, r_txt = "The answer is A.", "The answer is B."
        m_neu = margin(policy, tok, task["content"], c_txt, r_txt) \
            - margin(ref, tok, task["content"], c_txt, r_txt)
        for fr in task["frames"]:
            m_fr = margin(policy, tok, fr, c_txt, r_txt) \
                - margin(ref, tok, fr, c_txt, r_txt)
            gap_signed.append(m_fr - m_neu)
            gap_abs.append(abs(m_fr - m_neu))
    return {
        "fs": float(np.mean(fs_l)) if fs_l else None,
        "cs": float(np.mean(cs_l)) if cs_l else None,
        "flip_rate": float(np.mean(flip_l)) if flip_l else None,
        "mean_entropy": float(np.mean(ent_l)) if ent_l else None,
        "reward_gap_signed": float(np.mean(gap_signed)) if gap_signed else None,
        "reward_gap_abs": float(np.mean(gap_abs)) if gap_abs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(DATA_BY_TAG))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--save-at", type=str, default="",
                    help="逗号分隔的步数，在该步保存 LoRA adapter 到 b3_{tag}{suffix}_s{N}_adapter")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="轨迹/模型目录后缀（用于重跑变体，不覆盖原结果）")
    args = ap.parse_args()
    tag = args.tag
    suffix = args.out_suffix
    save_at = set(int(x) for x in args.save_at.split(",") if x.strip())
    torch.manual_seed(42)
    np.random.seed(42)
    out_path = os.path.join(THIS, "results", f"b3_traj_{tag}{suffix}.json")
    final_dir = os.path.join(r"D:\models\qwen15-ladder", f"b3_{tag}{suffix}")

    traj = {"tag": tag, "beta": BETA, "lr": LR, "batch": BATCH,
            "eval_every": EVAL_EVERY, "steps": [], "final_saved": False}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            traj = json.load(fh)
    if traj.get("final_saved"):
        print(f"[skip] {tag} trajectory complete", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(BASE)
    train_recs, probe = load_pairs(tag)
    n_steps_total = (len(train_recs) // BATCH) * args.epochs
    print(f"[{tag}] pairs={len(train_recs)} probe={len(probe)} total_steps={n_steps_total}", flush=True)

    tasks = get_tasks48()

    # --- ref 模型（常驻，轨迹与 D_refs 都要用）---
    ref = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    if not traj["steps"]:
        print(f"[{tag}] precomputing D_refs...", flush=True)
        D_refs = []
        for i, r in enumerate(train_recs):
            pc, pr, c, rj = pair_fields(tag, r)
            # standard: 同 prompt 两补全；frame_v2: 双 prompt 各自补全
            D_refs.append(single_logp(ref, tok, pc, c) - single_logp(ref, tok, pr, rj))
            if (i + 1) % 50 == 0:
                print(f"  ref {i+1}/{len(train_recs)}", flush=True)
        traj["d_ref_mean"] = float(np.mean(D_refs))
    else:
        D_refs = None  # 续跑时不再需要（只补评不补训练）

    # --- policy = SFT + LoRA ---
    from peft import LoraConfig, get_peft_model
    policy = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    policy.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    policy = get_peft_model(policy, lcfg)
    policy.train()
    opt = AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)

    done_units = len(traj["steps"])
    total_units = n_steps_total // EVAL_EVERY + 2

    def run_eval(step):
        ev = eval_state(policy, ref, tok, tasks, tag)
        ev["step"] = step
        traj["steps"].append(ev)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(traj, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
        heartbeat(tag, step, "eval", done_units, total_units)
        print(f"  [eval] step {step}: fs={ev['fs']:.5f} cs={ev['cs']:.4f} "
              f"flip={ev['flip_rate']:.3f} gap_abs={ev['reward_gap_abs']:.4f} "
              f"gap_signed={ev['reward_gap_signed']:.4f}", flush=True)

    # 轨迹起点（step 0）
    if not traj["steps"]:
        policy.eval()
        run_eval(0)
        policy.train()

    if D_refs is not None:
        nsteps = 0
        for epoch in range(args.epochs):
            acc, cnt = 0.0, 0
            dth_batch = []
            for i, r in enumerate(train_recs):
                pc, pr, c, rj = pair_fields(tag, r)
                segC, cids = split_logits(policy, tok, pc, c, no_grad=False)
                segR, rids = split_logits(policy, tok, pr, rj, no_grad=False)
                lpC, lpR = logp_of(segC, cids), logp_of(segR, rids)
                D_th = lpC - lpR
                loss = -torch.log(torch.sigmoid(BETA * (D_th - D_refs[i])))
                loss.backward()
                dth_batch.append(float(D_th.item()))
                acc += float(loss.item()); cnt += 1
                if (i + 1) % BATCH == 0:
                    opt.step(); opt.zero_grad()
                    nsteps += 1
                    if nsteps in save_at:
                        adir = os.path.join(r"D:\models\qwen15-ladder",
                                            f"b3_{tag}{suffix}_s{nsteps}_adapter")
                        policy.save_pretrained(adir, safe_serialization=True)
                        tok.save_pretrained(adir)
                        print(f"  adapter saved @ step {nsteps} -> {adir}", flush=True)
                    if nsteps % EVAL_EVERY == 0:
                        policy.eval()
                        run_eval(nsteps)
                        policy.train()
                    if nsteps >= n_steps_total:
                        break
            if cnt > 0:
                opt.step(); opt.zero_grad()
            if nsteps >= n_steps_total:
                break

    # --- 保存 merged 模型（新目录）---
    if not traj.get("final_saved"):
        policy = policy.merge_and_unload()
        os.makedirs(final_dir, exist_ok=True)
        policy.save_pretrained(final_dir, safe_serialization=True)
        tok.save_pretrained(final_dir)
        traj["final_saved"] = True
        traj["final_dir"] = final_dir
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(traj, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
        print(f"merged policy saved -> {final_dir}", flush=True)
    heartbeat(tag, n_steps_total, "done", total_units, total_units)
    del policy, ref
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
