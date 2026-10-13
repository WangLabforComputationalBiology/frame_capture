# -*- coding: utf-8 -*-
"""train_dpo_cure_v3: 治愈 v3 训练（S1 测量点对齐 + S3 帧扩容早停硬闸）。

数据 (dpo_cure_v3_train.jsonl):
  prompt   = frame+题+"\nAnswer:"      # 与测量一致
  chosen   = " A"                       # 众数字母(单token, 无外生格式)
  rejected = " X"                       # 该坏帧实际错答单token
起点 = qwen15-dpo;  Δ_ref 由冻结 DPO 同测量点预计算; KL 锚 DPO。

早停硬闸: 每 EVAL_EVERY 步用留出 dev_meta(30帧字母) 算 dev FS,
   dev FS <= 地板带上界(dev_base_floor) 即停;  低于地板下界 -> 过压/背帧。
用法:
  D:/conda_envs/flashhead/python.exe train_dpo_cure_v3.py
"""
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\qwen15-ladder\dpo"
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(THIS, "data", "dpo_cure_v3_train.jsonl")
DEV_META = sys.argv[3] if len(sys.argv) > 3 else os.path.join(THIS, "data", "dpo_cure_v3_dev.jsonl")
FINAL_DIR = sys.argv[4] if len(sys.argv) > 4 else r"D:\models\qwen15-ladder\dpo_cure_v3"
FRAMES_JSON = sys.argv[5] if len(sys.argv) > 5 else os.path.join(THIS, "data", "cure_v3_frames.json")
BETA = 0.1
LR = 3e-5
EPOCHS = 2
BATCH = 4
EVAL_EVERY = 40
FLOOR_UPPER = 0.010   # 地板带上界(约等于 base FS), 由 Step0 的帧集/dev 标定
FLOOR_LOWER = 0.005


def first_letter_lp(model, tok, prompt, no_grad=True):
    """测量点: 返回下一 token 对各字母 {c: logp} (tensor 或 float)."""
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(model.device) for k, v in ids.items()}
    if no_grad:
        with torch.no_grad():
            out = model(**ids)
    else:
        out = model(**ids)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    outd = {}
    for c in "ABCD":
        t = tok.encode(" " + c, add_special_tokens=False)
        if t:
            v = lp[t[0]]
            outd[c] = v.item() if no_grad else v  # train 保留 tensor
    return outd


def eval_dev_current(model, tok, dev, frames):
    """用当前模型重算 dev 逐题 30 帧字母, 返回帧间一致率(众数占比, 越高越稳)."""
    import numpy as np
    agrees = []
    for rec in dev:
        q = rec.get("question")
        letters = []
        for fn in frames:
            if not q:
                continue
            prompt = fn(q) + "\nAnswer:"
            d = first_letter_lp(model, tok, prompt, no_grad=True)
            if not d:
                continue
            letters.append(max(d, key=d.get))
        valid = [l for l in letters if l in "ABCD"]
        if len(valid) < 10:
            continue
        from collections import Counter
        mode, _ = Counter(valid).most_common(1)[0]
        agrees.append(float(sum(1 for l in valid if l == mode)) / len(valid))
    return float(np.mean(agrees)) if agrees else None


def delta_lp(tok, d, letter_expr):
    """从 {c: logp} 取出某字母单token 的 logp. letter_expr 如 ' A'|'D'."""
    c = letter_expr.strip()
    return d.get(c)


def policy_delta(model, tok, prompt, chosen, rejected, no_grad=True):
    """单token: Δ = logp(chosen) - logp(rejected). no_grad=True 返 float, 否则返 tensor."""
    d = first_letter_lp(model, tok, prompt, no_grad=no_grad)
    lc = delta_lp(tok, d, chosen)
    lr = delta_lp(tok, d, rejected)
    if lc is None or lr is None:
        return None
    diff = lc - lr
    # no_grad: lc/lr 已是 float; 训练: 是 tensor, 保留图
    return diff if no_grad else diff


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    dev = []
    if os.path.exists(DEV_META):
        with open(DEV_META, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dev.append(json.loads(line))
    print(f"train pairs: {len(recs)}; dev items: {len(dev)}")
    if not recs:
        print("no data")
        return

    # ref: 冻结 DPO 同测量点预计算 Δ_ref
    ref = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    ref.eval()
    D_refs = []
    for i, r in enumerate(recs):
        D_refs.append(policy_delta(ref, tok, r["prompt"], r["chosen"], r["rejected"]))
        if (i + 1) % 200 == 0:
            print(f"  ref {i+1}/{len(recs)}", flush=True)
    valid_ref = [d for d in D_refs if d is not None]
    if not valid_ref:
        print("ERROR: no valid ref deltas (chosen/rejected token not found)")
        return
    print(f"ref Δ mean={sum(valid_ref)/len(valid_ref):.4f}", flush=True)
    del ref
    torch.cuda.empty_cache()

    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.train()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    nsteps = 0
    for epoch in range(EPOCHS):
        acc = 0.0
        cnt = 0
        for i, r in enumerate(recs):
            lpC = policy_delta(model, tok, r["prompt"], r["chosen"], r["rejected"],
                               no_grad=False)
            if lpC is None:
                continue
            D_th = lpC
            loss = -torch.log(torch.sigmoid(BETA * (D_th - D_refs[i])))
            loss.backward()
            acc += loss.item()
            cnt += 1
            if cnt % BATCH == 0:
                opt.step()
                opt.zero_grad()
                nsteps += 1
                if nsteps % 25 == 0:
                    print(f"  step {nsteps}  dpo_loss={acc/cnt:.4f}", flush=True)
                if nsteps % EVAL_EVERY == 0:
                    # 早停: 当前模型 dev 帧一致率(越高越稳) + 地板对照
                    agree = eval_dev_current(model, tok, dev, FRAMES) if dev else None
                    print(f"    [eval step {nsteps}] dev agree={agree}", flush=True)
                    if agree is not None and agree >= 0.93:
                        print(f"    [EARLY STOP] dev agree={agree:.3f} >=0.93 (≈地板)", flush=True)
                        opt.zero_grad()
                        model = model.merge_and_unload()
                        os.makedirs(FINAL_DIR, exist_ok=True)
                        model.save_pretrained(FINAL_DIR, safe_serialization=True)
                        tok.save_pretrained(FINAL_DIR)
                        print(f"merged dpo_cure_v3 (early-stop) saved -> {FINAL_DIR}")
                        return
        if cnt > 0:
            opt.step()
            opt.zero_grad()

    model = model.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged dpo_cure_v3 saved -> {FINAL_DIR}")


# 30 帧模板 (与 make_dpo_cure_v3 一致)
FRAMES = [
    lambda q: f"Can you tell me: {q}",
    lambda q: f"Which of the following is correct? {q}",
    lambda q: f"Please answer this question: {q}",
    lambda q: q.upper(),
    lambda q: q.lower(),
    lambda q: f"You are a knowledgeable teacher. {q}",
    lambda q: f"You are a meticulous encyclopedia. {q}",
    lambda q: f"You are a casual person. {q}",
    lambda q: f"According to the authoritative reference book, {q}",
    lambda q: f"According to a famous Nobel laureate, {q}",
    lambda q: f"Most people answer this correctly. {q}",
    lambda q: f"The panel of experts unanimously agrees on this. {q}",
    lambda q: f"Please state your confidence first, then answer. {q}",
    lambda q: f"If you had to pick one answer, {q}",
    lambda q: f"Think carefully and answer: {q}",
    lambda q: f"Now respond to the following: {q}",
    lambda q: f"What is your answer to this? {q}",
    lambda q: q.title(),
    lambda q: f"You are a patient tutor of elementary students. {q}",
    lambda q: f"You are a strict university examiner. {q}",
    lambda q: f"You are an empathetic friend. {q}",
    lambda q: f"According to an encyclopedic reference, {q}",
    lambda q: f"According to a leading medical board, {q}",
    lambda q: f"The entire class agrees on this. {q}",
    lambda q: f"A group of peers suggests this answer. {q}",
    lambda q: f"Give a single best answer to: {q}",
    lambda q: f"You must answer definitively. {q}",
    lambda q: f"[Question] {q}",
    lambda q: f"Question: {q}",
    lambda q: f"Q: {q}",
]


if __name__ == "__main__":
    main()