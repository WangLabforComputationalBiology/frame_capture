# -*- coding: utf-8 -*-
"""手写 DPO（Qwen2.5-1.5B LoRA，阶梯第 3 档，趋势快跑）。

避开 trl/torch 版本不兼容：
1. 用冻结的 SFT 模型预计算每对的 ref 对数概率差分 D_ref（只算一次）；
2. 加载 SFT 为 policy（LoRA 可训），手写 DPO loss 训练。
loss = -log sigma(beta * (D_theta - D_ref)), D = logp(chosen) - logp(rejected)

用法：
    D:/conda_envs/flashhead/python.exe train_dpo_manual.py
"""
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\models\qwen15-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_frame_train.jsonl")
FINAL_DIR = r"D:\models\qwen15-ladder\dpo_frame"
BETA = 0.1
LR = 3e-5
EPOCHS = 1
BATCH = 8
MAX = 200
SYS = ("You are a helpful assistant. "
       "Answer the multiple-choice question by replying with the option letter "
       "preceded by 'The answer is'.")


def logp_of(completion_logits, completion_ids):
    """completion_logits: [n, vocab]（对 completion 每个位置的 next-token 分布）；
    返回该 completion 对数概率的标量张量（保留梯度）。"""
    lp = torch.log_softmax(completion_logits.float(), dim=-1)  # [n, V]
    ar = torch.arange(completion_logits.shape[0], device=completion_logits.device)
    return lp[ar, completion_ids].sum()


def split_logits(model, tok, prompt, completion, no_grad=True):
    """返回 completion 部分每个位置的 next-token logits[*,V] 和 completion 的 target ids。"""
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(completion, add_special_tokens=False)
    full = p_ids + c_ids
    ids = torch.tensor([full], device=model.device)
    if no_grad:
        with torch.no_grad():
            out = model(ids, use_cache=False).logits
    else:
        out = model(ids, use_cache=False).logits
    # position i 预测 full[i+1]；completion 区域 indices = len(p_ids)-1 .. len(full)-2
    start = len(p_ids) - 1
    seg = out[0, start:(start + len(c_ids))]   # [len(c), V]
    return seg, torch.tensor(c_ids, device=model.device)


def compute_DP(model, tok, prompt, chosen, rejected):
    """返回 D = logp(chosen) - logp(rejected)。"""
    segC, cids = split_logits(model, tok, prompt, chosen)
    segR, rids = split_logits(model, tok, prompt, rejected)
    return logp_of(segC, cids) - logp_of(segR, rids)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    recs = recs[:MAX]
    print(f"pairs: {len(recs)}")
    if not recs:
        print("no data"); return

    # --- 第一步：预计算 ref(log)SFT 的 D_ref（冻结前向，无梯度）---
    ref = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    ref.eval()
    D_refs = []
    for i, r in enumerate(recs):
        prompt = r["prompt"]
        D_refs.append(compute_DP(ref, tok, prompt, r["chosen"], r["rejected"]).item())
        if (i + 1) % 200 == 0:
            print(f"  ref {i+1}/{len(recs)}", flush=True)
    del ref
    torch.cuda.empty_cache()
    print("ref D computed, mean=%.4f" % (sum(D_refs) / len(D_refs)))

    # --- 第二步：policy=SFT + LoRA，手写 DPO 训练 ---
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
    SYS_TEXT = SYS

    nsteps = 0
    for epoch in range(EPOCHS):
        acc = 0.0
        cnt = 0
        for i, r in enumerate(recs):
            prompt = r["prompt"]
            segC, cids = split_logits(model, tok, prompt, r["chosen"], no_grad=False)
            segR, rids = split_logits(model, tok, prompt, r["rejected"], no_grad=False)
            lpC = logp_of(segC, cids)
            lpR = logp_of(segR, rids)
            D_th = lpC - lpR
            D_ref = D_refs[i]
            loss = -torch.log(torch.sigmoid(BETA * (D_th - D_ref)))
            loss.backward()
            acc += loss.item(); cnt += 1
            if (i + 1) % BATCH == 0:
                opt.step(); opt.zero_grad()
                nsteps += 1
                if nsteps % 20 == 0:
                    print(f"  step {nsteps}  dpo_loss={acc/cnt:.4f}", flush=True)
                acc = 0.0; cnt = 0
        if cnt > 0:
            opt.step(); opt.zero_grad()

    model = model.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged DPO saved -> {FINAL_DIR}")


if __name__ == "__main__":
    main()