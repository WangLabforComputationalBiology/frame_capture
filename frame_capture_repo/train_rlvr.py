# -*- coding: utf-8 -*-
"""RLVR（可验证奖励强化）——阶梯第 4 档，快速趋势版。

手写 REINFORCE（Leave-One-Out baseline）避开 trl 兼容问题。
奖励 = 可验证正确性（题库标准答案）：答对 +1、答错 -1。
起点：D:\models\qwen15-ladder\dpo_frame。
"""
import json
import os
import random
import re

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

START = r"D:\models\qwen15-ladder\dpo_frame"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_train.jsonl")
FINAL_DIR = r"D:\models\qwen15-ladder\rlvr"
LR = 2e-5
EPOCHS = 2
MAX = 100
MAX_NEW = 6


def parse_letter(text):
    m = re.search(r"[A-D]", text)
    return m.group(0) if m else None


def main():
    tok = AutoTokenizer.from_pretrained(START)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    random.seed(3)
    random.shuffle(recs)
    recs = recs[:MAX]
    print(f"rlvr items: {len(recs)}")

    from peft import LoraConfig, get_peft_model
    policy = AutoModelForCausalLM.from_pretrained(
        START, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    policy.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    policy = get_peft_model(policy, lcfg)
    opt = AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)

    def gen(prompt, grad=False):
        """生成 completion，返回 (text, cum_logprob_tensor or float)。"""
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)],
                           device=policy.device)
        input_ids = ids
        total_lp = torch.zeros(1, device=policy.device, requires_grad=grad)
        gen_tokens = []
        for _ in range(MAX_NEW):
            if grad:
                out = policy(input_ids).logits[:, -1, :]
            else:
                with torch.no_grad():
                    out = policy(input_ids).logits[:, -1, :]
            dist = torch.distributions.Categorical(logits=out.float())
            tid = dist.sample()
            lp = dist.log_prob(tid)
            if grad:
                total_lp = total_lp + lp
            input_ids = torch.cat([input_ids, tid.unsqueeze(-1)], dim=-1)
            gen_tokens.append(tid.item())
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        if not grad:
            total_lp = total_lp.item()
        return text, total_lp

    for epoch in range(EPOCHS):
        # 遍A：no_grad 采样，记 scores（算 LOO baseline）
        scores = []
        for rec in recs:
            prompt = rec["messages"][1]["content"]
            ans = rec["answer"]
            text, _ = gen(prompt, grad=False)
            pred = parse_letter(text)
            scores.append(1.0 if pred == ans else -1.0)
        mean_s = sum(scores) / len(scores)
        print(f"epoch {epoch+1}: baseline mean score = {mean_s:.3f}", flush=True)

        # 遍B：带梯度的 REINFORCE 更新，adv = s - baseline
        opt.zero_grad()
        n_samp = 0
        for i, rec in enumerate(recs):
            prompt = rec["messages"][1]["content"]
            ans = rec["answer"]
            text, lp = gen(prompt, grad=True)
            pred = parse_letter(text)
            score = 1.0 if pred == ans else -1.0
            ## Leave-one-out baseline 用均值，adv = score - mean
            adv = score - mean_s
            loss = -adv * lp
            loss.backward()
            n_samp += 1
        opt.step()
        print(f"  RLVR update done ({n_samp} samples)", flush=True)

    policy = policy.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    policy.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged RLVR saved -> {FINAL_DIR}")


if __name__ == "__main__":
    main()