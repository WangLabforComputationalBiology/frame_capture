# -*- coding: utf-8 -*-
"""make_dpo_frame_v2: 修正构造的帧重编码 DPO 数据（保留两个帧 + 同一正确答案）。

标准 DPO 需要 chosen/rejected 的 *completion 不同*（同一 prompt）。
但帧重编码的本意是"偏好对只改帧模板、两个 completion 都说同一正确答案"。
v1 的 bug：每条只存了答对帧的 prompt，chosen==rejected completion，且拒绝帧被丢弃
        → 同一 prompt 下 logp(chosen)-logp(rejected)≡0，梯度恒 0，等于没训。
v2 修正：每条存 {prompt_chosen=答对帧, prompt_rejected=答错帧, completion=正确答案 A}
        两个 prompt 不同、completion 相同(A)。
        loss(见 train_dpo_frame_v2) = 让 logp(A|答对帧) >= logp(A|答错帧)，
        配 SFT 作 ref → 训的是"帧不变性"（坏帧下也输出正确答案），梯度真实存在。

输出: data/dpo_frame_v2_train.jsonl
  {"prompt_chosen","prompt_rejected","chosen","rejected","answer"}
  chosen/rejected 字段 = completion 文本(两者相同, 均为 "The answer is A.")，
  prompt_chosen/prompt_rejected = 同一题的两个帧模板包装。
"""
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from items import get_tasks  # noqa: E402

N_ITEMS = 240
LETTERS = "ABCD"
# 用法: make_dpo_frame_v2.py <sft_judge_path> <out_jsonl> [data_jsonl]
#   sft_judge_path: 各族 SFT 判分模型 (e.g. D:\\models\\llama32-ladder\\sft)
#   out_jsonl:      输出偏好数据
#   data_jsonl:     题源 (默认 data/sft_train.jsonl, 各族同源)
SFT = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\qwen15-ladder\sft"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(THIS, "data", "dpo_frame_v2_train.jsonl")
DATA = sys.argv[3] if len(sys.argv) > 3 else os.path.join(THIS, "data", "sft_train.jsonl")


def choice_acc(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    best = None
    bestp = -1e9
    for c in LETTERS:
        t = tok.encode(" " + c, add_special_tokens=False)
        if not t:
            continue
        p0 = float(logp[t[0]])
        if p0 > bestp:
            bestp = p0
            best = c
    return best


def make_fvion(base_q, opts, fn, orig_q):
    q = orig_q
    if fn == 0:
        return f"Can you tell me: {q}"
    if fn == 1:
        return f"Which of the following is correct? {q}"
    if fn == 2:
        return f"Please answer this question: {q}"
    if fn == 3:
        return q.upper()
    if fn == 4:
        return q.lower()
    if fn == 5:
        return f"You are a knowledgeable teacher. {q}"
    if fn == 6:
        return f"You are a meticulous encyclopedia. {q}"
    if fn == 7:
        return f"You are a casual person. {q}"
    if fn == 8:
        return f"According to the authoritative reference book, {q}"
    if fn == 9:
        return f"According to an anonymous internet post, {q}"
    if fn == 10:
        return f"Most people answer this correctly. {q}"
    if fn == 11:
        return f"The panel of experts unanimously agrees on this. {q}"
    if fn == 12:
        return f"Please state your confidence first, then answer. {q}"
    return f"If you had to pick one answer, {q}"


def main():
    random.seed(13)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    random.shuffle(recs)
    recs = recs[:N_ITEMS]

    tok = AutoTokenizer.from_pretrained(SFT)
    model = AutoModelForCausalLM.from_pretrained(
        SFT, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()

    pairs = []
    n_none = 0
    for i, rec in enumerate(recs):
        msgs = rec["messages"]
        q = msgs[1]["content"]
        ans = rec["answer"]
        fm = [make_fvion(None, None, fn, q) for fn in range(14)]
        preds = {}
        for fname, fprompt in zip([f"f{j}" for j in range(14)], fm):
            preds[fname] = (choice_acc(model, tok, fprompt) == ans)
        correct = [k for k, v in preds.items() if v]
        wrong = [k for k, v in preds.items() if not v]
        if not correct or not wrong:
            n_none += 1
            continue
        ck = random.choice(correct)
        rk = random.choice(wrong)
        completion = f"The answer is {ans}."
        pairs.append({
            "prompt_chosen": fm[int(ck[1:])],   # 答对帧
            "prompt_rejected": fm[int(rk[1:])], # 答错帧
            "chosen": completion,               # = rejected (同一正确答案)
            "rejected": completion,
            "answer": ans,
        })
        if (i + 1) % 60 == 0:
            print(f"{i+1}/{N_ITEMS} items, {len(pairs)} pairs, none={n_none}", flush=True)

    print(f"total frame-recoded v2 pairs: {len(pairs)} / skipped {n_none}")
    random.shuffle(pairs)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()