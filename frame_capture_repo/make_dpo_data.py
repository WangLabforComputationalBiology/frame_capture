# -*- coding: utf-8 -*-
"""自举 DPO 偏好对（零外部数据）。

用 SFT 检查点对本地题库采样，正确答案为 chosen、错误为 rejected。
对：同一 prompt 下模型自己的输出（一个是正确格式+对答案，一个错答案）。

输出：data/dpo_train.jsonl {"prompt", "chosen", "rejected"}
"""
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT = r"D:\models\qwen15-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_train.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train.jsonl")
LETTERS = "ABCD"
N_SAMPLES = 4
MAX_PROMPTS = 3000


def main():
    random.seed(1)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    random.shuffle(recs)
    recs = recs[:MAX_PROMPTS]

    tok = AutoTokenizer.from_pretrained(SFT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SFT, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    pairs = []
    with torch.no_grad():
        for i, rec in enumerate(recs):
            msgs = rec["messages"][:2]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            outs = model.generate(
                **enc, do_sample=True, temperature=0.9, top_p=0.95,
                max_new_tokens=16, num_return_sequences=N_SAMPLES)
            gens = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    for o in outs]
            ans = rec["answer"]
            correct = [g for g in gens if re.match(rf"^[^A-D]*{ans}\b", g)
                       or f"is {ans}" in g]
            wrong = [g for g in gens if g not in correct]
            if correct and wrong:
                pairs.append({
                    "prompt": prompt,
                    "chosen": random.choice(correct),
                    "rejected": random.choice(wrong),
                })
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(recs)} prompts, {len(pairs)} pairs", flush=True)

    print(f"total pairs: {len(pairs)}")
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
