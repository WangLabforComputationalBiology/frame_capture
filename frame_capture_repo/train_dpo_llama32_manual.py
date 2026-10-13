# -*- coding: utf-8 -*-
"""Llama-3.2-3B 手写 DPO (W-D 复制梯子第 2 档) — 不依赖 trl.

复刻 qwen15 DPO 超参: LoRA r16 alpha32, beta 0.1, lr 5e-5, cosine, 1 epoch, 800 对.

ref logprob 用 disable_adapter() 取 backbone (== SFT 合并权重 == LoRA 初始态),
与冻结 ref_model 等价, 且只需 1 个 4bit 模型.

输出: D:\models\llama32-ladder\dpo_adapter_final ; 之后 merge_llama32.py dpo
"""
import json
import os
import sys

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

BASE = r"D:\models\llama32-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train_fast.jsonl")
ADAPTER_DIR = r"D:\models\llama32-ladder\dpo_adapter_final"
BETA = 0.1
LR = 5e-5
MAX_PAIRS = 800
BATCH = 8
GRAD_ACCUM = 1
WARMUP = 0.05

TEMPLATE = ("{% for message in messages %}"
            "{% if message['role']=='system' %}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role']=='user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role']=='assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")


def encode_resp(tok, user_text, resp_text):
    """返回 (full_ids, resp_mask) — resp_mask=True 标记 response token 位置(含 eos)."""
    user = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=True, add_generation_prompt=True)
    resp = tok(resp_text + tok.eos_token, add_special_tokens=False)["input_ids"]
    full = user + resp
    mask = [False] * len(user) + [True] * len(resp)
    return full, mask


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = TEMPLATE

    pair_l = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            pair_l.append(json.loads(line))
    pair_l = pair_l[:MAX_PAIRS]
    print(f"dpo pairs: {len(pair_l)}")

    sample_ids, sample_masks = [], []
    for r in pair_l:
        ci, cm = encode_resp(tok, r["prompt"], r["chosen"])
        qi, qm = encode_resp(tok, r["prompt"], r["rejected"])
        sample_ids.append((ci, qi))
        sample_masks.append((cm, qm))

    def pad_col(seqs, value, L):
        out = torch.full((len(seqs), L), value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def make_batch(lo, hi):
        # chosen / rejected 各自 pad, 再统一 pad 到 maxlen
        cids = [sample_ids[i][0] for i in range(lo, hi)]
        qids = [sample_ids[i][1] for i in range(lo, hi)]
        cms = [sample_masks[i][0] for i in range(lo, hi)]
        qms = [sample_masks[i][1] for i in range(lo, hi)]
        L = max([len(x) for x in cids + qids])
        ci = pad_col(cids, tok.pad_token_id, L)
        qi = pad_col(qids, tok.pad_token_id, L)
        cm = _bool_matrix(cms, L)
        qm = _bool_matrix(qms, L)
        cam = (ci != tok.pad_token_id).long()
        qam = (qi != tok.pad_token_id).long()
        return (ci.cuda(), cam, cm.cuda()), (qi.cuda(), qam, qm.cuda())

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=False,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="cuda",
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.train()

    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad], lr=LR)
    nbatch = (len(pair_l) + BATCH - 1) // BATCH
    steps = nbatch // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(WARMUP * steps), num_training_steps=steps)

    def logp(batch, disable):
        ids, am, mask = batch
        if disable:
            with model.disable_adapter():
                with torch.no_grad():
                    out = model(ids, attention_mask=am).logits
        else:
            out = model(ids, attention_mask=am).logits
        logp = F.log_softmax(out[:, :-1], dim=-1)       # [B,L-1,V]
        nxt = ids[:, 1:]                                 # [B,L-1]
        per = logp.gather(2, nxt.unsqueeze(-1)).squeeze(-1)  # [B,L-1]
        rmask = mask[:, 1:].float()                       # response 位置(右移对齐)
        return (per * rmask).sum(1) / rmask.sum(1).clamp(min=1)

    opt.zero_grad()
    for it in range(0, nbatch):
        bch, rch = make_batch(it * BATCH, min((it + 1) * BATCH, len(pair_l)))
        plogp_c = logp(bch, disable=False)
        plogp_r = logp(rch, disable=False)
        with torch.no_grad():
            rlogp_c = logp(bch, disable=True)
            rlogp_r = logp(rch, disable=True)
        loss = -F.logsigmoid(BETA * ((plogp_c - rlogp_c) - (plogp_r - rlogp_r))).mean()
        loss.backward()
        if (it + 1) % GRAD_ACCUM == 0 or (it + 1) == nbatch:
            opt.step(); sched.step(); opt.zero_grad()
        if it % 20 == 0:
            print(f"  {it}/{nbatch} loss={loss.item():.4f}", flush=True)
        del loss

    model = model.merge_and_unload() if False else model
    model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
    print(f"adapter saved -> {ADAPTER_DIR}  (运行 merge_llama32.py dpo)")


def _bool_matrix(masks, L):
    out = torch.zeros(len(masks), L, dtype=torch.bool)
    for i, m in enumerate(masks):
        for j, v in enumerate(m):
            out[i, j] = v
    return out


if __name__ == "__main__":
    main()