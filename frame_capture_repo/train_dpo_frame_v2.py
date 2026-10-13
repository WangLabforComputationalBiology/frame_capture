# -*- coding: utf-8 -*-
"""train_dpo_frame_v2: 修正构造的帧重编码 DPO 训练（Qwen2.5-1.5B）。

与 train_dpo_manual 的区别（这是修复的根本）：
  v1 用同一 prompt 算 logp(chosen)-logp(rejected)，completion 相同 → D≡0，训练空转。
  v2 用 *不同 prompt*（答对帧 vs 答错帧），*同一 completion*（正确答案 A）：
      Δ = logp_chosen(prompt_chosen) - logp_chosen(prompt_rejected)   ← 模型
      Δ_ref = logp_ref (prompt_chosen) - logp_ref (prompt_rejected)   ← 冻结 SFT
      loss = -log σ( β * (Δ - Δ_ref) )
  意义：推动正确帧下正确答案概率 >= 错误帧下 >-> 帧不变性(坏帧下也输出正确答案)。
  β, LoRA, 数据超参对齐 v1/ladder，保证与 generic DPO(qwen15-dpo) 仅在"信号构造"上不同。

用法:
    D:/conda_envs/flashhead/python.exe train_dpo_frame_v2.py
"""
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# 用法: train_dpo_frame_v2.py <sft_base_path> <data_jsonl> <out_dir>
#   sft_base_path: 起点 SFT (ref 与 policy 同源)
#   data_jsonl:    make_dpo_frame_v2 产出的偏好数据
#   out_dir:       DPO 权重复合保存目录
BASE = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\qwen15-ladder\sft"
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_frame_v2_train.jsonl")
FINAL_DIR = sys.argv[3] if len(sys.argv) > 3 else r"D:\models\qwen15-ladder\dpo_frame_v2"
BETA = 0.1
LR = 3e-5
EPOCHS = 1
BATCH = 4
MAX = 200


def logp_of(completion_logits, completion_ids):
    lp = torch.log_softmax(completion_logits.float(), dim=-1)
    ar = torch.arange(completion_logits.shape[0], device=completion_logits.device)
    return lp[ar, completion_ids].sum()


def split_logits(model, tok, prompt, completion, no_grad=True):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(completion, add_special_tokens=False)
    full = p_ids + c_ids
    ids = torch.tensor([full], device=model.device)
    out = model(ids, use_cache=False).logits
    start = len(p_ids) - 1
    seg = out[0, start:(start + len(c_ids))]
    return seg, torch.tensor(c_ids, device=model.device)


def model_logp_at_frame(model, tok, prompt, completion):
    seg, cids = split_logits(model, tok, prompt, completion, no_grad=False)
    return logp_of(seg, cids)


def ref_logp_at_frame(model, tok, prompt, completion):
    seg, cids = split_logits(model, tok, prompt, completion, no_grad=True)
    return logp_of(seg, cids)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    recs = recs[:MAX]
    print(f"pairs: {len(recs)}")
    if not recs:
        print("no data")
        return

    # --- ref: 冻结 SFT 预计算 Δ_ref = logp(答对帧) - logp(答错帧)，同一正确答案 ---
    ref = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    ref.eval()
    D_refs = []
    for i, r in enumerate(recs):
        comp = r["chosen"]
        d = ref_logp_at_frame(ref, tok, r["prompt_chosen"], comp) \
            - ref_logp_at_frame(ref, tok, r["prompt_rejected"], comp)
        D_refs.append(d.item())
        if (i + 1) % 200 == 0:
            print(f"  ref {i+1}/{len(recs)}", flush=True)
    del ref
    torch.cuda.empty_cache()
    mean_dr = sum(D_refs) / len(D_refs)
    print(f"ref delta computed, mean(Δ_ref)={mean_dr:.4f}"
          f"  (>0 即 SFT 在答对帧下正确答案概率本就更高的量级)", flush=True)

    # --- policy: SFT + LoRA ---
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
            comp = r["chosen"]
            lpC = model_logp_at_frame(model, tok, r["prompt_chosen"], comp)
            lpR = model_logp_at_frame(model, tok, r["prompt_rejected"], comp)
            D_th = lpC - lpR
            D_ref = D_refs[i]
            loss = -torch.log(torch.sigmoid(BETA * (D_th - D_ref)))
            loss.backward()
            acc += loss.item()
            cnt += 1
            if cnt % BATCH == 0:
                opt.step()
                opt.zero_grad()
                nsteps += 1
                if nsteps % 25 == 0:
                    print(f"  step {nsteps}  dpo_loss={acc/cnt:.4f}", flush=True)
                acc = 0.0
                cnt = 0
        if cnt > 0:
            opt.step()
            opt.zero_grad()

    model = model.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged dpo_frame_v2 saved -> {FINAL_DIR}")


if __name__ == "__main__":
    main()