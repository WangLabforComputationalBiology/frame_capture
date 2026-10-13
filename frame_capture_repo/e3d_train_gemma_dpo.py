# -*- coding: utf-8 -*-
"""E3d: Gemma-3-4B LoRA DPO（第三家族梯子第 2 档）.

基于 SFT 输出，用 dpo_train_fast.jsonl 做 DPO 训练.
RTX 3080 10GB: 4B bf16 + LoRA + grad_ckpt 可容.

输出: D:\models\gemma-ladder\dpo

用法:
    D:/conda_envs/flashhead/python.exe e3d_train_gemma_dpo.py
"""
import json, os, sys, time, random
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")
import torch, numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_DIR = r"D:\models\gemma-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train_fast.jsonl")
OUT_DIR = r"D:\models\gemma-ladder\dpo"

BETA = 0.1
LR = 3e-5
EPOCHS = 1
BATCH = 4
MAX_ITEMS = 500


def logp_of(completion_logits, completion_ids):
    lp = torch.log_softmax(completion_logits.float(), dim=-1)
    ar = torch.arange(completion_logits.shape[0], device=completion_logits.device)
    return lp[ar, completion_ids].sum()


def compute_DP(model, tok, prompt, chosen, rejected):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(chosen, add_special_tokens=False)
    r_ids = tok.encode(rejected, add_special_tokens=False)
    full_c = torch.tensor([p_ids + c_ids], device=model.device)
    full_r = torch.tensor([p_ids + r_ids], device=model.device)
    with torch.no_grad():
        out_c = model(full_c, use_cache=False).logits[0]
        out_r = model(full_r, use_cache=False).logits[0]
    start = len(p_ids) - 1
    segC = out_c[start:start + len(c_ids)]
    segR = out_r[start:start + len(r_ids)]
    cid = torch.tensor(c_ids, device=model.device)
    rid = torch.tensor(r_ids, device=model.device)
    return logp_of(segC, cid) - logp_of(segR, rid)


def main():
    if os.path.exists(os.path.join(OUT_DIR, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(OUT_DIR, "model.safetensors")):
        print(f"[skip] {OUT_DIR} already exists", flush=True)
        return

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== Gemma-3-4B DPO ===", flush=True)
    t_start = time.time()

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(SFT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}\n\n{{ message['content'] }}{% elif message['role'] == 'assistant' %}\n\n{{ message['content'] }}{% endif %}{% endfor %}\n\n"

    # Load data
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    random.shuffle(recs)
    recs = recs[:MAX_ITEMS]
    print(f"pairs: {len(recs)}", flush=True)
    if not recs:
        return

    # Step 1: ref log probs (use SFT model full precision)
    print("loading ref model...", flush=True)
    t0 = time.time()
    ref = AutoModelForCausalLM.from_pretrained(
        SFT_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    ref.eval()
    print(f"ref loaded in {time.time()-t0:.1f}s", flush=True)

    D_refs = []
    for i, r in enumerate(recs):
        D_refs.append(compute_DP(ref, tok, r["prompt"], r["chosen"], r["rejected"]).item())
        if (i + 1) % 200 == 0:
            print(f"  ref {i+1}/{len(recs)}", flush=True)
    del ref
    torch.cuda.empty_cache()
    print(f"ref D computed, mean={sum(D_refs)/len(D_refs):.4f}", flush=True)

    # Step 2: policy = SFT + LoRA
    from peft import LoraConfig, get_peft_model
    print("loading policy model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SFT_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.enable_input_require_grads()

    # Gemma3 has language_model submodule; LoRA needs to go on the LM part
    if hasattr(model, "language_model"):
        lm = model.language_model
        print("applying LoRA to language_model submodule", flush=True)
    else:
        lm = model

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.train()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}", flush=True)
    print(f"policy loaded in {time.time()-t0:.1f}s", flush=True)

    print("training start...", flush=True)
    t0 = time.time()
    nsteps = 0
    for epoch in range(EPOCHS):
        order = list(range(len(recs)))
        random.shuffle(order)
        acc = 0.0
        cnt = 0
        for idx in order:
            r = recs[idx]
            p_ids = tok.encode(r["prompt"], add_special_tokens=False)
            c_ids = tok.encode(r["chosen"], add_special_tokens=False)
            r_ids = tok.encode(r["rejected"], add_special_tokens=False)
            full_c = torch.tensor([p_ids + c_ids], device=model.device)
            full_r = torch.tensor([p_ids + r_ids], device=model.device)
            tf = time.time()
            out_c = model(full_c, use_cache=False).logits[0]
            out_r = model(full_r, use_cache=False).logits[0]
            start = len(p_ids) - 1
            lp_c = torch.log_softmax(out_c[start:start + len(c_ids)].float(), dim=-1)
            lp_r = torch.log_softmax(out_r[start:start + len(r_ids)].float(), dim=-1)
            cid = torch.tensor(c_ids, device=model.device)
            rid = torch.tensor(r_ids, device=model.device)
            D_th = lp_c[torch.arange(len(c_ids), device=model.device), cid].sum() - \
                   lp_r[torch.arange(len(r_ids), device=model.device), rid].sum()
            loss = -torch.log(torch.sigmoid(BETA * (D_th - D_refs[idx])))
            loss.backward()
            fwd = time.time() - tf
            acc += loss.item()
            cnt += 1
            if cnt % BATCH == 0:
                opt.step()
                opt.zero_grad()
                nsteps += 1
                if nsteps % 25 == 0:
                    print(f"  step {nsteps}  loss={acc/cnt:.4f}  fwd_avg={fwd:.2f}s  elapsed={time.time()-t0:.1f}s", flush=True)
                acc = 0.0
                cnt = 0
        if cnt > 0:
            opt.step()
            opt.zero_grad()

    print(f"training done in {time.time()-t0:.1f}s", flush=True)

    # Save merged model
    print("saving...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    print(f"saved -> {OUT_DIR}  total={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()