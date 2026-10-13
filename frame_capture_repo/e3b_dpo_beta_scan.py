# -*- coding: utf-8 -*-
"""E3b: DPO 剂量扫描 —— β ∈ {0.05, 0.3}（默认 0.1 已有 E3a 数据）.

Qwen1.5B SFT 上训练 DPO 不同 β 值.
输出: D:\models\qwen15-ladder\dpo_beta{0.05,0.3}

用法:
    D:/conda_envs/flashhead/python.exe e3b_dpo_beta_scan.py qwen15 --beta 0.05
    D:/conda_envs/flashhead/python.exe e3b_dpo_beta_scan.py qwen15 --beta 0.3
"""
import json, os, sys, time, random
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")
import torch, numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED_CONFIGS = {
    "qwen15": {
        "base": r"D:\models\qwen15-ladder\sft",
        "data": os.path.join(os.path.dirname(__file__), "data", "dpo_train_fast.jsonl"),
        "out_root": r"D:\models\qwen15-ladder",
        "max_items": 500,
    },
}

LR = 3e-5
EPOCHS = 1
BATCH = 4


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


def train_beta(model_name, beta, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = SEED_CONFIGS[model_name]
    beta_str = f"{beta:.2f}".replace(".", "_")
    out_dir = os.path.join(cfg["out_root"], f"dpo_beta{beta_str}")
    if os.path.exists(os.path.join(out_dir, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[skip] {out_dir} already exists", flush=True)
        return

    print(f"\n=== {model_name} beta={beta} ===", flush=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    recs = []
    with open(cfg["data"], encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    random.shuffle(recs)
    recs = recs[:cfg["max_items"]]
    print(f"pairs: {len(recs)}", flush=True)
    if not recs:
        return

    # Step 1: ref log probs
    print("loading ref model...", flush=True)
    ref = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    ref.eval()
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
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.train()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

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
            out_c = model(full_c, use_cache=False).logits[0]
            out_r = model(full_r, use_cache=False).logits[0]
            start = len(p_ids) - 1
            lp_c = torch.log_softmax(out_c[start:start + len(c_ids)].float(), dim=-1)
            lp_r = torch.log_softmax(out_r[start:start + len(r_ids)].float(), dim=-1)
            cid = torch.tensor(c_ids, device=model.device)
            rid = torch.tensor(r_ids, device=model.device)
            D_th = lp_c[torch.arange(len(c_ids), device=model.device), cid].sum() - \
                   lp_r[torch.arange(len(r_ids), device=model.device), rid].sum()
            loss = -torch.log(torch.sigmoid(beta * (D_th - D_refs[idx])))
            loss.backward()
            acc += loss.item()
            cnt += 1
            if cnt % BATCH == 0:
                opt.step()
                opt.zero_grad()
                nsteps += 1
                if nsteps % 25 == 0:
                    print(f"  step {nsteps}  loss={acc/cnt:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)
                acc = 0.0
                cnt = 0
        if cnt > 0:
            opt.step()
            opt.zero_grad()

    print(f"training done in {time.time()-t0:.1f}s", flush=True)

    # Save merged model
    print("saving...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"saved -> {out_dir}  total={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen15"])
    ap.add_argument("--beta", type=float, required=True)
    args = ap.parse_args()
    train_beta(args.model, args.beta)