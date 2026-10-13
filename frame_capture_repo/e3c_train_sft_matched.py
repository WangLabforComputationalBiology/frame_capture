# -*- coding: utf-8 -*-
"""E3c: 用 DPO chosen 侧重训 SFT（消除 SFT/DPO 数据混淆）.

从 dpo_train_fast.jsonl 提取 {prompt, chosen} 对作为 SFT 监督数据,
在 Qwen1.5B / Llama-3.2-3B base 上 LoRA SFT.

输出: D:\models\qwen15-ladder\sft_matched
      D:\models\llama32-ladder\sft_matched
"""
import json, os, sys, time
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL_CONFIGS = {
    "qwen15": {
        "base": r"D:\models\Qwen2.5-1.5B",
        "out": r"D:\models\qwen15-ladder\sft_matched",
        "adapter": r"D:\models\qwen15-ladder\sft_matched_adapter",
    },
    "llama32": {
        "base": r"D:\models\Llama-3.2-3B-base",
        "out": r"D:\models\llama32-ladder\sft_matched",
        "adapter": r"D:\models\llama32-ladder\sft_matched_adapter",
    },
}

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train_fast.jsonl")
SYSTEM = ("You are a helpful assistant. "
          "Answer the multiple-choice question by replying with the option letter "
          "preceded by 'The answer is'.")


def prepare_data():
    """将 DPO JSONL 转为 SFT messages 格式."""
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            recs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": d["prompt"]},
                    {"role": "assistant", "content": d["chosen"]},
                ]
            })
    print(f"prepared {len(recs)} SFT examples from DPO data", flush=True)
    return recs


def train(model_name, max_items=2000):
    cfg = MODEL_CONFIGS[model_name]
    out_dir = cfg["out"]
    adapter_dir = cfg["adapter"]

    if os.path.exists(os.path.join(out_dir, "model.safetensors.index.json")) or \
       os.path.exists(os.path.join(out_dir, "model.safetensors")):
        print(f"[skip] {out_dir} already exists", flush=True)
        return

    print(f"\n=== {model_name} SFT matched ===", flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}\n\n{{ message['content'] }}{% elif message['role'] == 'assistant' %}\n\n{{ message['content'] }}{% endif %}{% endfor %}\n\n"

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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}", flush=True)

    recs = prepare_data()
    ds = Dataset.from_list(recs)
    ds = ds.select(range(min(max_items, len(ds))))
    print(f"train examples: {len(ds)}", flush=True)

    def collate(batch):
        input_ids_list, labels_list = [], []
        for rec in batch:
            prompt = tok.apply_chat_template(
                rec["messages"][:2], tokenize=False, add_generation_prompt=True)
            full = prompt + rec["messages"][2]["content"] + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            input_ids_list.append(f_ids)
            labels_list.append(labels)
        maxlen = max(len(x) for x in input_ids_list)
        pad_id = tok.pad_token_id
        attn_list = []
        for i in range(len(input_ids_list)):
            n = len(input_ids_list[i])
            pad = maxlen - n
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad
            labels_list[i] = labels_list[i] + [-100] * pad
            attn_list.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids_list),
            "labels": torch.tensor(labels_list),
            "attention_mask": torch.tensor(attn_list),
        }

    args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_drop_last=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    print(f"training done in {time.time()-t0:.1f}s", flush=True)

    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"saved -> {out_dir}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen15", "llama32"])
    ap.add_argument("--max-items", type=int, default=2000)
    args = ap.parse_args()
    train(args.model, args.max_items)