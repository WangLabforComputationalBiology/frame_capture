# -*- coding: utf-8 -*-
"""Qwen2.5-1.5B LoRA SFT（阶梯第 2 档）。

RTX 3080 10.7GB：1.5B bf16 权重(3GB) + LoRA 梯度 + 短序列(≤512)可容。
输出合并后的完整权重到 D:\models\qwen15-ladder\sft。

用法：
    D:/conda_envs/flashhead/python.exe train_sft.py
"""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

BASE = r"D:\models\Qwen2.5-1.5B"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_train.jsonl")
OUT_ROOT = r"D:\models\qwen15-ladder"
ADAPTER_DIR = os.path.join(OUT_ROOT, "sft_adapter")
FINAL_DIR = os.path.join(OUT_ROOT, "sft")


def load_data():
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    ds = Dataset.from_list(recs)
    return ds.select(range(min(9000, len(recs)))), recs


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds, recs = load_data()
    ds = ds.select(range(min(2400, len(ds))))   # 快跑：小子集看趋势
    print(f"train examples: {len(ds)}")

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
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()

    model = model.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged model saved -> {FINAL_DIR}")


if __name__ == "__main__":
    main()
