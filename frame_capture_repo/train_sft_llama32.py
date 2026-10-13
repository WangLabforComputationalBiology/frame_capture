# -*- coding: utf-8 -*-
"""Llama-3.2-3B QLoRA SFT (W-D 复制梯子第 1 档).

与 qwen15 梯子同数据同超参 (sft_train.jsonl 前 2400 条, LoRA r16, lr 1e-4),
仅家族不同 → 隔离家族变量. 3B bf16 (6.4GB) 超 60% 显存上限, 故用
QLoRA (4-bit 底座) 训练, 适配器在 CPU 端合并回 bf16.

输出: D:\models\llama32-ladder\sft
"""
import json
import os
import sys

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, PeftModel

BASE = r"D:\models\Llama-3.2-3B-base"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_train.jsonl")
OUT_ROOT = r"D:\models\llama32-ladder"
ADAPTER_DIR = os.path.join(OUT_ROOT, "sft_adapter")
FINAL_DIR = os.path.join(OUT_ROOT, "sft")


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Llama-3.2 base 无 chat_template; 用 Llama-3 官方模板保持梯子两档 (SFT/DPO) 提示一致
    tok.chat_template = ("{% for message in messages %}"
                         "{% if message['role']=='system' %}"
                         "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% elif message['role']=='user' %}"
                         "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% elif message['role']=='assistant' %}"
                         "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% endif %}{% endfor %}"
                         "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")

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
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    ds = Dataset.from_list(recs)
    ds = ds.select(range(min(2400, len(ds))))
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
        save_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()

    os.makedirs(ADAPTER_DIR + "_final", exist_ok=True)
    model.save_pretrained(ADAPTER_DIR + "_final", safe_serialization=True)
    print(f"adapter saved -> {ADAPTER_DIR}_final")
    print("NOTE: 合并到 bf16 请运行 merge_llama32.py (CPU 端, 不占显存)")


if __name__ == "__main__":
    main()
