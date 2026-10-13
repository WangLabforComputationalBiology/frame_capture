# -*- coding: utf-8 -*-
"""Llama-3.2-3B QLoRA DPO (W-D 复制梯子第 2 档).

从 D:\models\llama32-ladder\sft 起始, 与 qwen15 DPO 同数据 (dpo_train_fast.jsonl
前 800 对) 同超参 (LoRA r16, beta 0.1, lr 5e-5). 输出适配器, 由 merge_llama32.py
在 CPU 端合并.
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
                          BitsAndBytesConfig, TrainingArguments)
from trl import DPOTrainer
from peft import LoraConfig

BASE = r"D:\models\llama32-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train_fast.jsonl")
ADAPTER_DIR = r"D:\models\llama32-ladder\dpo_adapter_final"


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = ("{% for message in messages %}"
                         "{% if message['role']=='system' %}"
                         "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% elif message['role']=='user' %}"
                         "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% elif message['role']=='assistant' %}"
                         "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                         "{% endif %}{% endfor %}"
                         "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")

    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    recs = recs[:800]
    print(f"dpo pairs: {len(recs)}")
    ds = Dataset.from_list(recs)

    def map_messages(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [{"role": "assistant", "content": ex["chosen"]}],
            "rejected": [{"role": "assistant", "content": ex["rejected"]}],
        }
    ds = ds.map(map_messages)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=False,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    args = TrainingArguments(
        output_dir=ADAPTER_DIR + "_ckpt",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = DPOTrainer(
        model=BASE,
        ref_model=None,
        beta=0.1,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
        peft_config=lcfg,
        quantization_config=bnb,
    )
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
    print(f"adapter saved -> {ADAPTER_DIR}")
    print("NOTE: 合并到 bf16 请运行 merge_llama32.py dpo (CPU 端)")


if __name__ == "__main__":
    main()
