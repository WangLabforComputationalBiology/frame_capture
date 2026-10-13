# -*- coding: utf-8 -*-
"""Qwen2.5-1.5B LoRA DPO（阶梯第 3 档）。

从 D:\models\qwen15-ladder\sft 起始，在自举偏好对上做 DPO。
输出合并权重到 D:\models\qwen15-ladder\dpo。

用法：
    D:/conda_envs/flashhead/python.exe train_dpo.py
"""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

BASE = r"D:\models\qwen15-ladder\sft"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dpo_train_fast.jsonl")
OUT_ROOT = r"D:\models\qwen15-ladder"
ADAPTER_DIR = os.path.join(OUT_ROOT, "dpo_adapter")
FINAL_DIR = os.path.join(OUT_ROOT, "dpo")


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.chat_template:
        tok.chat_template = None  # 用原始拼接

    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    recs = recs[:800]  # 快跑：子集看趋势
    print(f"dpo pairs: {len(recs)}")

    ds = Dataset.from_list(recs)

    # prompt/chosen/rejected 已是明文，转为 messages 格式
    def map_messages(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [{"role": "assistant", "content": ex["chosen"]}],
            "rejected": [{"role": "assistant", "content": ex["rejected"]}],
        }
    ds = ds.map(map_messages)

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=BASE,
        ref_model=None,          # 用当前 base 作为参考（DPOTrainer 从 model 冻结 ref）
        beta=0.1,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
        peft_config=lcfg,
    )
    trainer.train()

    # 合并权重
    model = trainer.model.merge_and_unload()
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR, safe_serialization=True)
    tok.save_pretrained(FINAL_DIR)
    print(f"merged DPO model saved -> {FINAL_DIR}")


if __name__ == "__main__":
    main()