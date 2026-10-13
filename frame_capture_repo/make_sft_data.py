# -*- coding: utf-8 -*-
"""从本地 MMLU auxiliary_train CSV 构造 SFT 监督数据（零下载）。

来源：D:\models\mmlu_dataset\data\auxiliary_train\{arc_easy,arc_hard,obqa,
science_elementary,science_middle}.csv
格式：[question, opt1..opt4, answer_letter]（无 header）

输出：D:\arxiv_llm_db\experiments\e1_ladder\data\sft_train.jsonl
     每行 {"messages": [system, user, assistant]}，assistant = "The answer is X."
"""
import csv
import json
import os
import random

SRC = r"D:\models\mmlu_dataset\data\auxiliary_train"
FILES = ["arc_easy.csv", "arc_hard.csv", "obqa.csv",
         "science_elementary.csv", "science_middle.csv"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_train.jsonl")
LETTERS = "ABCD"

SYSTEM = ("You are a helpful assistant. "
          "Answer the multiple-choice question by replying with the option letter "
          "preceded by 'The answer is'.")


def fmt_user(q, opts):
    lines = [q.strip()]
    for i, o in enumerate(opts):
        lines.append(f"{LETTERS[i]}. {o.strip()}")
    return "\n".join(lines)


def main():
    random.seed(0)
    rows = []
    for fn in FILES:
        p = os.path.join(SRC, fn)
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) < 6:
                    continue
                q, o1, o2, o3, o4, ans = r[0], r[1], r[2], r[3], r[4], r[5].strip().upper()
                if ans not in LETTERS:
                    continue
                opts = [o1, o2, o3, o4]
                if any(not o.strip() for o in opts):
                    continue
                rows.append((q, opts, ans))
    random.shuffle(rows)
    print(f"total usable items: {len(rows)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for q, opts, ans in rows:
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": fmt_user(q, opts)},
                    {"role": "assistant", "content": f"The answer is {ans}."},
                ],
                "answer": ans,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
