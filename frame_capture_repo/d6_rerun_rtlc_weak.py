# -*- coding: utf-8 -*-
"""重跑弱骨干（MiniMax-M3）的 RTLC，用足够大的 token 预算。

起因（2026-10-12 发现）：`judgebench_official.py` 里 RTLC 走 `MAXTOK_LONG`，**默认只有 60**。
而 RTLC 的 prompt 要求"先自己作答，再判谁更好"——60 token 写不完自己的答案，
于是 34.3% 的调用解析不出，被记成"不作答/判错"。这不是 RTLC 的性质，是**预算人为压制**。

本脚本用 maxtok=16000 重跑 350 题的 RTLC，其余条件（prompt、温度 0、无 system、同骨干）保持一致，
得到公平的弱骨干 RTLC 覆盖与精度。可断点续跑。

输出：
  results/rtlc_weak_fair_raw.json
  proposals/F8_rtlc_budget_fairness.md
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r"D:\arxiv_llm_db\experiments\behavioral")
sys.path.insert(0, r"D:\arxiv_llm_db\experiments\e1_ladder")
from client import chat            # noqa: E402
import judgebench_v41 as J         # noqa: E402

DATA = J.DATA
RES = J.OUTDIR
CACHE = os.path.join(RES, "rtlc_weak_fair_raw.json")
OUT = r"D:\arxiv_llm_db\proposals\F8_rtlc_budget_fairness.md"
MAXTOK = 16000
OLD_MAXTOK = 60
NTHR = 6


def load(p):
    if os.path.exists(p):
        try:
            return json.load(io.open(p, encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save(p, obj):
    tmp = "%s.%d.tmp" % (p, os.getpid())
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    for _ in range(5):
        try:
            os.replace(tmp, p)
            return
        except OSError:
            time.sleep(0.3)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    try:
        os.remove(tmp)
    except OSError:
        pass


rows = [json.loads(l) for l in io.open(DATA, encoding="utf-8")]
gt = {r["pair_id"]: ("A" if r["label"] == "A>B" else "B") for r in rows}
cache = load(CACHE)
todo = [r for r in rows if r["pair_id"] not in cache]
print("items=%d  cached=%d  todo=%d  maxtok=%d" % (
    len(rows), len(cache), len(todo), MAXTOK), flush=True)


def run(r):
    prompt = J.RTLC_TMPL.format(question=r["question"], answer_a=r["response_A"],
                                answer_b=r["response_B"])
    t0 = time.time()
    try:
        text, meta = chat(prompt, max_tokens=MAXTOK, temperature=0.0, backend="minimax")
    except Exception as e:                       # noqa: BLE001
        return r["pair_id"], {"err": repr(e)[:200]}
    ch, rule = J.parse_choice_r(text)
    return r["pair_id"], {"choice": ch, "rule": rule, "lat": round(time.time() - t0, 1),
                          "stop": meta.get("stop_reason"),
                          "text_len": len(text or ""),
                          "out_tok": (meta.get("usage") or {}).get("output_tokens")}


done = 0
with ThreadPoolExecutor(max_workers=NTHR) as ex:
    futs = [ex.submit(run, r) for r in todo]
    try:
        for f in as_completed(futs):
            k, rec = f.result()
            cache[k] = rec
            done += 1
            if done % 10 == 0 or done == len(todo):
                print("  %d/%d" % (done, len(todo)), flush=True)
                save(CACHE, cache)
    except BaseException:
        for x in futs:
            x.cancel()
        save(CACHE, cache)
        raise
save(CACHE, cache)

# ---- 汇总：新旧对照 ----
mx = {x["pair_id"]: x for x in
      json.load(io.open(os.path.join(RES, "judgebench_official.json"), encoding="utf-8"))["items"]}
new = cache
ok = [v for v in new.values() if v.get("choice")]
acc = sum(1 for pid, v in new.items() if v.get("choice") == gt[pid]) / len(ok)
old_cov = sum(1 for x in mx.values() if x.get("rtlc") is not None) / len(mx)
old_acc = sum(1 for x in mx.values() if x.get("rtlc") is True) / sum(
    1 for x in mx.values() if x.get("rtlc") is not None)
trunc = sum(1 for v in new.values() if v.get("stop") == "max_tokens")
toks = [v.get("out_tok") or 0 for v in new.values()]

# 门控在"门控作答 ∩ 新 RTLC 作答"上的对照
both = [x for x in mx.values()
        if x.get("stable_sel", {}).get("0.8") is not None
        and new.get(x["pair_id"], {}).get("choice") is not None]
ga = sum(1 for x in both if x["stable_sel"]["0.8"]) / len(both)
ra = sum(1 for x in both
         if new[x["pair_id"]]["choice"] == gt[x["pair_id"]]) / len(both)


def bt(b, c):
    import math
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


b = sum(1 for x in both if x["stable_sel"]["0.8"]
        and new[x["pair_id"]]["choice"] != gt[x["pair_id"]])
c = sum(1 for x in both if not x["stable_sel"]["0.8"]
        and new[x["pair_id"]]["choice"] == gt[x["pair_id"]])

L = []
P = L.append
P("# F8 · 弱骨干 RTLC 的 token 预算公平性修正（2026-10-12）\n")
P("## 问题\n")
P("`judgebench_official.py` 给 RTLC 用的是 `MAXTOK_LONG`，**默认 %d**（`WB_MAXTOK_LONG` 未设置）。" % OLD_MAXTOK)
P("而 RTLC 的 prompt 明确要求「First, generate your own answer to the instruction.」——"
  "60 个输出 token 连它自己的答案都写不完，于是大量调用在给出「Output (a)/(b)」之前就停了。\n")
P("⇒ 弱骨干上 RTLC 的 **34.3% 解析失败是预算人为造成的**，不是 RTLC 的性质；"
  "它 0.657 的精度也只是「能在 60 token 内说完的那些题」上的精度。"
  "原论文把它写成「coverage is set by output parseability」——字面没错，但**归因错了**。\n")
P("## 修正\n")
P("同骨干（MiniMax-M3）、同 prompt、同温度 0、无 system，只把 token 预算提到 **%d** 重跑 350 题。\n" % MAXTOK)
P("| | 原（maxtok=%d） | 修正后（maxtok=%d） |" % (OLD_MAXTOK, MAXTOK))
P("|---|---|---|")
P("| 覆盖率 | %.1f%% (%d/350) | **%.1f%% (%d/350)** |" % (
    100 * old_cov, round(old_cov * 350), 100 * len(ok) / len(rows), len(ok)))
P("| 精度 | %.4f | **%.4f** |" % (old_acc, acc))
P("| 截断(max_tokens) | — | %d (%.1f%%) |" % (trunc, 100 * trunc / len(rows)))
P("| 平均输出 token | — | %.0f |" % (sum(toks) / len(toks)))
P("")
P("## 对门控 vs RTLC 的影响（τ=0.8 门控 ∩ RTLC 作答，弱骨干）\n")
P("- 共同作答 %d 题：门控 **%.4f** vs RTLC **%.4f**（差 **%+.1f 点**）" % (
    len(both), ga, ra, 100 * (ga - ra)))
P("- McNemar 门控独对 %d / RTLC 独对 %d，精确 p = **%.4f**" % (b, c, bt(b, c)))
P("")
P("## 结论\n")
if acc > old_acc + 0.01:
    P("- 修正后 RTLC 精度**上升**，说明原稿**低估**了 RTLC —— 必须改。")
else:
    P("- 修正后 RTLC 精度变化不大，原稿结论方向不变。")
P("- 论文写法建议：**把弱骨干那列的 RTLC 数字换成修正值**，并在 tablenotes 说明"
  "原 60-token 预算是遗留默认值、已修正；同时保留「两列 token 预算一致」这一声明。")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written ->", OUT)
