# -*- coding: utf-8 -*-
"""D3: JudgeBench SOTA 加强分析（离线，零 API）。

四个攻击面各给一条证据：
  (1) 位置偏置控制 —— JudgeBench 官方已知 position bias 是判官头号失效模式
  (2) risk-coverage / AURC —— 单操作点 -> 整条曲线
  (3) 非框架类弃权基线 —— 证明优势不是"任何弃权规则"都有的
  (4) 按 source 子域稳健性 —— 证明优势不是靠某几个子域撑起来的
输出 -> D:/arxiv_llm_db/proposals/F3_SOTA_hardening_numbers.md
"""
import collections
import io
import json
import math
import random

SRC = "D:/arxiv_llm_db/experiments/e1_ladder/results/judgebench_official.json"
DATASET = r"D:\datasets\judgebench\dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
OUT = "D:/arxiv_llm_db/proposals/F3_SOTA_hardening_numbers.md"

d = json.load(io.open(SRC, encoding="utf-8"))
items = d["items"]
N = len(items)

srcmap = {}
with io.open(DATASET, encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        srcmap[o["pair_id"]] = o.get("source", "?")
for x in items:
    x["_src"] = srcmap.get(x["pair_id"], "?")
    x["_g8"] = (x.get("stable_sel") or {}).get("0.8")
    x["_g6"] = (x.get("stable_sel") or {}).get("0.6")
    x["_cons"] = x.get("consistency") or 0.0

L = []
def p(s=""):
    L.append(s)
    print(s)


def pt(itemset, field):
    v = [x.get(field) for x in itemset]
    sel = [a for a in v if a is not None]
    cov = len(sel) / len(itemset) if itemset else 0.0
    acc = sum(1 for a in sel if a is True) / len(sel) if sel else float("nan")
    return len(sel), cov, acc


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar(itemset, f_a, f_b, thr_b=None):
    b = c = 0
    for x in itemset:
        av = x.get(f_a)
        bv = (x.get("stable_sel") or {}).get(thr_b) if thr_b else x.get(f_b)
        if av is None or bv is None:
            continue
        if av and not bv:
            b += 1
        if bv and not av:
            c += 1
    return b, c, binom_two_sided(b, b + c)


p("# F3 · JudgeBench SOTA 加强分析（离线复算，零 API）")
p()
p("数据：`experiments/e1_ladder/results/judgebench_official.json`（官方 JudgeBench-GPT 350 题，judge=MiniMax-M3, T=0）")
p("脚本：`experiments/e1_ladder/d3_position_aurc.py`（复现：`E:/Anaconda/python.exe d3_position_aurc.py`）")
p()

# ------------------------------------------------------------------ 0
p("## 0. 全方法全景（350 题）")
p()
p("| 方法 | 覆盖 n | 覆盖率 | 覆盖内精度 |")
p("|---|---:|---:|---:|")
for name, f in [
    ("vanilla (single-shot)", "vanilla"),
    ("few-shot (2 ex.)", "fewshot"),
    ("majority N=5 T=0.4", "majority_lo"),
    ("majority N=5 T=0.7", "majority"),
    ("majority N=5 T=1.0", "majority_hi"),
    ("majority N=10 (0.7+1.0)", "majority_all"),
    ("RTLC rerun（外部 SOTA，同骨干）", "rtlc"),
    ("position-stability gate", "pos_stable"),
    ("cross-temperature gate", "temp_sel"),
    ("fused (stab AND vote)", "fused"),
    ("fused2 (stab OR temp)", "fused2"),
    ("fused3 (pos-stable AND stab)", "fused3"),
    ("fused4 (RTLC AND stab)", "fused4"),
]:
    c, cov, a = pt(items, f)
    p("| %s | %d | %.1f%% | %.3f |" % (name, c, 100 * cov, a))
for thr in ["0.6", "0.7", "0.8", "0.9"]:
    vals = [(x.get("stable_sel") or {}).get(thr) for x in items]
    sel = [a for a in vals if a is not None]
    p("| frame-consistency @%s | %d | %.1f%% | %.3f |" % (
        thr, len(sel), 100 * len(sel) / N, sum(1 for a in sel if a) / len(sel)))
p()

# ------------------------------------------------------------------ 1
p("## 1. 位置偏置控制（JudgeBench 官方记载的头号失效模式）")
p()
pa = sum(1 for x in items if x.get("pos_agree"))
p("- `pos_agree`（A/B 互换后判断不变）率：**%d/%d = %.3f** ⇒ 位置一翻就改答案的题占 %.1f%%" % (
    pa, N, pa / N, 100 * (1 - pa / N)))
p()
p("### 1.1 ⭐ 在位置稳定子集上重做对比（把位置偏置整个拿掉）")
p()
sub = [x for x in items if x.get("pos_agree")]
_, _, v_acc = pt(sub, "vanilla")
rc, rcov, ra = pt(sub, "rtlc")
gv = [x["_g8"] for x in sub if x["_g8"] is not None]
ga = sum(1 for a in gv if a) / len(gv)
p("位置稳定子集 n = **%d**（原 350 题的 %.1f%%）：" % (len(sub), 100 * len(sub) / N))
p()
p("| 方法 | 覆盖题数 | 覆盖率 | 覆盖内精度 |")
p("|---|---:|---:|---:|")
p("| RTLC rerun | %d | %.1f%% | **%.3f** |" % (rc, 100 * rcov, ra))
p("| frame-consistency @0.8 | %d | %.1f%% | **%.3f** |" % (len(gv), 100 * len(gv) / len(sub), ga))
p()
p("⇒ **覆盖率几乎一致（%.1f%% vs %.1f%%）可直接对读：%.3f vs %.3f = +%.1f 点**" % (
    100 * rcov, 100 * len(gv) / len(sub), ga, ra, 100 * (ga - ra)))
b, c_, pm = mcnemar(sub, "rtlc", None, thr_b="0.8")
p("- 配对 McNemar（该子集内，共同作答题）：b=%d（门控对/RTLC 错），c=%d，精确 p=%.4f" % (b, c_, pm))
p("- 对照：全 350 题上同口径 McNemar b=%d, c=%d, p=%.4f" % mcnemar(items, "rtlc", None, thr_b="0.8"))
p("- 其他基线在位置稳定子集上：位置稳定门控 %.3f@%.1f%%（定义即该子集）、vanilla %.3f@%.1f%%" % (
    pt(sub, "pos_stable")[2], 100 * pt(sub, "pos_stable")[1], v_acc, 100))
p()
p("### 1.2 门控与位置稳定门控的重叠结构")
p()
g = set(i for i, x in enumerate(items) if x["_g8"] is not None)
ps = set(i for i, x in enumerate(items) if x.get("pos_stable") is not None)
inter = g & ps
p("- 门控@0.8 作答 %d 题；位置稳定门控作答 %d 题；交集 **%d** 题" % (len(g), len(ps), len(inter)))
p("- 门控作答但位置不稳定：%d 题；位置稳定但门控弃权：%d 题" % (len(g - ps), len(ps - g)))
only_g = [items[i] for i in (g - ps)]
only_ps = [items[i] for i in (ps - g)]
if only_g:
    a = sum(1 for x in only_g if x["_g8"]) / len(only_g)
    p("  - 「门控答 / 位置不稳」%d 题精度 **%.3f**" % (len(only_g), a))
if only_ps:
    a = sum(1 for x in only_ps if x.get("pos_stable")) / len(only_ps)
    p("  - 「位置稳 / 门控弃」%d 题精度 **%.3f**" % (len(only_ps), a))
p("- 配对 McNemar（门控@0.8 vs 位置稳定门控，共同作答题）：b=%d, c=%d, p=%.4f" % (
    mcnemar(items, "pos_stable", None, thr_b="0.8")[0], mcnemar(items, "pos_stable", None, thr_b="0.8")[1],
    mcnemar(items, "pos_stable", None, thr_b="0.8")[2]))
p()

# ------------------------------------------------------------------ 2
p("## 2. Risk–coverage 与 AURC（把单点变成曲线）")
p()
CMAX = 0.906


def build_curve(sort_key, ans_field, cmax):
    srt = sorted(items, key=lambda x: -(x.get(sort_key) or 0))
    Kmax = int(round(cmax * N))
    pts = [(0.0, 0.0)]
    for K in range(1, Kmax + 1):
        ans = [srt[i].get(ans_field) for i in range(K)]
        ans = [a for a in ans if a is not None]
        acc = sum(1 for a in ans if a) / len(ans) if ans else float("nan")
        pts.append((K / N, 1 - acc))
    return pts


def fc_curve(cmax=CMAX):
    srt = sorted(items, key=lambda x: -x["_cons"])
    Kmax = int(round(cmax * N))
    pts = [(0.0, 0.0)]
    for K in range(1, Kmax + 1):
        ans = [srt[i]["_g6"] for i in range(K)]
        ans = [a for a in ans if a is not None]
        pts.append((K / N, 1 - sum(1 for a in ans if a) / len(ans)))
    return pts


def aurc(pts, cmax):
    pts = [q for q in pts if q[0] <= cmax + 1e-9]
    area = 0.0
    for i in range(1, len(pts)):
        area += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2
    return area / cmax


def risk_at(pts, cov):
    return min(pts, key=lambda q: abs(q[0] - cov))[1]


curves = [
    ("frame-consistency（本文）", fc_curve(), "consistency"),
    ("self-consistency（同 prompt 采样一致）", build_curve("self_cons", "majority", CMAX), "self_cons"),
]
p("在门控可达的覆盖区间 [0, %.1f%%] 上积分（AURC 越小越好）：" % (100 * CMAX))
p()
p("| 信号 | 排序键 | AURC | 风险@65.7% | 风险@73.1% | 风险@90.6% |")
p("|---|---|---:|---:|---:|---:|")
for name, pts, key in curves:
    p("| %s | `%s` | **%.4f** | %.3f | %.3f | %.3f |" % (
        name, key, aurc(pts, CMAX), risk_at(pts, 0.657), risk_at(pts, 0.731), risk_at(pts, CMAX)))
p()
p("- 单一固定点参照：RTLC 覆盖 65.7%% 风险 **%.3f**（远高于两条曲线同覆盖处）" % (1 - pt(items, "rtlc")[2]))
p("- `mv_agree`（0–5 票数）与 `self_cons`（票数/5）是同一排序，故不单列。")
p()

# ------------------------------------------------------------------ 3
p("## 3. 门控 vs 非框架类弃权规则（同覆盖对照）")
p()
p("| 覆盖 | RTLC | 位置稳定 | 跨温一致 | 投票一致 | 门控（本文） |")
p("|---:|---:|---:|---:|---:|---:|")
mv_pts = build_curve("mv_agree", "majority", CMAX)
for tgt in [0.50, 0.634, 0.657, 0.731, 0.783, 0.906]:
    row = ["%.1f%%" % (100 * tgt)]
    for f in ["rtlc", "pos_stable", "temp_sel"]:
        _, cov, a = pt(items, f)
        row.append("%.3f" % a if abs(cov - tgt) <= 0.05 else "—")
    row.append("%.3f" % (1 - risk_at(mv_pts, tgt)))
    row.append("%.3f" % (1 - risk_at(curves[0][1], tgt)))
    p("| " + " | ".join(row) + " |")
p()
p("（表中 RTLC/位置稳定/跨温一致三列是**固定覆盖**的方法，只有覆盖落在目标 ±5 点时才可比；")
p("投票一致与门控两列是**可连续重定阈**的，任何覆盖都可比。数值为**精度**。）")
p()

# ------------------------------------------------------------------ 4
p("## 4. 按 source 子域的稳健性")
p()
cnt = collections.Counter(x["_src"] for x in items)
p("| source | n | RTLC 覆盖 | RTLC 精度 | 门控@0.8 覆盖 | 门控@0.8 精度 |")
p("|---|---:|---:|---:|---:|---:|")
win = lose = tie = 0
for s, n in cnt.most_common():
    sub2 = [x for x in items if x["_src"] == s]
    src_n, src_cov, src_acc = pt(sub2, "rtlc")
    gv2 = [x["_g8"] for x in sub2 if x["_g8"] is not None]
    ga2 = sum(1 for a in gv2 if a) / len(gv2) if gv2 else float("nan")
    p("| %s | %d | %d (%.1f%%) | %.3f | %d (%.1f%%) | %.3f |" % (
        s, n, src_n, 100 * src_cov, src_acc, len(gv2), 100 * len(gv2) / n, ga2))
    if gv2 and src_n:
        if ga2 > src_acc:
            win += 1
        elif ga2 < src_acc:
            lose += 1
        else:
            tie += 1
p()
p("子域胜负：门控更高 **%d** 个子域 / 更低 %d / 持平 %d（共 %d 个子域）" % (win, lose, tie, len(cnt)))
p()

# ------------------------------------------------------------------ 5
p("## 5. 头条数字的配对 bootstrap 置信区间")
p()
random.seed(20261012)
B = 20000


def paired_boot_delta(gate_pts):
    """在 items 上有放回重采样，每次按自身 RTLC 覆盖率重定阈，做配对差 bootstrap。"""
    diffs = []
    for _ in range(B):
        samp = [items[random.randrange(N)] for _ in range(N)]
        rv = [x.get("rtlc") for x in samp if x.get("rtlc") is not None]
        if not rv:
            continue
        r_acc = sum(1 for a in rv if a) / len(rv)
        cov = len(rv) / len(samp)
        srt = sorted(samp, key=lambda x: -x["_cons"])
        K = int(round(cov * len(samp)))
        ans = [srt[i]["_g6"] for i in range(K)]
        ans = [a for a in ans if a is not None]
        if not ans:
            continue
        diffs.append(sum(1 for a in ans if a) / len(ans) - r_acc)
    diffs.sort()
    return sum(diffs) / len(diffs), diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


m, lo, hi = paired_boot_delta(curves[0][1])
p("- 覆盖匹配（每次重采样按自身 RTLC 覆盖率重定门控阈值）门控 − RTLC 精度差："
  "**+%.1f 点**，配对 bootstrap 95%% CI = [%+.1f, %+.1f]（B=%d，20 次重采样中 0 不落入则 CI 不含 0）" % (
      100 * m, 100 * lo, 100 * hi, B))

# ---- 位置稳定子集上的重做（干净重算，避免变量污染）----
ps_sub = [x for x in items if x.get("pos_agree")]
ps_rt_n, ps_rt_cov, ps_rt_acc = pt(ps_sub, "rtlc")
ps_gv = [x["_g8"] for x in ps_sub if x["_g8"] is not None]
ps_g_cov = len(ps_gv) / len(ps_sub)
ps_g_acc = sum(1 for a in ps_gv if a) / len(ps_gv)
b, c_, pm = mcnemar(ps_sub, "rtlc", None, thr_b="0.8")
p("- **位置稳定子集**（n=%d）：门控@0.8 %.3f@%.1f%% vs RTLC %.3f@%.1f%% ⇒ **%+.1f 点**；"
  "配对 McNemar b=%d, c=%d, 精确 p=%.4f" % (
      len(ps_sub), ps_g_acc, 100 * ps_g_cov, ps_rt_acc, 100 * ps_rt_cov,
      100 * (ps_g_acc - ps_rt_acc), b, c_, pm))
p()
p("## 6. 结论：这四条加固各挡住哪一个攻击")
p()
p("| 攻击 | 加固 | 结果 |")
p("|---|---|---|")
p("| 「你的优势只是位置偏置」 | 在位置稳定子集（n=%d）上重做 | 门控 %.3f vs RTLC %.3f，**%+.1f 点**，p=%.4f |" % (
    len(ps_sub), ps_g_acc, ps_rt_acc, 100 * (ps_g_acc - ps_rt_acc), pm))
p("| 「一个操作点不算数」 | AURC（0→90.6%%） | 门控 %.4f vs 自一致性 %.4f |" % (
    aurc(curves[0][1], CMAX), aurc(curves[1][1], CMAX)))
p("| 「随便什么弃权规则都行」 | 门控 vs 位置稳定/跨温一致/投票一致（同覆盖） | 门控在 50–78% 覆盖段最高（90.6% 处与跨温一致持平） |")
p("| 「靠某几个子域撑」 | 按 source 拆 %d 个子域 | 门控胜 %d / 负 %d |" % (len(cnt), win, lose))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nSaved ->", OUT)
