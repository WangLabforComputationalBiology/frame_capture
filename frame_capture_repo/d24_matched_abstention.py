# -*- coding: utf-8 -*-
"""F20 · 同拒答率（matched abstention）下的 SOTA 对标。

主人 2026-10-13 的硬要求：不能拿「门控 63.4% 拒答 vs RTLC 99.4% 全答」比完就撤回主张，
必须在**同样的拒答**下对标 SOTA。

做法：RTLC 修完协议后是全覆盖判官（不会拒答）。要同拒答，就把**同一个置信信号**
套在 RTLC 的判定上，让 RTLC 在与门控**完全相同的题集**上作答：

  gate@τ      = vanilla 判定，仅在 consistency >= τ 的题上作答
  rtlc+FC@τ   = RTLC(v3) 判定，仅在**同一批** consistency >= τ 的题上作答

两边拒答集逐题相同 ⇒ 覆盖率相同 ⇒ 精度可直接配对比较（McNemar）。
信号族也给 RTLC 换别的（self_cons / mv_agree / pos_agree），防止「只给 RTLC 我们的信号」的口实。
另算整条 risk-coverage 曲线 + AURC（同拒答全扫）。

输入（全部离线，零 API）
  弱骨干 results/judgebench_official.json + rtlc_weak_fair_text_raw*.json（v3 重析）
  强骨干 results/judgebench_v41.json + judgebench_v41_raw.json（v3 重析）

输出
  results/f20_matched_abstention.json
  proposals/F20_matched_abstention.md
"""
import io, json, os, sys
from math import comb

sys.path.insert(0, r"D:\arxiv_llm_db\experiments\e1_ladder")
import d15_rtlc_parser_v3 as P3

RES = r"D:\arxiv_llm_db\experiments\e1_ladder\results"
OUT_JSON = os.path.join(RES, "f20_matched_abstention.json")
OUT_MD = r"D:\arxiv_llm_db\proposals\F20_matched_abstention.md"
THRS = ["0.6", "0.7", "0.8", "0.9"]


def mc(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def pair(items, ka, kb, sel):
    """在给定题集 sel 上配对比较两个方法的判定（同拒答集）。"""
    sub = [x for x in items if x["pair_id"] in sel
           and x.get(ka) is not None and x.get(kb) is not None]
    if not sub:
        return None
    a_only = sum(1 for x in sub if x[ka] and not x[kb])
    b_only = sum(1 for x in sub if x[kb] and not x[ka])
    return dict(n=len(sub),
                acc_a=sum(1 for x in sub if x[ka]) / len(sub),
                acc_b=sum(1 for x in sub if x[kb]) / len(sub),
                delta=(a_only - b_only) / len(sub) * 0 + (sum(1 for x in sub if x[ka]) - sum(1 for x in sub if x[kb])) / len(sub),
                a_only=a_only, b_only=b_only, p=mc(a_only, b_only))


def curve(items, vkey, skey):
    """按信号降序的 risk-coverage 曲线；返回 (coverages, accs)（只在方法有判定的题上累计）。"""
    rank = sorted(items, key=lambda x: -x.get(skey, 0.0))
    have = [x for x in rank if x.get(vkey) is not None]
    cov, acc, cum_c, cum_n = [], [], 0, 0
    for k, x in enumerate(have, 1):
        cum_c += 1 if x[vkey] else 0
        cum_n += 1
        cov.append(k)
        acc.append(cum_c / cum_n)
    return cov, acc


def aurc(cov, acc, n_total, lo=None, hi=None):
    """梯形 AURC，在 [lo,hi] 覆盖区间（题数）上积分，归一到区间宽。"""
    pts = [(c, 1 - a) for c, a in zip(cov, acc)]
    lo = pts[0][0] if lo is None else lo
    hi = pts[-1][0] if hi is None else hi
    sel = [(c, r) for c, r in pts if lo <= c <= hi]
    if len(sel) < 2:
        return None
    s = 0.0
    for (c0, r0), (c1, r1) in zip(sel, sel[1:]):
        s += (r0 + r1) / 2 * (c1 - c0)
    return s / (hi - lo)


def load_weak():
    items = json.load(io.open(os.path.join(RES, "judgebench_official.json"),
                              encoding="utf-8"))["items"]
    wtxt = {}
    for fn in ("rtlc_weak_fair_text_raw.json", "rtlc_weak_fair_text_raw_s1.json",
               "rtlc_weak_fair_text_raw_s2.json", "rtlc_weak_fair_text_raw_s3.json"):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            try:
                for k, v in json.load(io.open(p, encoding="utf-8")).items():
                    if v.get("text") or k not in wtxt:
                        wtxt[k] = v
            except (OSError, ValueError):
                pass
    wtxt = {k: v for k, v in wtxt.items() if v.get("text")}
    by = {x["pair_id"]: x for x in items}
    for pid, v in wtxt.items():
        if pid not in by:
            continue
        c3, _ = P3.parse_v3(v.get("text"))
        by[pid]["rtlc_v3"] = (c3 == by[pid]["correct"]) if c3 else None
    for x in items:
        x["gate"] = x["stable_sel"]["0.8"]
        x["self_cons"] = x.get("self_cons", 0.0)
    return items


def load_strong():
    items = json.load(io.open(os.path.join(RES, "judgebench_v41.json"),
                              encoding="utf-8"))["items"]
    sraw = json.load(io.open(os.path.join(RES, "judgebench_v41_raw.json"),
                             encoding="utf-8"))["calls"]
    sb = {x["pair_id"]: x for x in items}
    for k, v in sraw.items():
        pid, kind = k.rsplit("|", 1)
        if kind == "rtlc" and pid in sb:
            c3, _ = P3.parse_v3(v.get("text"))
            sb[pid]["rtlc_v3"] = (c3 == sb[pid]["correct"]) if c3 else None
    for x in items:
        x["gate"] = x["stable_sel"]["0.8"]
    return items


D, L = {}, []
P = L.append
P("# F20 · 同拒答率下的 SOTA 对标（matched abstention）\n")
P("> 起因：主人 2026-10-13 指出——RTLC 修完协议是 99.4% 全覆盖，拿门控 63.4% 拒答")
P("> 与之相比再撤回主张，**没有回答原始问题**。本档在**完全相同的拒答集**上重比。\n")

SIGNALS = {
    "FC(consistency)": "consistency",
    "self_cons": "self_cons",
    "mv_agree": "mv_agree",
    "pos_agree": "pos_agree01",
}

for tag, items in (("weak", load_weak()), ("strong", load_strong())):
    for x in items:
        x["pos_agree01"] = 1.0 if x.get("pos_agree") else 0.0
    D[tag] = {}
    n_total = len(items)
    P("## %s（n=%d）\n" % (tag, n_total))

    # ---- 1) 逐 τ：同拒答配对比较 gate vs RTLC+FC ----
    P("### 1. 同拒答逐点：gate@τ vs RTLC+FC@τ（拒答集逐题相同）\n")
    P("| τ | 共同作答 n | 覆盖率 | gate | RTLC+FC | Δ(gate−RTLC+FC) | McNemar | p |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in THRS:
        sel = {x["pair_id"] for x in items if x["stable_sel"][t] is not None}
        r = pair(items, "gate", "rtlc_v3", sel) if False else None
        # gate 判定字段按 τ 取
        sub = [x for x in items if x["pair_id"] in sel
               and x["stable_sel"][t] is not None and x.get("rtlc_v3") is not None]
        if not sub:
            continue
        ga = [x["stable_sel"][t] for x in sub]
        ra = [x["rtlc_v3"] for x in sub]
        a_only = sum(1 for g, r_ in zip(ga, ra) if g and not r_)
        b_only = sum(1 for g, r_ in zip(ga, ra) if r_ and not g)
        acc_g = sum(ga) / len(sub)
        acc_r = sum(ra) / len(sub)
        p_ = mc(a_only, b_only)
        P("| %s | %d | %.1f%% | %.4f | %.4f | %+.4f | %d:%d | %.4f |" % (
            t, len(sub), 100 * len(sub) / n_total, acc_g, acc_r,
            acc_g - acc_r, a_only, b_only, p_))
        D[tag]["same_abstain@%s" % t] = dict(
            n=len(sub), gate=acc_g, rtlc_fc=acc_r, delta=acc_g - acc_r, p=p_)
    P("")

    # ---- 2) RTLC+FC 是否超过 RTLC 全覆盖（我们的信号给 SOTA 带来什么） ----
    P("### 2. RTLC 全覆盖 vs RTLC+FC@各 τ（我们的信号套在 SOTA 上）\n")
    P("| 系统 | 覆盖 n | 覆盖率 | 精度 |")
    P("|---|---:|---:|---:|")
    full = [x for x in items if x.get("rtlc_v3") is not None]
    acc_full = sum(1 for x in full if x["rtlc_v3"]) / len(full) if full else None
    P("| RTLC（全覆盖） | %d | %.1f%% | %.4f |" % (len(full), 100 * len(full) / n_total, acc_full))
    for t in THRS:
        sel = [x for x in items if x["stable_sel"][t] is not None
               and x.get("rtlc_v3") is not None]
        if not sel:
            continue
        acc = sum(1 for x in sel if x["rtlc_v3"]) / len(sel)
        P("| RTLC+FC@%s | %d | %.1f%% | %.4f |" % (t, len(sel), 100 * len(sel) / n_total, acc))
        D[tag]["rtlc_fc@%s" % t] = dict(n=len(sel), coverage=len(sel) / n_total, acc=acc)
    P("")

    # ---- 3) 给 RTLC 换别的信号（防「只给 RTLC 我们的信号」口实） ----
    P("### 3. RTLC+各信号 @τ=0.8 同拒答集（信号对 RTLC 的影响）\n")
    sel0 = {x["pair_id"] for x in items if x["stable_sel"]["0.8"] is not None}
    sub0 = [x for x in items if x["pair_id"] in sel0 and x.get("rtlc_v3") is not None]
    P("τ=0.8 拒答集（n=%d 共同作答）下 RTLC 精度：**%.4f**（对照组，FC 信号选出的题集）" % (
        len(sub0), sum(1 for x in sub0 if x["rtlc_v3"]) / len(sub0) if sub0 else 0))
    P("")
    P("| 信号 | RTLC 全覆盖精度 | AURC(同拒答全扫) |")
    P("|---|---:|---:|")
    for nm, sk in SIGNALS.items():
        cv, ac = curve(items, "rtlc_v3", sk)
        if len(cv) < 10:
            continue
        a_lo, a_hi = None, None
        # 统一在 FC 门控可扫的覆盖区间 [n@0.9, n@0.6] 上比，避免区间口径打架
        n9 = sum(1 for x in items if x["stable_sel"]["0.9"] is not None
                 and x.get("rtlc_v3") is not None)
        n6 = sum(1 for x in items if x["stable_sel"]["0.6"] is not None
                 and x.get("rtlc_v3") is not None)
        a = aurc(cv, ac, n_total, lo=max(1, n9), hi=n6)
        P("| RTLC+%s | %.4f | %s |" % (nm, acc_full, ("%.4f" % a) if a else "—"))
        D[tag]["aurc_rtlc+%s" % nm] = a
    P("")

    # ---- 4) 门控曲线 vs RTLC+FC 曲线（同拒答全扫 AURC） ----
    cvg, acg = curve(items, "vanilla", "consistency")
    cvr, acr = curve(items, "rtlc_v3", "consistency")
    n9 = sum(1 for x in items if x["stable_sel"]["0.9"] is not None)
    n6 = sum(1 for x in items if x["stable_sel"]["0.6"] is not None)
    a_g = aurc(cvg, acg, n_total, lo=max(1, n9), hi=n6)
    a_r = aurc(cvr, acr, n_total, lo=max(1, n9), hi=n6)
    P("### 4. 同拒答 risk-coverage（FC 排序，覆盖区间 [%d,%d] 题）\n" % (n9, n6))
    P("- gate(vanilla+FC) AURC = **%.4f**" % a_g)
    P("- RTLC+FC AURC = **%.4f**" % a_r)
    P("")
    D[tag]["aurc_gate"] = a_g
    D[tag]["aurc_rtlc_fc"] = a_r

    # ---- 5) fused：门控信号 ∧ RTLC 都答（对 RTLC 的增量） ----
    P("### 5. RTLC+FC 是否「到 SOTA」：与全覆盖 RTLC 的关系\n")
    for t in THRS:
        sel = [x for x in items if x["stable_sel"][t] is not None
               and x.get("rtlc_v3") is not None]
        if not sel:
            continue
        acc = sum(1 for x in sel if x["rtlc_v3"]) / len(sel)
        d = acc - acc_full
        P("- τ=%s：RTLC+FC **%.4f** @%.1f%% vs RTLC 全覆盖 **%.4f** @%.1f%% → **%+.4f**（拒答 %.1f%% 换 %+.1f 点）" % (
            t, acc, 100 * len(sel) / n_total, acc_full, 100 * len(full) / n_total,
            d, 100 * (1 - len(sel) / len(full)), 100 * d))
    P("")

json.dump(D, io.open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
io.open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
print("\n->", OUT_MD, "|", OUT_JSON)
