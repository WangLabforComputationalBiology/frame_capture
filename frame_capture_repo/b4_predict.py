# -*- coding: utf-8 -*-
"""B4: 未见数据 flip 预测 —— item-(task,frame) 级，四重 hold-out（离线 + B1 特征）.

预测目标（方案文档 B4 节）: Y_flip=1 若保答案 frame 改变所选答案（argmax != 任务众数）。

特征（全部可从已有数据恢复，无新推理）:
  frame_jsd     该帧分布对帧混合均值的 JSD
  task_fs       任务级 FS
  fam_fs        家族内局部 FS（该任务该家族帧的平均 JSD）
  conf_kappa    该帧分布的 max prob
  entropy       该帧分布的熵
  content_margin 任务级众数分布的前两概率差
  cap_proxy     neutral 渲染 argmax 是否等于 canonical 答案（capability 代理）
  subspace_proj B1 LOTO 共享子空间投影占比（仅 4 个 ladder 模型有 B1 数据）

hold-out 四重:
  H1 tasks     按 task id 哈希 32 训练 / 16 测试
  H2 templates 按 frame family 训练 {rephrase,case,persona,verbal} / 测试 {authority,peer}
  H3 family    训练 qwen15 系 checkpoint / 测试 llama32 系（及反向）
  H4 domain    训练 48 任务电池 / 测试 b0_heldout 24 新任务电池（subspace 特征不可用）

模型（numpy 逻辑回归，标准化 + L2）:
  capability-only / confidence-only / entropy-only / FS-only / fam-local-only /
  subspace-only / combined / combined+subspace
指标: AUROC / AUPRC / Brier / ECE(10 bins) / top-decile 富集
判据: combined(+subspace) 在各 hold-out 上 ≥ 三基线，且 subspace 加入有独立增益。
输出: results/b4_prediction.json
用法: D:/conda_envs/flashhead/python.exe b4_predict.py
"""
import glob
import json
import os
import sys

for p in list(sys.path):
    if "\\Roaming\\Python\\Python3" in p:
        sys.path.remove(p)
sys.path.insert(0, r"D:\conda_envs\flashhead\Lib\site-packages")

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")
OUT = os.path.join(R, "b4_prediction.json")

FAM_OF_STD14 = {i: f for i, f in enumerate(
    ["rephrase"] * 3 + ["case"] * 2 + ["persona"] * 3 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAM_OF_NEW12 = {i: f for i, f in enumerate(
    ["rephrase"] * 2 + ["case"] * 2 + ["persona"] * 2 +
    ["authority"] * 2 + ["peer"] * 2 + ["verbal"] * 2)}
FAMS = ["rephrase", "case", "persona", "authority", "peer", "verbal"]
BATTERY_CKPTS = ["qwen15-base", "qwen15-sft", "qwen15-dpo",
                 "llama32-3b-base", "llama32-sft", "llama32-dpo"]
LADDER_WITH_B1 = ["qwen15-sft", "qwen15-dpo", "llama32-sft", "llama32-dpo"]
HELDOUT_CKPTS = ["qwen15-sft", "qwen15-dpo", "llama32-sft", "llama32-dpo"]


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = np.maximum(0.5 * (p + q), 1e-15); eps = 1e-15
    klp = np.sum(np.where(p > eps, p * np.log2(p / np.maximum(m, eps)), 0.0))
    klq = np.sum(np.where(q > eps, q * np.log2(q / np.maximum(m, eps)), 0.0))
    return 0.5 * (klp + klq)


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 0.5
        i = j + 1
    return ranks


def auc_mw(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels) > 0
    n1 = int(labels.sum()); n0 = int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels) > 0
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(~y)
    prec = tp / np.maximum(tp + fp, 1)
    n1 = labels.sum()
    if n1 == 0:
        return float("nan")
    # AP = sum over true positives of precision at that rank
    return float((prec * y).sum() / n1)


def brier(scores, labels):
    return float(np.mean((np.asarray(scores, float) - np.asarray(labels, float)) ** 2))


def ece(scores, labels, bins=10):
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (scores > edges[i]) & (scores <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(e)


def topdec_enrichment(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    k = max(1, len(scores) // 10)
    idx = np.argsort(-scores)[:k]
    base = labels.mean()
    return float(labels[idx].mean() / base) if base > 0 else float("nan")


class Logit:
    """numpy 逻辑回归（全批梯度下降 + L2，特征外部标准化）。"""

    def __init__(self, lr=0.1, epochs=3000, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(axis=0); self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.w = np.zeros(d); self.b = 0.0
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            gw = Z.T @ (p - y) / n + self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * gw; self.b -= self.lr * gb
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))


FEATS_ALL = ["frame_jsd", "task_fs", "fam_fs", "conf_kappa", "entropy",
             "content_margin", "cap_proxy"]
FEATS_SUB = ["subspace_proj"]


def build_items_battery():
    """6 个 checkpoint 的 48 电池 item 表（含 cap_proxy/content_margin 重算）。"""
    import b0_reanalysis as B0
    items = []
    for name in BATTERY_CKPTS:
        path = os.path.join(R, f"phi_unified48_{name}_bf16.json")
        ck = B0.parse_checkpoint(path)
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)["detail"]
        for tid, t in ck.items():
            for i in range(14):
                if i >= len(t["frame_jsd"]):
                    continue
                vec = raw[tid]["frames_first"][i]
                p = np.array(vec, float) if vec is not None else None
                kappa = float(p.max()) if p is not None else np.nan
                ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum()) if p is not None else np.nan
                fam = FAM_OF_STD14[i]
                items.append({
                    "ckpt": name, "task": tid, "frame": i, "family": fam,
                    "flip": t["frame_flip"][i] if i < len(t["frame_flip"]) else 0,
                    "frame_jsd": t["frame_jsd"][i],
                    "task_fs": t["fs_jsd"], "fam_fs": t["fam_fs"].get(fam, np.nan),
                    "conf_kappa": kappa, "entropy": ent,
                    "content_margin": np.nan, "cap_proxy": np.nan,
                    "subspace_proj": np.nan,
                })
        # content_margin 用 ref 分布前两差；cap_proxy = ref argmax 是否为 canonical 答案 A
        by_task = {}
        for it in items:
            if it["ckpt"] == name:
                by_task.setdefault(it["task"], []).append(it)
        for tid, group in by_task.items():
            vecs = [raw[tid]["frames_first"][i] for i in range(14)
                    if raw[tid]["frames_first"][i] is not None]
            if not vecs:
                for it in group:
                    it["content_margin"] = np.nan
                    it["cap_proxy"] = 0.0
                continue
            ref = np.mean([np.array(v, float) for v in vecs], axis=0)
            srt = np.sort(ref)[::-1]
            margin = float(srt[0] - srt[1])
            am = int(np.argmax(ref))
            for it in group:
                it["content_margin"] = margin
                it["cap_proxy"] = float(am == 0)  # canonical 答案 A
    return items


def build_subspace_feature():
    """LOTO 共享子空间投影: {ckpt: {(task,frame): ratio}}（final-layer 方向）。"""
    feats = {}
    for name in LADDER_WITH_B1:
        root = os.path.join(R, f"b1_hidden_{name}")
        if not os.path.isdir(root):
            continue
        stacks = {}
        for fp in sorted(os.listdir(root)):
            if fp.endswith(".npy"):
                stacks[fp[:-4]] = np.load(os.path.join(root, fp))[:, -1, :].astype(np.float64)
        tids = sorted(stacks)
        f = {}
        for tid in tids:
            hs = stacks[tid]
            neutral = hs[0]
            dirs = hs[1:15] - neutral
            others = [t for t in tids if t != tid]
            M = np.concatenate([stacks[t][1:15] - stacks[t][0] for t in others], axis=0)
            X = M - M.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            P = Vt[:3]
            for i in range(14):
                d = dirs[i]
                nrm = np.linalg.norm(d) + 1e-12
                f[(tid, i)] = float(np.linalg.norm(P.T @ (P @ d)) / nrm)
        feats[name] = f
        print(f"  subspace feats: {name} ({len(f)} items)", flush=True)
    return feats


def build_items_heldout():
    """b0_heldout 电池（seed 0）item 表：H4 domain hold-out 用。"""
    from items_b0 import _TASKS_B0
    ans = {t[0]: t[3] for t in _TASKS_B0}
    items = []
    for name in HELDOUT_CKPTS:
        path = os.path.join(R, f"b0_heldout_{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)["seeds"]["0"]
        for tid, e in d.items():
            vecs = [v for v in (e["frames_std"] + e["frames_new"]) if v is not None]
            if len(vecs) < 6:
                continue
            ref = np.mean([np.array(v, float) for v in vecs], axis=0)
            task_fs = float(np.mean([jsd(np.array(v, float), ref) for v in vecs]))
            srt = np.sort(ref)[::-1]
            mode = int(np.argmax(ref))
            fams = ([FAM_OF_STD14[i] for i in range(14)] +
                    [FAM_OF_NEW12[i] for i in range(12)])
            allf = e["frames_std"] + e["frames_new"]
            fam_jsd = {}
            for i, v in enumerate(allf):
                if v is None:
                    continue
                fam_jsd.setdefault(fams[i], []).append(jsd(np.array(v, float), ref))
            for i, v in enumerate(allf):
                if v is None:
                    continue
                p = np.array(v, float)
                fam = fams[i]
                items.append({
                    "ckpt": name, "task": "H4_" + tid, "frame": i, "family": fam,
                    "flip": int(int(np.argmax(p)) != mode),
                    "frame_jsd": float(jsd(p, ref)),
                    "task_fs": task_fs,
                    "fam_fs": float(np.mean(fam_jsd[fam])),
                    "conf_kappa": float(p.max()),
                    "entropy": float(-(p * np.log(np.maximum(p, 1e-12))).sum()),
                    "content_margin": float(srt[0] - srt[1]),
                    "cap_proxy": float(mode == (ord(ans[tid]) - 65)),
                    "subspace_proj": np.nan,
                })
    return items


def attach_subspace(items, sub):
    for it in items:
        key = (it["task"], it["frame"])
        if it["ckpt"] in sub and key in sub[it["ckpt"]]:
            it["subspace_proj"] = sub[it["ckpt"]][key]
    return items


MODELS = {
    "capability_only": ["cap_proxy"],
    "confidence_only": ["conf_kappa"],
    "entropy_only": ["entropy"],
    "fs_only": ["task_fs"],
    "famlocal_only": ["fam_fs"],
    "subspace_only": ["frame_jsd", "task_fs", "fam_fs", "subspace_proj"],
    "combined": FEATS_ALL,
    "combined_plus_subspace": FEATS_ALL + FEATS_SUB,
}


def eval_split(train, test, models=MODELS):
    res = {}
    ytr = np.array([t["flip"] for t in train], float)
    yte = np.array([t["flip"] for t in test], float)
    for mname, feats in models.items():
        tr_ok = [t for t in train if not any(np.isnan(t.get(f, np.nan)) for f in feats)]
        te_ok = [t for t in test if not any(np.isnan(t.get(f, np.nan)) for f in feats)]
        if len(te_ok) < 30 or len(set(t["flip"] for t in te_ok)) < 2:
            res[mname] = None
            continue
        Xtr = [[t.get(f, np.nan) for f in feats] for t in tr_ok]
        Xte = [[t.get(f, np.nan) for f in feats] for t in te_ok]
        ytr2 = np.array([t["flip"] for t in tr_ok], float)
        yte2 = np.array([t["flip"] for t in te_ok], float)
        m = Logit().fit(Xtr, ytr2)
        s = m.predict(Xte)
        res[mname] = {
            "n_test": len(te_ok), "base_rate": r6(yte2.mean()),
            "auroc": r6(auc_mw(s, yte2)), "auprc": r6(auprc(s, yte2)),
            "brier": r6(brier(s, yte2)), "ece": r6(ece(s, yte2)),
            "top_dec_enrich": r6(topdec_enrichment(s, yte2)),
        }
    return res


def r6(x):
    x = float(x)
    return round(x, 6) if x == x else None


def main():
    print("building battery items...", flush=True)
    items = build_items_battery()
    print(f"  battery items: {len(items)}", flush=True)
    print("building subspace features...", flush=True)
    sub = build_subspace_feature()
    items = attach_subspace(items, sub)
    items_sub = [it for it in items if not np.isnan(it["subspace_proj"])]

    out = {"holdouts": {}}

    # H1: held-out tasks（哈希 32/16）
    def thash(t):
        return int(abs(hash(t)) % 2)
    tr = [it for it in items if thash(it["task"]) == 0]
    te = [it for it in items if thash(it["task"]) == 1]
    ntr_tasks = len(set(t["task"] for t in tr)); nte_tasks = len(set(t["task"] for t in te))
    print(f"H1 tasks: train {ntr_tasks} tasks/{len(tr)} items, test {nte_tasks} tasks/{len(te)}", flush=True)
    out["holdouts"]["H1_tasks"] = {"train_tasks": ntr_tasks, "test_tasks": nte_tasks,
                                   "results": eval_split(tr, te)}

    # H2: held-out templates（家族划分）
    tr_f = {"rephrase", "case", "persona", "verbal"}
    tr = [it for it in items if it["family"] in tr_f]
    te = [it for it in items if it["family"] in {"authority", "peer"}]
    print(f"H2 templates: train {len(tr)}, test {len(te)}", flush=True)
    out["holdouts"]["H2_templates"] = {"train_families": sorted(tr_f),
                                       "test_families": ["authority", "peer"],
                                       "results": eval_split(tr, te)}

    # H3: held-out model family（双向）
    for held in ["llama32", "qwen15"]:
        tr = [it for it in items if not it["ckpt"].startswith(held)]
        te = [it for it in items if it["ckpt"].startswith(held)]
        print(f"H3 family held-out={held}: train {len(tr)}, test {len(te)}", flush=True)
        out["holdouts"][f"H3_family_{held}"] = {"results": eval_split(tr, te)}

    # H3s: 同 H3 但只用带 subspace 的 ladder 模型 + combined_plus_subspace
    tr = [it for it in items_sub if not it["ckpt"].startswith("llama32")]
    te = [it for it in items_sub if it["ckpt"].startswith("llama32")]
    out["holdouts"]["H3s_family_llama32_with_subspace"] = {"results": eval_split(
        tr, te, models={"combined": MODELS["combined"],
                        "combined_plus_subspace": MODELS["combined_plus_subspace"],
                        "subspace_only": MODELS["subspace_only"]})}

    # H4: domain hold-out（训练 48 电池 / 测试 b0 新任务电池）
    held = build_items_heldout()
    print(f"H4 domain: train {len(items)}, test {len(held)}", flush=True)
    out["holdouts"]["H4_domain_b0battery"] = {"results": eval_split(items, held)}

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"saved {OUT}", flush=True)

    # 控制台摘要
    for h, hd in out["holdouts"].items():
        print(f"\n== {h}")
        for m, v in (hd["results"] or {}).items():
            if v:
                print(f"  {m:24} AUROC={v['auroc']} AUPRC={v['auprc']} "
                      f"ECE={v['ece']} enrich={v['top_dec_enrich']}")


if __name__ == "__main__":
    main()
