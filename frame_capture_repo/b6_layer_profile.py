"""b6_layer_profile.py — 纯 CPU 层间扫描：帧方向 vs 内容子空间的对齐度。

目的：b6 主实验只跑了 L20/L24 两层。本脚本用 b1_hidden 存档（全部层），
对每一层算：
  - cos(frame_dir, content_subspace)：逐条帧方向投到内容子空间上的余弦（LOTO 口径）
  - energy_ret：正交化后帧方向模长保留率
  - rank_shared：top-3 帧子空间正交化后剩下的有效秩
用来回答：FS 的"共享分量"是 L24 独有，还是全层普遍。

用法： python b6_layer_profile.py [model_name]
"""
import os
import sys
import json
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from b6_math import (load_hidden_dirs, topk_subspace, content_subspace,  # noqa: E402
                     orth_to_content, orth_matched)

K = 3


def profile_layer(root, stack_idx):
    loto = load_hidden_dirs(root, stack_idx)
    tids = sorted(loto)
    cos_c, ret_c, ret_m, ranks = [], [], [], []
    for t in tids:
        others = [x for x in tids if x != t]
        C = content_subspace(loto, others)
        Fr = np.concatenate([loto[x]["frames"] for x in others], axis=0)
        FrP = orth_to_content(Fr, C)
        # 逐条方向对内容子空间的投影占比
        proj = Fr @ (C.T @ C)
        c = np.linalg.norm(proj, axis=1) / (np.linalg.norm(Fr, axis=1) + 1e-12)
        cos_c.append(c.mean())
        ret_c.append(np.linalg.norm(FrP, axis=1).mean() /
                     (np.linalg.norm(Fr, axis=1).mean() + 1e-12))
        Psh = topk_subspace(Fr, K)
        Pm = orth_matched(Psh, C)
        ret_m.append(np.linalg.norm(Pm, axis=1).mean() /
                     (np.linalg.norm(Psh, axis=1).mean() + 1e-12))
        ranks.append(int(Pm.shape[0]))
    return dict(cos=float(np.mean(cos_c)), ret_frames=float(np.mean(ret_c)),
                ret_topk=float(np.mean(ret_m)), rank_matched=float(np.mean(ranks)))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen15-sft"
    root = os.path.join(THIS, "results", f"b1_hidden_{name}")
    if not os.path.isdir(root):
        print("missing", root)
        return
    nlayer = np.load(os.path.join(root, sorted(os.listdir(root))[0])).shape[1]
    out = {}
    print(f"== {name}  (n_layers(stack)={nlayer})")
    print("%5s %10s %12s %12s %8s" % ("layer", "cos_fc", "ret_frames", "ret_top3", "rank_m"))
    for l in range(1, nlayer):
        try:
            r = profile_layer(root, l)
        except Exception as ex:
            print(f"  L{l}: err {ex}")
            continue
        out[str(l)] = r
        mark = "  <== main" if l in (21, 25) else ""
        print("%5d %10.4f %12.4f %12.4f %8.1f%s"
              % (l, r["cos"], r["ret_frames"], r["ret_topk"], r["rank_matched"], mark))
    dst = os.path.join(THIS, "results", f"b6_layer_profile_{name}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved", dst)


if __name__ == "__main__":
    main()
