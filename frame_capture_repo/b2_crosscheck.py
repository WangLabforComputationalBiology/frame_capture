# -*- coding: utf-8 -*-
"""b2_crosscheck.py — 把 b2 原始存档（论文 §9 B2 的出处）四模型×两层的
ablate_shared / ablate_content ΔFS/ΔCS 全部列出来，作为 b6 的跨家族对照基线。"""
import glob
import json
import os

THIS = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(THIS, "results")

for fp in sorted(glob.glob(os.path.join(R, "b2_patch_analysis_*_v2.json"))):
    name = os.path.basename(fp)[len("b2_patch_analysis_"):-len("_v2.json")]
    d = json.load(open(fp, encoding="utf-8"))
    print(f"== {name}")
    for L, lay in sorted(d.get("layers", {}).items(), key=lambda kv: int(kv[0])):
        parts = []
        for c in ("ablate_shared", "ablate_content"):
            v = lay.get(c)
            if not v:
                continue
            ci = v.get("delta_fs_ci") or [None, None]
            sig = "**" if (ci[0] is not None and ci[1] is not None
                           and (ci[1] < 0 or ci[0] > 0)) else "  "
            parts.append("%s dFS=%+.5f[%s,%s]%s dCS=%+.4f"
                         % (c, v["delta_fs_mean"],
                            "%+.5f" % ci[0] if ci[0] is not None else "?",
                            "%+.5f" % ci[1] if ci[1] is not None else "?",
                            sig, v["delta_cs_mean"]))
        print(f"  L{L}  " + " | ".join(parts))
