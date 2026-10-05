"""E5 × E8:critic 列出的每条错误按相对关闭区间的位置(之前 / 区间 ±1 / 之后)分组,看 8B 判定是否属实,
以及该错误所在位置 teacher 续写的成功率(E1 里最近的截断点)。"""
import json, collections, sys, numpy as np
O = sys.argv[1] if len(sys.argv) > 1 else "out"
R = {r["rollout_id"]: r for r in map(json.loads, open(f"{O}/rollouts.jsonl"))}
A = {int(k): v for k, v in json.load(open(f"{O}/closing.json"))["A"].items()}
import glob
J = {(x["kind"], x["rid"]): x["labels"] for f in sorted(glob.glob(f"{O}/judge_[0-9]*.jsonl")) for x in map(json.loads, open(f))}
E1 = collections.defaultdict(dict)
for x in map(json.loads, open(f"{O}/graded_e1.jsonl")): E1[x["rid"]][(x["actor"], x["cut"])] = np.mean([s["correct"] for s in x["samples"]])
for kind in ("full_nogt", "full_gt"):
    tab = collections.defaultdict(collections.Counter); vt = collections.defaultdict(list); first = collections.defaultdict(collections.Counter)
    for x in map(json.loads, open(f"{O}/critic_{kind}.jsonl")):
        rid = x["rid"]; a = A[rid]; lo, hi = a["prev_seg"], a["close_seg"] - 1; r = R[rid]
        labs = J.get((kind, rid), [None] * len(x["errs_raw"]))
        for i, (e, l) in enumerate(zip(x["errs_raw"], labs)):
            g = "before" if e[0] < lo - 1 else ("in±1" if e[0] <= hi + 1 else "after")
            tab[g][l or "UNPARSED"] += 1
            if i == 0: first[g][l or "UNPARSED"] += 1
            c = min([c for c in range(len(r["cut_points"])) if r["cut_points"][c] >= e[0] + 1] or [len(r["cut_points"]) - 1])   # 该错误之后最近的截断点
            vt[g].append(E1[rid][("teacher", c)])
    print(f"== {kind}")
    for g in ("before", "in±1", "after"):
        n = sum(tab[g].values()); nf = sum(first[g].values())
        gen = lambda c: (sum(c.values()) - c["NOT_AN_ERROR"] - c["UNPARSED"]) / max(sum(c.values()) - c["UNPARSED"], 1)
        print(f"  {g:7s} 全部错误 {n:3d} 属实 {gen(tab[g]):.2f} {dict(tab[g])}")
        print(f"  {'':7s} 第一条   {nf:3d} 属实 {gen(first[g]):.2f};该位置 teacher 续写成功率 {np.mean(vt[g]) if vt[g] else float('nan'):.2f}")
