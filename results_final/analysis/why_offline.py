#!/usr/bin/env python3
"""CriticOPD 为什么有效:不需要训练的分析(文档阶段 1 与 2D)。读统一协议评测的逐条生成(32768 上限)。
1A  相同生成上限下的准确率:上限 B 时的结果 = 生成长度 <= B 且答对(32768 的生成截到 B 与直接用 B 生成同分布)。
1B  回头检查类表达的次数;不同候选答案(\\boxed)个数;第一个候选 -> 最终答案的转移(错->对 / 对->错 / 不变)。
1C  第一个候选就对且最终也对的回答中,出现「我前面错了」类表达的比例。
1D  按 teacher 在该题上的正确率分组(AIME 24/25 合并 60 题)。
2D  OPD 逐 token advantage 与 V^T 的关系(Section 2 诊断数据)。
输出 why_offline.json 与 why_offline.md。"""
import asyncio, collections, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, "/home/kzhao2/Relay-OPD/diag2")
os.environ.setdefault("DIAG_OUT", "/home/kzhao2/Relay-OPD/diag2/out_all")
import diag as D   # noqa: E402  (判分子进程池、final_answer)

EV = "/home/kzhao2/Relay-OPD/outputs/eval_out"
OUT = "/home/kzhao2/Relay-OPD/results_final/analysis"
B6 = ["aime24", "aime25", "amc23", "math500", "olympiad", "minerva"]
BUDGETS = [4096, 8192, 16384, 32768]
MODELS = {   # 方法 -> [(评测目录名, 步数)]
    "OPD": [("opd_1p7b", 80), ("opd_1p7b_seed43", 80), ("opd_1p7b_seed44", 80)],
    "CriticOPD": [(f"criticopd_final_pt17b_seed{s}", 40) for s in (42, 43, 44)],
    "RelayOPD": [("relay_1p7b", 40), ("relay_1p7b_seed43", 40), ("relay_1p7b_seed44", 40)],
    "FastOPD": [("fastopd8192_1p7b", 60), ("fastopd8192_1p7b_seed43", 60), ("fastopd8192_1p7b_seed44", 60)],
    "CriticOPD-every-rollout": [("criticopd_all_pt17b_seed42", 40)],
    "CriticOPD-prefix-loss": [("criticopd_prefix_pt17b_seed42", 40)],
    "CriticOPD-first-error(R4GT)": [("criticopd_R4GT_pt17b", 40)],
    "Rewrite-no-feedback(R2)": [("criticopd_R2v2_pt17b", 60)],
}
REFLECT = re.compile(r"\b(wait|let me (re-?check|verify|double-?check|re-?examine|re-?evaluate|reconsider)|double-?check|"
                     r"i made (a|an) (mistake|error)|that'?s (wrong|incorrect)|mistake|re-?examine|let'?s check)\b", re.I)
SELF_WRONG = re.compile(r"\b(i made (a|an) (mistake|error)|that'?s (wrong|incorrect)|was (wrong|incorrect)|mistake)\b", re.I)


def boxed_all(t):
    out, i = [], 0
    while True:
        i = t.find("\\boxed{", i)
        if i < 0: return out
        j, d = i + 7, 1
        while j < len(t) and d:
            d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
        if d == 0: out.append((i, t[i + 7:j - 1].strip()))
        i = j


def load(run, st, bench):
    f = f"{EV}/{run}_u32/step_{st}/{bench}.jsonl"
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def main():
    R = {}
    pairs, keys = [], []          # 第一个候选答案的判分任务
    rows_all = {}
    for m, runs in MODELS.items():
        rows_all[m] = {}
        for run, st in runs:
            for b in B6:
                rows = load(run, st, b)
                rows_all[m][(run, b)] = rows
                for k, r in enumerate(rows):
                    c = boxed_all(r["gen_text"])
                    r["_cands"] = c
                    if len(c) >= 2 and c[0][1] != c[-1][1]:
                        pairs.append((f"\\boxed{{{c[0][1]}}}", r["gt"])); keys.append((m, run, b, k))
    print("grading first candidates:", len(pairs), flush=True)
    res = D.grade_many(pairs) if pairs else []
    first_ok = {k: bool(v) for k, v in zip(keys, res)}
    for m, runs in MODELS.items():
        acc = {B: [] for B in BUDGETS}; lens = collections.defaultdict(list)
        refl, refl_any, ncand, trans, spur = [], [], [], collections.Counter(), []
        for run, st in runs:
            per_b = {B: [] for B in BUDGETS}
            for b in B6:
                rows = rows_all[m][(run, b)]
                if not rows: continue
                for B in BUDGETS:
                    per_b[B].append(np.mean([r["correct"] and r["gen_len"] <= B for r in rows]))
                lens[b] += [r["gen_len"] for r in rows]
                for k, r in enumerate(rows):
                    t = r["gen_text"]; c = r["_cands"]
                    n = len(REFLECT.findall(t)); refl.append(n); refl_any.append(n > 0)
                    ncand.append(len({x[1] for x in c}))
                    if not c: trans["no_answer"] += 1; continue
                    f_ok = first_ok.get((m, run, b, k), bool(r["correct"])) if len(c) >= 2 and c[0][1] != c[-1][1] else bool(r["correct"])
                    trans[("right" if f_ok else "wrong") + "->" + ("right" if r["correct"] else "wrong")] += 1
                    if f_ok and r["correct"]:
                        spur.append(bool(SELF_WRONG.search(t[c[0][0]:])))
            for B in BUDGETS:
                if len(per_b[B]) == 6: acc[B].append(100 * np.mean(per_b[B]))
        tot = sum(trans.values())
        R[m] = {"n_seeds": len(acc[32768]),
                "avg6_by_budget": {B: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None) for B, v in acc.items()},
                "len_median": {b: float(np.median(v)) for b, v in lens.items()}, "len_p90": {b: float(np.percentile(v, 90)) for b, v in lens.items()},
                "reflect_per_response": float(np.mean(refl)), "reflect_any": float(np.mean(refl_any)),
                "distinct_candidates": float(np.mean(ncand)),
                "transitions": {k: v / tot for k, v in trans.items()},
                "spurious_self_correction": float(np.mean(spur)) if spur else None}
        print(m, json.dumps({k: R[m][k] for k in ("avg6_by_budget", "reflect_any", "transitions")}, default=str)[:400], flush=True)
    # 1D:按 teacher 在每道 AIME 题上的正确率分组
    tdir = sorted(glob.glob(f"{EV}/teacher_4b*/step_*")) + sorted(glob.glob(f"{EV}/teacher_4b*"))
    tacc = {}
    for d in tdir:
        for b in ("aime24", "aime25"):
            f = f"{d}/{b}.jsonl"
            if os.path.exists(f) and (b not in {k[0] for k in tacc}):
                rows = [json.loads(l) for l in open(f)]
                per = collections.defaultdict(list)
                for r in rows: per[r["problem_idx"]].append(r["correct"])
                for p, v in per.items(): tacc[(b, p)] = float(np.mean(v))
    def cat(v): return "teacher>=0.75" if v >= 0.75 else ("teacher 0.25-0.75" if v >= 0.25 else "teacher<0.25")
    cats = collections.Counter(cat(v) for v in tacc.values())
    by_cat = {}
    for m, runs in MODELS.items():
        acc = collections.defaultdict(list)
        for run, st in runs:
            for b in ("aime24", "aime25"):
                per = collections.defaultdict(list)
                for r in rows_all[m][(run, b)]: per[r["problem_idx"]].append(r["correct"])
                for p, v in per.items():
                    if (b, p) in tacc: acc[cat(tacc[(b, p)])].append(np.mean(v))
        by_cat[m] = {c: float(100 * np.mean(v)) for c, v in acc.items()}
    # 2D:OPD advantage 与 V^T(Section 2 诊断数据:错误轨迹的逐 token logprob + 各截断点 V^T)
    O = os.environ["DIAG_OUT"]
    rollouts = {r["rollout_id"]: r for r in D.jl(f"{O}/rollouts.jsonl")}
    sc = collections.defaultdict(dict)
    for a in ("teacher", "student"):
        for x in D.jlg(f"{O}/score_{a}_[0-9]*.jsonl"): sc[x["rid"]][a] = np.array(x["lp"])
    VT = collections.defaultdict(dict)
    for x in D.jl(f"{O}/graded_e1.jsonl"):
        if x["actor"] == "teacher": VT[x["rid"]][x["cut"]] = float(np.mean([s["correct"] for s in x["samples"]]))
    bins = collections.defaultdict(list)
    for rid, s in sc.items():
        if len(s) < 2 or rid not in rollouts: continue
        r = rollouts[rid]; adv = s["teacher"] - s["student"]; ct = r["cut_tokens"]
        for c in range(len(ct)):
            if c not in VT[rid]: continue
            lo, hi = ct[c], (ct[c + 1] if c + 1 < len(ct) else len(adv))
            v = VT[rid][c]; g = "<0.2" if v < 0.2 else ("0.2-0.5" if v < 0.5 else ("0.5-0.8" if v < 0.8 else ">=0.8"))
            bins[g] += adv[lo:hi].tolist()
    adv_vt = {g: {"n_tokens": len(v), "mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "frac_positive": float(np.mean(np.array(v) > 0))} for g, v in bins.items()}
    out = {"models": R, "aime_by_teacher_category": by_cat, "category_sizes": dict(cats), "adv_by_VT": adv_vt}
    json.dump(out, open(f"{OUT}/why_offline.json", "w"), indent=1, default=str)
    L = ["# CriticOPD 为什么有效:离线分析(阶段 1、2D)\n", "## 1A 相同生成上限下的六项均值(三个 seed 的均值 ± 标准差)\n",
         "| 方法 | " + " | ".join(str(B) for B in BUDGETS) + " | AIME 中位长度 | AIME p90 |", "|---" * (len(BUDGETS) + 3) + "|"]
    for m, d in R.items():
        cells = [f"{d['avg6_by_budget'][B][0]:.2f}" + (f" ± {d['avg6_by_budget'][B][1]:.2f}" if d['avg6_by_budget'][B][1] is not None else "") for B in BUDGETS]
        al = np.mean([d["len_median"].get("aime24", np.nan), d["len_median"].get("aime25", np.nan)])
        ap = np.mean([d["len_p90"].get("aime24", np.nan), d["len_p90"].get("aime25", np.nan)])
        L.append(f"| {m} | " + " | ".join(cells) + f" | {al:.0f} | {ap:.0f} |")
    L += ["\n## 1B / 1C 回头检查与答案转移(全部六项的回答)\n",
          "| 方法 | 每条回答的检查类表达 | 至少一次 | 不同候选答案数 | 错→对 | 对→错 | 对→对 | 错→错 | 无答案 | 无端「我错了」 |", "|---" * 10 + "|"]
    for m, d in R.items():
        t = d["transitions"]
        L.append(f"| {m} | {d['reflect_per_response']:.2f} | {d['reflect_any']:.1%} | {d['distinct_candidates']:.2f} | {t.get('wrong->right', 0):.1%} | "
                 f"{t.get('right->wrong', 0):.1%} | {t.get('right->right', 0):.1%} | {t.get('wrong->wrong', 0):.1%} | {t.get('no_answer', 0):.1%} | "
                 f"{d['spurious_self_correction']:.1%} |")
    L += [f"\n## 1D AIME 按 teacher 正确率分组的准确率(题数 {dict(cats)})\n", "| 方法 | " + " | ".join(sorted(cats)) + " |", "|---" * (len(cats) + 1) + "|"]
    for m, d in by_cat.items(): L.append(f"| {m} | " + " | ".join(f"{d.get(c, float('nan')):.1f}" for c in sorted(cats)) + " |")
    L += ["\n## 2D OPD 逐 token advantage 按 V^T 分组(Section 2 诊断的错误轨迹)\n", "| V^T | token 数 | 均值 | p10 | 中位数 | p90 | 为正的比例 |", "|---" * 7 + "|"]
    for g in ("<0.2", "0.2-0.5", "0.5-0.8", ">=0.8"):
        if g in adv_vt:
            d = adv_vt[g]; L.append(f"| {g} | {d['n_tokens']} | {d['mean']:.3f} | {d['p10']:.3f} | {d['p50']:.3f} | {d['p90']:.3f} | {d['frac_positive']:.1%} |")
    open(f"{OUT}/why_offline.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
