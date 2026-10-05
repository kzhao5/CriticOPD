#!/usr/bin/env python3
"""主表(tab:main)多 seed 版:从统一协议的评测汇总读每个 seed 的六项分数,三个 seed 都完整的行写成「均值 ± 标准差」,
否则保留 seed 42 的单次结果(并在 JSON 里记下缺哪些 seed)。标准差为样本标准差(ddof=1)。
输出 main_table_multiseed.{json,md} 与 main_table_body.tex(tabular 内容,从 \\toprule 到 \\bottomrule)。
每个分片都按题数 x 采样数核对,与评测守护(multiseed/eval_guard.sh)的核对一致。"""
import json, os
import numpy as np

EV = "/home/kzhao2/Relay-OPD/outputs/eval_out"
OUT = "/home/kzhao2/Relay-OPD/results_final"
BENCH = {"aime24": (30, 32), "aime25": (30, 32), "amc23": (83, 32), "math500": (500, 8), "olympiad": (675, 8), "minerva": (272, 8)}
B = list(BENCH)
FIXED = {   # 只评一次的行(论文现值)
    "Teacher": [62.19, 48.12, 85.99, 94.83, 72.26, 45.54],
    "Student-0.6B": [2.08, 1.35, 18.98, 45.52, 16.94, 13.69],
    "Student-1.7B": [11.98, 9.17, 41.00, 73.22, 39.80, 29.23],
}
# 方法  -> (seed 42 的运行名, seed 43/44 的基础名, 所选步数)
SPEC = {
    "0.6B": [("SFT", "sft_pt06b", "sft_pt06b", 139), ("KD", "seqkd_pt06b", "seqkd_pt06b", 139), ("GRPO", "grpo_pt06b", "grpo_pt06b", 80),
             ("OPD", "opd_pt06b", "opd_pt06b", 40), ("TRD", "trd_pt06b", "trd_pt06b", 139), ("FastOPD", "fastopd8192_pt06b", "fastopd8192_pt06b", 40),
             ("SKD", "skd_pt06b", "skd_pt06b", 40), ("RelayOPD", "relay_pt06b", "relay_pt06b", 60),
             ("CriticOPD", "criticopd_final_pt06b_seed42", "criticopd_final_pt06b", None)],
    "1.7B": [("SFT", "sft_pt17b", "sft_pt17b", 40), ("KD", "seqkd_pt17b", "seqkd_pt17b", 139), ("GRPO", "grpo_pt17b", "grpo_pt17b", 120),
             ("OPD", "opd_1p7b", "opd_1p7b", 80), ("TRD", "trd_pt17b", "trd_pt17b", 40), ("FastOPD", "fastopd8192_1p7b", "fastopd8192_1p7b", 60),
             ("SKD", "skd_pt17b", "skd_pt17b", 120), ("RelayOPD", "relay_1p7b", "relay_1p7b", 40),
             ("CriticOPD", "criticopd_final_pt17b_seed42", "criticopd_final_pt17b", 40)],
}


def load(run, step):
    """返回六项分数(百分数)或 None(缺任何一项或题数 / 采样数不对)。"""
    d = f"{EV}/{run}_u32/step_{step}"
    out = []
    for b, (n, k) in BENCH.items():
        f = f"{d}/{b}.summary.json"
        if not os.path.exists(f): return None
        s = json.load(open(f))
        if s["n_problems"] != n or s["n_samples"] != k: return None
        jf = f"{d}/{b}.jsonl"
        if os.path.exists(jf) and sum(1 for _ in open(jf)) != n * k: return None
        out.append(100 * s["avg@k"])
    return out


def pick_critic_step(base, scale):
    """0.6B 的 CriticOPD 没有先验峰值:三个 seed 在 40、60、80 都评完后,取前四项均值(主表选点规则)按 seed 平均最高的那一步。"""
    cand = {}
    for st in (40, 60, 80):
        rs = [load(f"{base}_seed{s}", st) for s in (42, 43, 44)]
        if all(r is not None for r in rs): cand[st] = np.mean([np.mean(r[:4]) for r in rs])
    return max(cand, key=cand.get) if len(cand) == 3 else None


R, RAW = {}, {}
for scale, rows in SPEC.items():
    for name, r42, base, step in rows:
        if name == "CriticOPD" and step is None: step = pick_critic_step(base, scale)
        seeds = {}
        if step is not None:
            for s in (42, 43, 44):
                run = r42 if s == 42 else f"{base}_seed{s}"
                v = load(run, step)
                if v is not None: seeds[s] = v
        RAW[f"{scale}/{name}"] = seeds
        rec = {"step": step, "seeds": {s: [round(x, 2) for x in v] for s, v in seeds.items()}, "complete": len(seeds) == 3}
        if seeds:
            M = np.array(list(seeds.values())); av = M.mean(1)
            rec["mean"] = [round(x, 2) for x in M.mean(0)] + [round(av.mean(), 2)]
            if len(seeds) == 3: rec["std"] = [round(x, 2) for x in M.std(0, ddof=1)] + [round(av.std(ddof=1), 2)]
        R[f"{scale}/{name}"] = rec
json.dump(R, open(f"{OUT}/main_table_multiseed.json", "w"), indent=1)


# ---------------- LaTeX
def cells(scale, name):
    """表里显示的 (数值列表, 标准差列表或 None, 步数);CriticOPD 不完整时用占位行。"""
    r = R[f"{scale}/{name}"]
    if r["complete"]: return r["mean"], r["std"], r["step"]
    v = RAW[f"{scale}/{name}"].get(42)
    return (list(v) + [float(np.mean(v))]) if v else None, None, r["step"]


def fmt(x, sd, rank):
    s = f"{x:.2f}"
    s = f"\\textbf{{{s}}}" if rank == 0 else (f"\\underline{{{s}}}" if rank == 1 else s)
    return s + (f"$_{{\\pm {sd:.2f}}}$" if sd is not None else "")


def block(scale):
    names = [n for n, *_ in SPEC[scale]]
    shown = {n: cells(scale, n) for n in names}
    rank = {}
    for j in range(7):                                 # 每列(六项 + Avg)按显示的数值排名,只在学生方法之间比较
        vals = sorted({round(shown[n][0][j], 2) for n in names if shown[n][0] is not None}, reverse=True)
        for n in names:
            if shown[n][0] is not None:
                v = round(shown[n][0][j], 2); rank[(n, j)] = vals.index(v) if v in vals[:2] else 9
    L = [f"\\midrule\n\\multicolumn{{8}}{{c}}{{\\textit{{Student: Qwen3-{scale}-Non-Thinking}}}} \\\\\n\\midrule\n"]
    st = FIXED[f"Student-{scale}"]
    L.append("Student\n& " + " & ".join(f"{x:.2f}" for x in st) + f"\n& {np.mean(st):.2f} \\\\\n")
    for n in names:
        v, sd, step = shown[n]
        label = "\\textbf{\\ours{}}" if n == "CriticOPD" else n
        label += f"$_{{@{step}}}$" if step else ""
        pre = "\\rowcolor{blue!8}\n" if n == "CriticOPD" else ""
        if v is None:
            L.append(pre + label + "\n& " + " & ".join(["XX"] * 6) + "\n& XX \\\\\n"); continue
        c = [fmt(v[j], sd[j] if sd else None, rank[(n, j)]) for j in range(7)]
        L.append(pre + label + "\n& " + " & ".join(c[:6]) + "\n& " + c[6] + " \\\\\n")
    # Δ vs OPD(显示的 CriticOPD 减显示的 OPD)
    cv, ov = shown["CriticOPD"][0], shown["OPD"][0]
    if cv is None or ov is None:
        d = ["XX"] * 7
    else:
        d = []
        for j in range(7):
            x = cv[j] - ov[j]
            d.append(f"\\textcolor{{green!50!black}}{{+{x:.2f}}}" if x >= 0 else f"\\textcolor{{red!60!black}}{{$-${-x:.2f}}}")
        d = [s for s in d]
    if cv is None or ov is None:
        d = ["\\textcolor{green!50!black}{XX}"] * 7
    L.append("\\rowcolor{blue!8}\n\\textcolor{green!50!black}{$\\Delta$ vs OPD}\n" + "".join(f"& {x}\n" for x in d) + "\\\\\n")
    return "\n".join(L)


T = ["\\toprule\n\\textbf{Method}\n& \\textbf{AIME24}\n& \\textbf{AIME25}\n& \\textbf{AMC23}\n& \\textbf{MATH500}\n& \\textbf{OlympiadBench}\n"
     "& \\textbf{Minerva}\n& \\textbf{Avg}\n\\\\\n\\midrule\n",
     "Teacher\n& " + " & ".join(f"{x:.2f}" for x in FIXED["Teacher"]) + f"\n& {np.mean(FIXED['Teacher']):.2f} \\\\\n",
     block("0.6B"), block("1.7B"), "\\bottomrule\n"]
open(f"{OUT}/main_table_body.tex", "w").write("\n".join(T))

# ---------------- Markdown 总览
md = ["| 规模 | 方法 | 步数 | 完整 seed | " + " | ".join(B) + " | Avg |", "|---" * (len(B) + 5) + "|"]
for k, r in R.items():
    sc, n = k.split("/")
    if "mean" in r:
        vals = [f"{m:.2f}" + (f" ± {s:.2f}" if "std" in r else "") for m, s in zip(r["mean"], r.get("std", [0] * 7))]
    else: vals = ["—"] * 7
    md.append(f"| {sc} | {n} | {r['step']} | {','.join(map(str, sorted(r['seeds'])))} | " + " | ".join(vals) + " |")
open(f"{OUT}/main_table_multiseed.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))


# ---------------- 主表下面那段正文(1.7B),全部数字由表中显示的值算出
def paragraph():
    names = [n for n, *_ in SPEC["1.7B"]]
    shown = {n: cells("1.7B", n)[0] for n in names}
    c, o = shown["CriticOPD"], shown["OPD"]
    if c is None or o is None: return None
    r2 = lambda x: float(f"{x:.2f}")                    # 正文与表格一致:先按显示的两位小数取值再相减
    d = [r2(c[j]) - r2(o[j]) for j in range(7)]
    full = ["AIME 2024", "AIME 2025", "AMC 2023", "MATH500", "OlympiadBench", "Minerva Math"]
    best = [j for j in range(6) if all(r2(c[j]) >= r2(shown[n][j]) for n in names if shown[n] is not None)]
    num = {6: "all six", 5: "five of the six", 4: "four of the six", 3: "three of the six", 2: "two of the six", 1: "one of the six"}
    def vs(n):
        g = r2(c[6]) - r2(shown[n][6])
        return f"${abs(g):.2f}$ points {'above' if g >= 0 else 'below'} that of {n}"
    rb = [full[j] for j in range(6) if r2(shown["RelayOPD"][j]) > r2(c[j])]
    rb_txt = (", which is better only on " + rb[0]) if len(rb) == 1 else ((", which is better on " + " and ".join(rb)) if rb else "")
    sgn = lambda x: f"$+{x:.2f}$" if x >= 0 else f"$-{-x:.2f}$"
    best_avg = all(r2(c[6]) >= r2(shown[n][6]) for n in names if shown[n] is not None)
    lines = ["Table~\\ref{tab:main} reports the main results.",
             f"For the Qwen3-1.7B student, \\ours{{}} raises the average of OPD from ${r2(o[6]):.2f}$ to",
             f"${r2(c[6]):.2f}$.",
             f"The gains are largest on the competition benchmarks, with {sgn(d[0])} on AIME 2024,",
             f"{sgn(d[1])} on AIME 2025, and {sgn(d[2])} on AMC 2023, while MATH500 improves by ${d[3]:.2f}$",
             f"and Minerva Math by ${d[5]:.2f}$."]
    if best:
        lines.append(f"Among the students, \\ours{{}} obtains {'the best average and ' if best_avg else ''}the best accuracy on")
        lines.append(f"{num[len(best)]} benchmarks.")
    lines.append(f"Its average is {vs('FastOPD')} and {vs('RelayOPD')}{rb_txt}.")
    lines += ["Both baselines change how rollouts are generated, whereas \\ours{} changes how",
              "incorrect rollouts are supervised."]
    assert min(d[:3]) > max(d[3:6]), "竞赛题增益最大的说法不再成立,需改写正文"
    return "\n".join(lines)


P_ = paragraph()
if P_: open(f"{OUT}/main_table_paragraph.tex", "w").write(P_ + "\n"); print(P_)
