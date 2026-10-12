#!/usr/bin/env python3
"""Figures 3, 4 and 5(a) of the CriticOPD paper from a diagnosis output directory.

Definitions are exactly those of the paper (checked to reproduce the paper's Qwen3-1.7B figures):
  Figure 3   success rate of the teacher and the student from each cut (mean over the e1 continuations);
             (a) correct rollouts, (b) incorrect rollouts split by the teacher's pass rate on the empty prefix
             (>= 0.5: "teacher solves", solid; else dashed).
  Figure 4   (a) incorrect rollouts whose problem the teacher solves: outcome of every teacher continuation
             (correct / the student's original wrong answer / other); (b) critiques of truncated prefixes
             (class A rollouts, cuts close-2..close+4): per critique, the fraction of listed errors that the judge
             does not label NOT_AN_ERROR, averaged within groups of the teacher's success V^T at that prefix;
             dashed line: mean V^T of the group.
  Figure 5(a) success rates around the method cut: before the cut from e1, after the cut on the original tail
             (dashed, nobody sees the critique) and on the rewrite (solid; the teacher sees the critique, the
             student does not).
"""
import collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2B6CB0", "#E8743B"
DASH = (0, (2.6, 1.4))
rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def jlg(pat): return [x for f in sorted(glob.glob(pat)) for x in jl(f)]


def boot(v, n=2000):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) == 0: return (float("nan"),) * 3 + (0,)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)


def load(d):
    """Main batch only (Figures 3 and 4); the extension batch enters Figure 5(a) only, as in the paper."""
    R = {r["rollout_id"]: r for r in jl(f"{d}/rollouts.jsonl") if r.get("batch", "main") == "main"}
    E1 = collections.defaultdict(dict)
    for x in jl(f"{d}/graded_e1.jsonl"): E1[x["rid"]][(x["actor"], x["cut"])] = x["samples"]
    return R, E1


def acc(ss): return float(np.mean([s["correct"] for s in ss]))
def rlen(r): return r["resp_len"] if "resp_len" in r else len(r["resp_ids"])
def solvable(E1, rid): return acc(E1[rid][("teacher", 0)]) >= 0.5


def pos(R, rids, c):
    return float(np.mean([100 * R[r]["cut_tokens"][c] / max(1, rlen(R[r])) for r in rids if c < len(R[r]["cut_tokens"])]))


def curves(R, E1, rids):
    n = max(len(R[r]["cut_points"]) for r in rids)
    xs = [pos(R, rids, c) for c in range(n)]
    return xs, {k: [boot([acc(E1[r][(a, c)]) for r in rids if (a, c) in E1[r]]) for c in range(n)]
                for k, a in (("T", "teacher"), ("S", "student"))}


def band(ax, xs, st, color, ls, marker, alpha_line=1.0, z=3):
    m = np.array([s[0] for s in st]); lo = np.array([s[1] for s in st]); hi = np.array([s[2] for s in st])
    ax.fill_between(xs, lo, hi, color=color, alpha=0.14 if ls == "-" else 0.08, lw=0, zorder=z - 1)
    ax.plot(xs, m, color=color, ls=ls, lw=1.2, marker=marker, ms=3.4 if ls == "-" else 2.9, alpha=alpha_line,
            markeredgecolor="white", markeredgewidth=0.45, zorder=z)


def actor_legend(fig, B0, S, H):
    h = [Line2D([0], [0], color=BLUE, lw=1.2, marker="o", ms=3.6, markeredgecolor="white", markeredgewidth=0.45),
         Line2D([0], [0], color=ORANGE, lw=1.2, marker="s", ms=3.6, markeredgecolor="white", markeredgewidth=0.45)]
    fig.legend(h, ["Teacher", "Student"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, (B0 + S) / H + 0.115),
               handlelength=2.0, columnspacing=2.2, borderaxespad=0)


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(f"{out}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def fig3(d, out, draw=True):
    R, E1 = load(d)
    cor = sorted(r for r in R if R[r]["correct"]); inc = sorted(r for r in R if not R[r]["correct"])
    sol = [r for r in inc if solvable(E1, r)]; uns = [r for r in inc if not solvable(E1, r)]
    N = {"n_correct": len(cor), "n_incorrect": len(inc), "n_teacher_solves": len(sol), "n_teacher_does_not": len(uns)}
    xs, cv = curves(R, E1, cor); N["correct"] = {k: [round(s[0], 3) for s in v] for k, v in cv.items()}
    parts = []
    for rids, ls, key in ((sol, "-", "incorrect_teacher_solves"), (uns, DASH, "incorrect_teacher_does_not")):
        if rids:
            xs2, cv2 = curves(R, E1, rids); parts.append((xs2, cv2, ls))
            N[key] = {k: [round(s[0], 3) for s in v] for k, v in cv2.items()}
    if not draw: return N
    W, H = 3.03, 1.86; S, L, G, B0 = 1.06, 0.40, 0.30, 0.36
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([L / W, B0 / H, S / W, S / H]); b = fig.add_axes([(L + S + G) / W, B0 / H, S / W, S / H])
    for k, col, mk in (("T", BLUE, "o"), ("S", ORANGE, "s")): band(a, xs, cv[k], col, "-", mk)
    for xs2, cv2, ls in parts:
        for k, col, mk in (("T", BLUE, "o"), ("S", ORANGE, "s")): band(b, xs2, cv2[k], col, ls, mk, alpha_line=1.0 if ls == "-" else 0.7)
        if ls == "-":
            for k, col in (("T", BLUE), ("S", ORANGE)):
                b.text(xs2[0], cv2[k][0][0] + 0.07, f"{cv2[k][0][0]:.2f}", fontsize=6.5, color=col, ha="left", va="center")
                b.text(xs2[-1] + 2, cv2[k][-1][0] - 0.07, f"{cv2[k][-1][0]:.2f}", fontsize=6.5, color=col, ha="left", va="center")
    for ax, t in ((a, f"(a) Correct ({len(cor)})"), (b, f"(b) Incorrect ({len(inc)})")):
        ax.set_xlim(-4, 100); ax.set_ylim(0, 1.02); ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.grid(True, color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("Position in rollout (%)", labelpad=1.5); ax.set_title(t, pad=4)
    a.set_ylabel("Success rate", labelpad=2); b.set_yticklabels([])
    st = [Line2D([0], [0], color="#555555", lw=1.1, ls="-"), Line2D([0], [0], color="#555555", lw=1.1, ls=DASH, alpha=0.7)]
    b.legend(st, [f"teacher solves ({len(sol)})", f"teacher does not ({len(uns)})"], loc="upper right", frameon=False,
             fontsize=6.3, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
    actor_legend(fig, B0, S, H)
    save(fig, out, "fig3")
    return N


# ------------------------------------------------------------------ Figure 4
def fig4(d, out, draw=True):
    R, E1 = load(d)
    inc = sorted(r for r in R if not R[r]["correct"] and solvable(E1, r))
    n = max(len(R[r]["cut_points"]) for r in inc)
    xs = [pos(R, inc, c) for c in range(n)]
    frac = {"correct": [], "original": [], "other": []}
    for c in range(n):
        cnt = collections.Counter()
        for r in inc:
            for s in E1[r].get(("teacher", c), []):
                cnt["correct" if s["correct"] else ("original" if s["matches_original_answer"] else "other")] += 1
        tot = sum(cnt.values())
        for k in frac: frac[k].append(cnt[k] / tot if tot else float("nan"))
    A = {int(k): v for k, v in json.load(open(f"{d}/closing.json"))["A"].items()}
    J = {(x["rid"], x["rel"]): x["labels"] for x in jlg(f"{d}/judge_partial_[0-9]*.jsonl")}
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    grp = [[] for _ in edges[:-1]]
    for (rid, rel), labs in J.items():
        if rid not in A or rid not in R: continue
        labs = [l for l in labs if l is not None]
        if not labs: continue
        v = acc(E1[rid][("teacher", A[rid]["close"] + rel)])
        g = next(i for i in range(len(edges) - 1) if edges[i] <= v < edges[i + 1])
        grp[g].append((v, [l != "NOT_AN_ERROR" for l in labs]))
    real = [boot([np.mean(x[1]) for x in g]) for g in grp]
    vt = [float(np.mean([x[0] for x in g])) if g else float("nan") for g in grp]
    N = {"n_incorrect_teacher_solves": len(inc), "outcome": {k: [round(x, 3) for x in v] for k, v in frac.items()},
         "n_judged_critiques": sum(len(g) for g in grp), "real_by_group": [round(r[0], 3) for r in real],
         "n_by_group": [r[3] for r in real], "vt_by_group": [round(x, 3) for x in vt]}
    if not draw: return N
    W, H = 3.03, 1.74; S, L, G, B0 = 1.06, 0.40, 0.32, 0.36
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([L / W, B0 / H, S / W, S / H]); b = fig.add_axes([(L + S + G) / W, B0 / H, S / W, S / H])
    y0 = np.zeros(n); cols = {"correct": "#4C7FBF", "original": "#D06A6A", "other": "#DADDE1"}
    for k in ("correct", "original", "other"):
        y1 = y0 + np.array(frac[k]); a.fill_between(xs, y0, y1, color=cols[k], lw=0); y0 = y1
    a.text(xs[1], 0.25, "correct", fontsize=6.8, color="white", ha="left")
    a.text(xs[1], 0.93, "other answer", fontsize=6.8, color="#555555", ha="left", va="center")
    a.text(xs[-1] - 1, frac["correct"][-1] + frac["original"][-1] / 2, "student's\nwrong\nanswer", fontsize=6.3,
           color="white", ha="right", va="center", linespacing=0.95)
    a.set_xlim(xs[0], xs[-1]); a.set_ylim(0, 1); a.set_xticks([0, 25, 50, 75]); a.set_yticks([0, 0.5, 1.0])
    a.set_xlabel("Position in rollout (%)", labelpad=1.5); a.set_ylabel("Fraction", labelpad=2)
    a.set_title("(a) Where the teacher ends", pad=4)
    gx = np.arange(len(real)); m = np.array([r[0] for r in real]); lo = np.array([r[1] for r in real]); hi = np.array([r[2] for r in real])
    b.fill_between(gx, lo, hi, color="#1F3A68", alpha=0.12, lw=0)
    b.plot(gx, m, color="#1F3A68", lw=1.4, marker="D", ms=3.4, markeredgecolor="white", markeredgewidth=0.4)
    b.plot(gx, vt, color="#8A929C", lw=1.2, ls=(0, (3, 1.6)), marker="o", ms=3.0)
    b.text(0, 0.97, "flagged errors that are real", fontsize=6.5, color="#1F3A68", ha="left", va="top")
    b.text(len(real) - 1, 0.38, "teacher continues", fontsize=6.5, color="#8A929C", ha="right", va="top")
    b.set_xticks(gx); b.set_xticklabels(["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"])
    b.set_ylim(0, 1); b.set_yticks([0, 0.5, 1.0]); b.yaxis.grid(True, color="#e8e8e8", lw=0.6); b.set_axisbelow(True)
    b.set_xlabel("Teacher success $V^T$", labelpad=1.5); b.set_title("(b) Critique vs. continuation", pad=4)
    save(fig, out, "fig4")
    return N


# ------------------------------------------------------------------ Figure 5(a)
def fig5a(d, out, draw=True):
    def rate(x): return float(np.mean([s["correct"] for s in x["samples"]]))
    R = {r["rollout_id"]: r for r in jl(f"{d}/rollouts.jsonl")}
    MC = {int(k): v for k, v in json.load(open(f"{d}/method_cut.json")).items() if v.get("fb")}
    E6 = collections.defaultdict(dict); E7 = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in jl(f"{d}/graded_e6.jsonl"): E6[(x["actor"], x["cond"])][x["rid"]] = rate(x)
    for x in jl(f"{d}/graded_e7.jsonl"): E7[(x["cond"], x["actor"], x["q"])][x["rid"]].append(rate(x))
    E1 = collections.defaultdict(dict)
    for x in jl(f"{d}/graded_e1.jsonl"): E1[x["rid"]][(x["actor"], x["cut"])] = rate(x)
    BEFORE = [(-100, -75), (-75, -50), (-50, -25), (-25, 0)]
    bx = [float(np.mean(b)) for b in BEFORE]
    before = {"T": [], "S": []}
    for lo, hi in BEFORE:
        per = {"T": [], "S": []}
        for rid, m in MC.items():
            r = R[rid]; kt = m["keep_tok"]; vals = {"T": [], "S": []}
            for c, t in enumerate(r["cut_tokens"]):
                if t >= kt or kt == 0: continue
                p = 100 * (t / kt - 1)
                if lo <= p < hi or (lo == -100 and p == -100):
                    for a_, k in (("teacher", "T"), ("student", "S")):
                        if (a_, c) in E1[rid]: vals[k].append(E1[rid][(a_, c)])
            for k in ("T", "S"):
                if vals[k]: per[k].append(np.mean(vals[k]))
        for k in ("T", "S"): before[k].append(boot(per[k]))
    QS = (0.25, 0.5, 0.75)
    def e7(cond, actor, q): return [float(np.mean(v)) for v in E7[(cond, actor, q)].values()]
    orig = {"T": [boot(list(E6[("teacher", "direct")].values()))] + [boot(e7("tail", "teacher", q)) for q in QS],
            "S": [boot(list(E6[("student", "direct")].values()))] + [boot(e7("tail", "student", q)) for q in QS]}
    rew = {"T": [boot(list(E6[("teacher", "critique")].values()))] + [boot(e7("rw_fb", "teacher", q)) for q in QS],
           "S": [boot(list(E6[("student", "direct")].values()))] + [boot(e7("rw_nofb", "student", q)) for q in QS]}
    kept = np.array([MC[r]["keep_tok"] / rlen(R[r]) for r in MC])
    N = {"n_method_cuts": len(MC), "kept_median": round(float(np.median(kept)), 3),
         "original": {k: [round(s[0], 3) for s in v] for k, v in orig.items()},
         "rewrite": {k: [round(s[0], 3) for s in v] for k, v in rew.items()},
         "n_tail": orig["T"][1][3], "n_rewrite": rew["T"][1][3]}
    if not draw: return N
    W, H = 1.75, 1.86; S, L, B0 = 1.06, 0.40, 0.36
    fig = plt.figure(figsize=(W, H)); a = fig.add_axes([L / W, B0 / H, S / W, S / H])
    ax_after = [0, 25, 50, 75]
    for k, col, mk in (("T", BLUE, "o"), ("S", ORANGE, "s")):
        band(a, bx + [0], before[k] + [orig[k][0]], col, "-", mk)
        band(a, ax_after, orig[k], col, DASH, mk, alpha_line=0.55, z=2)
        band(a, ax_after, rew[k], col, "-", mk, z=4)
    a.axvline(0, color="#666666", lw=0.6, ls=":", zorder=1)
    a.text(-4, 0.97, "cut", fontsize=6.8, color="#666666", ha="right", va="top")
    a.set_xlim(-100, 112); a.set_ylim(0, 1); a.set_xticks([-100, -50, 0, 50, 100]); a.set_xticklabels(["$-$100", "", "0", "", "100"])
    a.set_yticks([0, 0.5, 1.0]); a.yaxis.grid(True, color="#e8e8e8", lw=0.6); a.set_axisbelow(True)
    a.set_xlabel("Position relative to cut (%)", labelpad=1.5); a.set_ylabel("Success rate", labelpad=2)
    a.set_title("Original vs. rewrite", pad=4)
    ends = sorted([(rew["T"][-1][0], BLUE, 1.0), (rew["S"][-1][0], ORANGE, 1.0), (orig["T"][-1][0], BLUE, 0.8), (orig["S"][-1][0], ORANGE, 0.8)], reverse=True)
    ly = [e[0] for e in ends]
    for i in range(1, len(ly)): ly[i] = min(ly[i], ly[i - 1] - 0.085)
    for (v, col, al), y in zip(ends, ly):
        a.text(83, max(y, 0.035), f"{v:.2f}", fontsize=6.5, color=col, alpha=al, va="center", ha="left")
    style = [Line2D([0], [0], color="#555555", lw=1.1, ls="-"), Line2D([0], [0], color="#555555", lw=1.1, ls=DASH, alpha=0.6)]
    hi_ = max(x[0] for x in rew["T"]) > 0.55
    a.legend(style, ["rewrite", "original"], loc="upper right", bbox_to_anchor=(1.02, 0.56 if hi_ else 0.86), frameon=False,
             fontsize=6.5, handlelength=1.8, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.2)
    actor_legend(fig, B0, S, H)
    save(fig, out, "fig5a")
    return N


def run(d, check_only=False):
    out = f"{d}/figs"; os.makedirs(out, exist_ok=True)
    N = {}
    for name, f, need in (("fig3", fig3, "graded_e1.jsonl"), ("fig4", fig4, "closing.json"), ("fig5a", fig5a, "graded_e7.jsonl")):
        if not os.path.exists(f"{d}/{need}"):
            print(f"[{name}] skipped: {need} not found yet"); continue
        N[name] = f(d, out, draw=not check_only)
    json.dump(N, open(f"{out}/numbers.json", "w"), indent=1)
    print(json.dumps(N, indent=1))
    if not check_only: print(f"figures written to {out}")
