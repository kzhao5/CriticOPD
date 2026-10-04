#!/usr/bin/env python3
"""Section 2 诊断实验:汇总全部数字并画图(图 2、图 3 与附录图)。读 diag.py 各阶段的输出。
所有均值先在每条轨迹内平均,再对轨迹平均;置信区间按轨迹 bootstrap(1000 次,2.5% / 97.5%)。
定位一律用 critic 列出的第一个错误去对关闭区间(后面的错误没有蒙特卡洛标签,它们的准确度见 E8)。"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, "/home/kzhao2/Relay-OPD/diag2")
import diag as D   # noqa: E402
M = D.M
OUT = D.OUT
FIG = os.environ.get("DIAG_FIG", f"{OUT}/figs")
RNG = np.random.default_rng(0)
BINS = ["≤−4", "−3", "−2", "−1", "0", "+1", "+2", "+3", "≥+4"]
BLUE, ORANGE, BAND, WIN, POST = "#2B6CB0", "#E8743B", "#B7DCCF", "#FDF3D0", "#EEEEEE"
LIGHT_OR = "#F4B38A"
TAIL, REWRITE = "#4a3aa7", "#169B6B"           # 图 3(c):被丢弃的尾部 / 带反馈的重写(配色已过 CVD 校验)
QS = D.QS


def b_of(r): return 0 if r <= -4 else (8 if r >= 4 else r + 4)


def boot(per_traj, n=1000):
    """per_traj: 每条轨迹一个值(可为 nan)。返回 (均值, 下界, 上界)。"""
    v = np.array([x for x in per_traj if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    bs = v[RNG.integers(0, len(v), size=(n, len(v)))].mean(1)
    return (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def load():
    R = {r["rollout_id"]: r for r in D.jl(f"{OUT}/rollouts.jsonl")}
    C = json.load(open(f"{OUT}/closing.json")); A = {int(k): v for k, v in C["A"].items()}
    E1 = collections.defaultdict(dict)
    for x in D.jl(f"{OUT}/graded_e1.jsonl"): E1[x["rid"]][(x["actor"], x["cut"])] = x["samples"]
    return R, C, A, E1


def vt(E1, rid, c, part="curve"):   # teacher:关闭点判定用 0-7,曲线用 8-15;part="all" 用全部 16 次
    s = E1[rid][("teacher", c)]; s = s[8:16] if part == "curve" else (s[:8] if part == "det" else s)
    return float(np.mean([x["correct"] for x in s]))
def vs(E1, rid, c): return float(np.mean([x["correct"] for x in E1[rid][("student", c)]]))
def rate(x): return float(np.mean([s["correct"] for s in x["samples"]]))


def interval(a): return a["prev_seg"], a["close_seg"] - 1          # 关闭区间(段号,闭区间)
def located(seg, a):
    lo, hi = interval(a); return seg is not None and lo - 1 <= seg <= hi + 1
def deviation(seg, a):
    lo, hi = interval(a); return 0 if lo <= seg <= hi else (seg - hi if seg > hi else seg - lo)


def main():
    os.makedirs(FIG, exist_ok=True)
    R, C, A, E1 = load()
    N = {}
    inc = [rid for rid, r in R.items() if not r["correct"]]; cor = [rid for rid, r in R.items() if r["correct"]]
    N["n_correct"], N["n_incorrect"] = len(cor), len(inc)
    N["class_A"], N["class_B"], N["class_C"] = len(A), len(C["B"]), len(C["C"])
    if os.path.exists(f"{OUT}/raw_capped.json"):
        cap = json.load(open(f"{OUT}/raw_capped.json")); N["raw_capped_share"] = float(np.mean(list(cap.values())))

    # ---------------- E1
    curve = {"T": [[None] * len(A) for _ in BINS], "S": [[None] * len(A) for _ in BINS]}
    win_d, post_d = [], []
    for i, (rid, a) in enumerate(sorted(A.items())):
        per = collections.defaultdict(lambda: {"T": [], "S": []})
        wd, pd_ = [], []
        for c in range(len(R[rid]["cut_points"])):
            r = c - a["close"]; t, s = vt(E1, rid, c), vs(E1, rid, c)
            per[b_of(r)]["T"].append(t); per[b_of(r)]["S"].append(s)
            (wd if r < 0 else pd_).append(t - s)
        for b, d in per.items():
            curve["T"][b][i] = float(np.mean(d["T"])); curve["S"][b][i] = float(np.mean(d["S"]))
        win_d.append(np.mean(wd) if wd else np.nan); post_d.append(np.mean(pd_) if pd_ else np.nan)
    CV = {k: [boot(curve[k][b]) for b in range(len(BINS))] for k in ("T", "S")}
    N["curve"] = {k: [round(x[0], 3) if not np.isnan(x[0]) else None for x in CV[k]] for k in CV}
    N["window_delta"], N["post_delta"] = boot(win_d), boot(post_d)
    wb = [b for b in range(4) if not np.isnan(CV["T"][b][0])]
    N["window_VT_range"] = (CV["T"][wb[0]][0], CV["T"][3][0]); N["window_VS_range"] = (CV["S"][wb[0]][0], CV["S"][3][0])
    post_T = [np.mean([vt(E1, rid, c) for c in range(a["close"], len(R[rid]["cut_points"]))]) for rid, a in A.items()]
    post_S = [np.mean([vs(E1, rid, c) for c in range(a["close"], len(R[rid]["cut_points"]))]) for rid, a in A.items()]
    N["post_VT"], N["post_VS"] = boot(post_T), boot(post_S)
    N["A_earliest_VT"] = boot([vt(E1, rid, 0) for rid in A])
    cT, cS, cD, low = [], [], [], 0
    for rid in cor:
        cs = range(1, len(R[rid]["cut_points"]))
        t = [vt(E1, rid, c, "all") for c in cs]; s = [vs(E1, rid, c) for c in cs]
        if t: cT.append(np.mean(t)); cS.append(np.mean(s)); cD.append(np.mean(np.array(t) - np.array(s)))
        low += any(x < 0.5 for x in t)
    N["correct_VT"], N["correct_VS"], N["correct_delta"] = boot(cT), boot(cS), boot(cD)
    N["correct_ever_VT_below_half"] = low
    N["empty_prefix_delta"] = boot([vt(E1, rid, 0, "all") - vs(E1, rid, 0) for rid in R])

    # ---------------- E2
    has = [rid for rid in A if R[rid]["final_answer"] is not None]
    N["A_without_final_answer"] = len(A) - len(has)
    e2 = {"T": [[None] * len(has) for _ in BINS], "S": [[None] * len(has) for _ in BINS]}
    cats = {k: collections.defaultdict(list) for k in ("T", "S")}
    postx = {"T": [], "S": []}
    for i, rid in enumerate(sorted(has)):
        a = A[rid]; per = collections.defaultdict(lambda: {"T": [], "S": []}); pp = {"T": [], "S": []}
        for c in range(len(R[rid]["cut_points"])):
            r = c - a["close"]
            for k, ss in (("T", E1[rid][("teacher", c)][8:16]), ("S", E1[rid][("student", c)])):
                f = float(np.mean([bool(x["matches_original_answer"]) for x in ss]))
                per[b_of(r)][k].append(f)
                if r >= 0: pp[k].append(f)
                for x in ss:
                    cat = "correct" if x["correct"] else ("original" if x["matches_original_answer"] else
                          ("none" if x["truncated"] or D.final_answer(x["text"]) is None else "other"))
                    cats[k][b_of(r)].append(cat)
        for b, d in per.items():
            for k in ("T", "S"): e2[k][b][i] = float(np.mean(d[k]))
        for k in ("T", "S"): postx[k].append(np.mean(pp[k]) if pp[k] else np.nan)
    E2 = {k: [boot(e2[k][b]) for b in range(len(BINS))] for k in e2}
    N["post_original_answer_T"], N["post_original_answer_S"] = boot(postx["T"]), boot(postx["S"])

    # ---------------- E3
    sc = {}
    for a_ in ("teacher", "student"):
        for x in D.jlg(f"{OUT}/score_{a_}_[0-9]*.jsonl"): sc.setdefault(x["rid"], {})[a_] = np.array(x["lp"])
    tot_inc = sum(len(R[r]["resp_ids"]) for r in inc)
    post_tok = sum(len(R[r]["resp_ids"]) - A[r]["close_tok"] for r in A)
    N["post_token_share_of_incorrect"] = post_tok / tot_inc
    N["post_token_share_incl_C"] = (post_tok + sum(len(R[r]["resp_ids"]) for r in C["C"])) / tot_inc
    raw = D.jlg(f"{OUT}/raw_rollouts_*.jsonl")
    graded_all = [(r["row"], len(r["resp_ids"])) for r in raw]
    rows_correct = json.load(open(f"{OUT}/raw_correct.json")) if os.path.exists(f"{OUT}/raw_correct.json") else None
    if rows_correct:   # 训练 batch 里错误轨迹的 token 占比:同一个 checkpoint 在全部题上的 rollout
        L_inc = sum(l for row, l in graded_all if not rows_correct[str(row)]); L_all = sum(l for _, l in graded_all)
        N["incorrect_token_share_of_batch"] = L_inc / L_all
        N["post_token_share_of_batch"] = N["post_token_share_of_incorrect"] * L_inc / L_all
    wa, pa = [], []
    bins_w, bins_p = collections.defaultdict(list), collections.defaultdict(list)
    for rid, a in A.items():
        if rid not in sc or len(sc[rid]) < 2: continue
        adv = sc[rid]["teacher"] - sc[rid]["student"]; n0 = a["close_tok"]
        if n0 > 0: wa.append(adv[:n0].mean())
        if n0 < len(adv): pa.append(adv[n0:].mean())
        for t in range(len(adv)):
            (bins_w if t < n0 else bins_p)[t // 1024].append(adv[t])
    N["adv_window"], N["adv_post"] = boot(wa), boot(pa)
    diffs, wts = [], []
    for b in sorted(set(bins_w) & set(bins_p)):
        if len(bins_w[b]) >= 200 and len(bins_p[b]) >= 200:
            diffs.append(np.mean(bins_w[b]) - np.mean(bins_p[b])); wts.append(min(len(bins_w[b]), len(bins_p[b])))
    N["adv_position_matched_diff"] = float(np.average(diffs, weights=wts)) if diffs else None
    N["adv_position_bins_used"] = len(diffs)

    # ---------------- E4:截断解答(未写完版提示词,不给参考答案)
    P = D.jl(f"{OUT}/critic_partial.jsonl")
    rel_vals = (-2, -1, 0, 1, 2, 3, 4)
    e4 = {r: collections.defaultdict(list) for r in rel_vals}; e4t = {r: {} for r in rel_vals}
    fa, la, miss, nerr = (collections.defaultdict(list) for _ in range(4)); inval = 0
    for x in P:
        a = A[x["rid"]]; rel = x["rel"]
        if not x["valid"]: inval += 1; continue
        first = x["errs_raw"][0][0] if x["errs_raw"] else None
        good = x["none"] if rel < 0 else located(first, a)
        e4[rel][x["rid"]].append(float(good)); e4t[rel][x["rid"]] = vt(E1, x["rid"], a["close"] + rel)
        nerr[rel].append(len(x["errs_raw"]))
        if rel < 0: fa[x["rid"]].append(float(not x["none"]))
        else: la[x["rid"]].append(float(good)); miss[x["rid"]].append(float(x["none"]))
    pm = lambda d: [np.mean(v) for v in d.values()]
    N["e4_false_alarm"], N["e4_localization"], N["e4_miss"] = boot(pm(fa)), boot(pm(la)), boot(pm(miss))
    N["e4_invalid"] = inval
    N["e4_n_errors_listed"] = {r: float(np.mean(v)) if v else None for r, v in nerr.items()}
    N["e4_truncated"] = int(sum(x["truncated"] for x in P))
    E4 = {r: (boot(pm(e4[r])), boot(list(e4t[r].values())), len(e4[r])) for r in rel_vals}
    N["e4_by_rel"] = {r: {"critic_correct": E4[r][0][0], "teacher_continues": E4[r][1][0], "n": E4[r][2]} for r in rel_vals}

    # ---------------- E5:完整解答(主结果不给参考答案;给参考答案的版本 = 训练用的 critic,放附录)
    pts, dev_first, dev_last = [], [], []
    for name in ("full_nogt", "full_gt"):
        F = D.jl(f"{OUT}/critic_{name}.jsonl"); m, mf, ne, none, inv, dl = [], [], [], 0, 0, []
        for x in F:
            a = A[x["rid"]]
            if not x["valid"]: inv += 1; continue
            if x["none"]: none += 1; m.append(0.0); continue
            first = x["errs_raw"][0][0]; last = x["errs_raw"][-1][0]
            m.append(float(located(first, a))); ne.append(len(x["errs_raw"]))
            mf.append(float(located(x["errs"][0][0], a)) if x["errs"] else 0.0)
            dl.append(deviation(last, a))
            if name == "full_nogt":
                lo, hi = interval(a); pts.append(((lo + hi) / 2, first, lo, hi))
                dev_first.append(deviation(first, a)); dev_last.append(deviation(last, a))
        n_raw = sum(len(x["errs_raw"]) for x in F); n_kept = sum(len(x["errs"]) for x in F)
        N[f"e5_{name}"] = {"match_first": boot(m), "match_first_after_leak_filter": boot(mf), "none": none, "invalid": inv,
                           "n": len(F), "n_errors_listed_mean": float(np.mean(ne)) if ne else None,
                           "n_errors_listed_hist": dict(collections.Counter(ne)),
                           "leak_drop_rate": (n_raw - n_kept) / n_raw if n_raw else None,
                           "truncated": int(sum(x["truncated"] for x in F)),
                           "last_error_deviation_median": float(np.median(dl)) if dl else None,
                           "last_error_after_interval": float(np.mean([d > 1 for d in dl])) if dl else None}

    # ---------------- 方法断点 + E6
    MC = {int(k): v for k, v in json.load(open(f"{OUT}/method_cut.json")).items()}
    use = {rid: m for rid, m in MC.items() if m.get("fb")}
    N["method_cut"] = {"usable": len(use), "excluded": dict(collections.Counter(m["reason"] for m in MC.values() if not m.get("fb"))),
                       "n_err_mean": float(np.mean([m["n_err"] for m in use.values()])) if use else None,
                       "at_or_after_closing_point": float(np.mean([m["keep_seg"] >= A[r]["close_seg"] for r, m in use.items()])) if use else None,
                       "past_last_good_cut": float(np.mean([m["keep_seg"] > A[r]["prev_seg"] for r, m in use.items()])) if use else None}
    G6 = D.jl(f"{OUT}/graded_e6.jsonl")
    cond = collections.defaultdict(dict)
    for x in G6: cond[(x["actor"], x["cond"])][x["rid"]] = rate(x)
    order = [("Teacher", ("teacher", "direct")), ("Student", ("student", "direct")),
             ("Student\n+ generic", ("student", "generic")), ("Student\n+ feedback", ("student", "critique")),
             ("Teacher\n+ feedback", ("teacher", "critique"))]
    B6 = [(lab, boot(list(cond[k].values())), len(cond[k])) for lab, k in order]
    N["e6"] = {lab.replace("\n", " "): (m, n) for lab, m, n in B6}
    def paired(k1, k2):
        both = [r for r in cond[k1] if r in cond[k2]]; return boot([cond[k1][r] - cond[k2][r] for r in both])
    N["e6_feedback_minus_generic"] = paired(("student", "critique"), ("student", "generic"))
    N["e6_feedback_minus_direct_student"] = paired(("student", "critique"), ("student", "direct"))
    N["e6_teacher_feedback_minus_direct"] = paired(("teacher", "critique"), ("teacher", "direct"))
    N["e6_teacher_direct_below_half"] = float(np.mean([v < 0.5 for v in cond[("teacher", "direct")].values()])) if cond[("teacher", "direct")] else None

    # ---------------- E7:可教性差 Δ —— 被丢弃的尾部(都不带反馈) vs 带反馈的重写(训练条件:teacher 带、学生不带)
    G7 = D.jl(f"{OUT}/graded_e7.jsonl")
    v7 = collections.defaultdict(lambda: collections.defaultdict(list))     # (cond, actor, q) -> rid -> [各重写的成功率]
    for x in G7: v7[(x["cond"], x["actor"], x["q"])][x["rid"]].append(rate(x))
    def per(cnd, act, q): return {r: float(np.mean(v)) for r, v in v7[(cnd, act, q)].items()}
    def dlt(t, s): return boot([t[r] - s[r] for r in t if r in s])
    E7 = {}
    for q in QS:
        tT, tS = per("tail", "teacher", q), per("tail", "student", q)
        fT, fS = per("rw_fb", "teacher", q), per("rw_fb", "student", q)
        nT, nS = per("rw_nofb", "teacher", q), per("rw_nofb", "student", q)
        E7[q] = {"tail": dlt(tT, tS), "rewrite_train": dlt(fT, nS), "rewrite_both_fb": dlt(fT, fS), "rewrite_no_fb": dlt(nT, nS),
                 "VT_tail": boot(list(tT.values())), "VS_tail": boot(list(tS.values())),
                 "VT_rw_fb": boot(list(fT.values())), "VS_rw_fb": boot(list(fS.values())),
                 "VT_rw_nofb": boot(list(nT.values())), "VS_rw_nofb": boot(list(nS.values())),
                 "n_tail": len(tT), "n_rewrite": len(fT)}
    N["e7"] = {f"{q}": {k: (v[0] if isinstance(v, tuple) else v) for k, v in d.items()} for q, d in E7.items()}

    # ---------------- E8:逐条判断列出的错误是否属实(Qwen3-8B 思考模式)
    if os.path.exists(f"{OUT}/judge.jsonl"):
        J = D.jl(f"{OUT}/judge.jsonl"); e8 = {}
        for kind in ("full_nogt", "full_gt"):
            rows = [x for x in J if x["kind"] == kind]
            pos = collections.defaultdict(collections.Counter)
            for x in rows:
                for i, l in enumerate(x["labels"]): pos[min(i + 1, 3)][l or "UNPARSED"] += 1
            allc = sum((pos[p] for p in pos), collections.Counter())
            n = sum(allc.values())
            parsed = n - allc["UNPARSED"]
            e8[kind] = {"n_rollouts": len(rows), "n_errors": n,
                        "genuine_rate": (parsed - allc["NOT_AN_ERROR"]) / parsed if parsed else None,
                        "by_position": {("1" if p == 1 else ("2" if p == 2 else "3+")): dict(c) for p, c in sorted(pos.items())}}
        N["e8"] = e8

    # ---------------- 附加分析(为 2.2 的叙事决定准备)
    def after_cut(r, seg):          # 该段之后最近的 E1 截断点
        return min([c for c in range(len(r["cut_points"])) if r["cut_points"][c] >= seg + 1] or [len(r["cut_points"]) - 1])
    def grp(seg, a):
        lo, hi = interval(a); return "before" if seg < lo - 1 else ("in" if seg <= hi + 1 else "after")
    # (1) E5 × E8:列出的每条错误相对关闭区间的位置,判定是否属实,该处之后 teacher 续写成功率
    if os.path.exists(f"{OUT}/judge.jsonl"):
        JL = {(x["kind"], x["rid"]): x["labels"] for x in D.jl(f"{OUT}/judge.jsonl")}
        LOC = {}
        for kind in ("full_nogt", "full_gt"):
            t = {g: {"all": collections.Counter(), "first": collections.Counter(), "vt_first": []} for g in ("before", "in", "after")}
            for x in D.jl(f"{OUT}/critic_{kind}.jsonl"):
                rid = x["rid"]; a = A[rid]; labs = JL.get((kind, rid), [None] * len(x["errs_raw"]))
                for i, (e, l) in enumerate(zip(x["errs_raw"], labs)):
                    g = grp(e[0], a); t[g]["all"][l or "UNPARSED"] += 1
                    if i == 0:
                        t[g]["first"][l or "UNPARSED"] += 1; t[g]["vt_first"].append(vt(E1, rid, after_cut(R[rid], e[0]), "all"))
            def gen(c): n = sum(c.values()) - c["UNPARSED"]; return (n - c["NOT_AN_ERROR"]) / n if n else None
            LOC[kind] = {g: {"n_errors": sum(v["all"].values()), "genuine": gen(v["all"]), "n_first": sum(v["first"].values()),
                             "first_genuine": gen(v["first"]), "VT_after_first": float(np.mean(v["vt_first"])) if v["vt_first"] else None,
                             "labels": dict(v["all"])} for g, v in t.items()}
        N["e5_by_position"] = LOC
    # (2) 对照:附录 C 原来的单错误提示词(找不可挽回的那一步),不给参考答案
    if os.path.exists(f"{OUT}/critic_single_full.jsonl"):
        F = D.jl(f"{OUT}/critic_single_full.jsonl")
        N["single_e5_match"] = boot([float(bool(x["errs_raw"]) and located(x["errs_raw"][0][0], A[x["rid"]])) for x in F])
        N["single_e5_position"] = dict(collections.Counter(grp(x["errs_raw"][0][0], A[x["rid"]]) if x["errs_raw"] else ("none" if x["none"] else "invalid") for x in F))
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for x in D.jl(f"{OUT}/critic_single_partial.jsonl"):
            f = x["errs_raw"][0][0] if x["errs_raw"] else None
            by[x["rel"]][x["rid"]].append(float(x["none"] if x["rel"] < 0 else (f is not None and located(f, A[x["rid"]]))))
        N["single_e4_by_rel"] = {r: boot([np.mean(v) for v in by[r].values()])[0] for r in sorted(by)}
    # (3) 截断解答的批改,逐条判定:按相对关闭点的位置,批改列出错误的比例、列出的错误属实的比例
    JP = None
    if os.path.exists(f"{OUT}/judge_partial.jsonl"): JP = D.jl(f"{OUT}/judge_partial.jsonl")
    elif D.glob.glob(f"{OUT}/judge_partial_[0-9]*.jsonl"): JP = D.jlg(f"{OUT}/judge_partial_[0-9]*.jsonl")
    E4B = None
    if JP is not None:
        JPd = {(x["rid"], x["rel"]): x["labels"] for x in JP}
        flag, prec, first = (collections.defaultdict(dict) for _ in range(3))
        for x in P:
            rel, rid = x["rel"], x["rid"]
            if not x["valid"]: continue
            flag[rel][rid] = float(bool(x["errs_raw"]))
            labs = [l for l in JPd.get((rid, rel), []) if l]
            if labs:
                prec[rel][rid] = float(np.mean([l != "NOT_AN_ERROR" for l in labs])); first[rel][rid] = float(labs[0] != "NOT_AN_ERROR")
        E4B = {r: {"flag": boot(list(flag[r].values())), "precision": boot(list(prec[r].values())), "first_genuine": boot(list(first[r].values())),
                   "VT": E4[r][1]} for r in rel_vals}
        N["e4_judged_by_rel"] = {r: {k: (v[0] if isinstance(v, tuple) else v) for k, v in d.items()} for r, d in E4B.items()}
    # (4) E6 按方法断点相对关闭点的位置拆分
    split = {}
    for name, sel in (("cut_at_or_after_closing", lambda r: use[r]["keep_seg"] >= A[r]["close_seg"]),
                      ("cut_before_closing", lambda r: use[r]["keep_seg"] < A[r]["close_seg"])):
        rs = [r for r in use if sel(r)]
        d = {lab.replace("\n", " "): boot([cond[k][r] for r in rs if r in cond[k]]) for lab, k in order}
        d["n"] = len(rs)
        d["teacher_feedback_minus_direct"] = boot([cond[("teacher", "critique")][r] - cond[("teacher", "direct")][r] for r in rs if r in cond[("teacher", "critique")] and r in cond[("teacher", "direct")]])
        d["student_feedback_minus_generic"] = boot([cond[("student", "critique")][r] - cond[("student", "generic")][r] for r in rs if r in cond[("student", "critique")] and r in cond[("student", "generic")]])
        split[name] = d
    N["e6_split"] = split
    # (5) 问题 1:崩溃区间 = teacher 和学生都低。按状态统计窗口内与关闭点之后两者都低的比例
    st = {"window": [], "post": []}
    for rid, a in A.items():
        for c in range(len(R[rid]["cut_points"])):
            st["window" if c < a["close"] else "post"].append((vt(E1, rid, c), vs(E1, rid, c)))
    N["q1_states"] = {k: {"n": len(v), "both_below_0.5": float(np.mean([t < 0.5 and s_ < 0.5 for t, s_ in v])),
                          "both_below_0.25": float(np.mean([t < 0.25 and s_ < 0.25 for t, s_ in v])),
                          "VT_median": float(np.median([t for t, _ in v])), "VS_median": float(np.median([s_ for _, s_ in v]))} for k, v in st.items()}
    # (6) E7 按轨迹配对:重写(训练条件)相对被丢弃尾部,teacher 与学生成功率的提升,以及两者都低的比例
    e7p = {}
    for q in QS:
        tT, tS, fT, nS = per("tail", "teacher", q), per("tail", "student", q), per("rw_fb", "teacher", q), per("rw_nofb", "student", q)
        rs = [r for r in tT if r in fT and r in nS and r in tS]
        e7p[str(q)] = {"n": len(rs), "VT_rewrite_minus_tail": boot([fT[r] - tT[r] for r in rs]), "VS_rewrite_minus_tail": boot([nS[r] - tS[r] for r in rs]),
                       "both_low_tail": float(np.mean([tT[r] < 0.5 and tS[r] < 0.5 for r in rs])) if rs else None,
                       "both_low_rewrite": float(np.mean([fT[r] < 0.5 and nS[r] < 0.5 for r in rs])) if rs else None}
    N["e7_paired"] = e7p
    # (7) E10:以错误为锚,错误段前后各截一刀(16 次续写),判断崩溃从哪里开始
    if os.path.exists(f"{OUT}/graded_e10.jsonl") and os.path.exists(f"{OUT}/e10_anchors.json"):
        G10 = collections.defaultdict(dict)
        for x in D.jl(f"{OUT}/graded_e10.jsonl"): G10[x["rid"]][(x["actor"], x["ntok"])] = rate(x)
        AN = {int(k): v for k, v in json.load(open(f"{OUT}/e10_anchors.json")).items()}
        e10 = {}
        for name in ("judge_first", "critic_first", "critic_last"):
            rows = []
            for rid, an in AN.items():
                if name not in an: continue
                b, f = an[name]["before"], an[name]["after"]; g = G10.get(rid, {})
                if ("teacher", b) not in g or ("teacher", f) not in g: continue
                vb, va = g[("teacher", b)], g[("teacher", f)]
                rows.append({"VT_before": vb, "VT_after": va, "VS_before": g.get(("student", b)), "VS_after": g.get(("student", f)),
                             "cls": "precede" if vb < 0.5 else ("coincide" if va < 0.5 else "contain"),
                             "seg_vs_interval": grp(an[name]["seg"], A[rid])})
            if not rows: continue
            cnt = collections.Counter(r["cls"] for r in rows)
            e10[name] = {"n": len(rows), "classes": {k: cnt[k] / len(rows) for k in ("contain", "coincide", "precede")},
                         "VT_before": boot([r["VT_before"] for r in rows]), "VT_after": boot([r["VT_after"] for r in rows]),
                         "VS_before": boot([r["VS_before"] for r in rows if r["VS_before"] is not None]),
                         "VS_after": boot([r["VS_after"] for r in rows if r["VS_after"] is not None]),
                         "VT_drop": boot([r["VT_before"] - r["VT_after"] for r in rows]),
                         "position_vs_closing_interval": dict(collections.Counter(r["seg_vs_interval"] for r in rows))}
        if os.path.exists(f"{OUT}/judge_first.jsonl") or D.glob.glob(f"{OUT}/judge_first_[0-9]*.jsonl"):
            JF = D.jlg(f"{OUT}/judge_first_[0-9]*.jsonl") or D.jl(f"{OUT}/judge_first.jsonl")
            e10["judge_first_found"] = {"n": len(JF), "with_first": sum(x["first"] is not None for x in JF), "none": sum(x["none"] for x in JF)}
        N["e10"] = e10

    json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    plot_fig2(CV, N, E2)
    plot_fig3(E4, B6, E7)
    if E4B: plot_fig3_alt(E4B, B6, E7)
    plot_appendix(cats, E7, pts, dev_first, dev_last, N)
    print(json.dumps(N, indent=1, default=str)[:8000])


# ---------------------------------------------------------------- 画图
def style():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    return plt


def regions(ax, xs, close_x):
    ax.axvspan(xs[0] - 0.5, close_x - 0.5, color=WIN, zorder=0, lw=0)
    ax.axvspan(close_x - 0.5, xs[-1] + 0.5, color=POST, zorder=0, lw=0)
    ax.axvline(close_x, color="#555555", ls=":", lw=1, zorder=1)


def line(ax, xs, stats, color, label, marker="o", ls="-"):
    m = np.array([s[0] for s in stats]); lo = np.array([s[1] for s in stats]); hi = np.array([s[2] for s in stats])
    ok = ~np.isnan(m)
    ax.fill_between(np.array(xs)[ok], lo[ok], hi[ok], color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(np.array(xs)[ok], m[ok], color=color, lw=2, ls=ls, marker=marker, ms=4.5, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", dpi=220, bbox_inches="tight")


def plot_fig2(CV, N, E2):
    plt = style()
    xs = list(range(len(BINS)))
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"wspace": 0.28})
    a = ax[0]; regions(a, xs, 4)
    ct, cs = N["correct_VT"][0], N["correct_VS"][0]
    a.axhspan(min(ct, cs), max(ct, cs) if abs(ct - cs) > 0.01 else min(ct, cs) + 0.01, color=BAND, alpha=0.9, zorder=1, lw=0)
    a.text(8.4, (ct + cs) / 2, "correct rollouts", fontsize=7.5, color="#2f6b5a", ha="right", va="center")
    line(a, xs, CV["T"], BLUE, "Teacher"); line(a, xs, CV["S"], ORANGE, "Student", marker="s")
    a.text(1.5, 0.06, f"Δ = {N['window_delta'][0]:+.2f}", ha="center", fontsize=8)
    a.text(6.0, 0.62, f"Δ = {N['post_delta'][0]:+.2f}", ha="center", fontsize=8)
    a.set_ylabel("Success rate from the state"); a.set_title("(a) Success rate", loc="left")
    b = ax[1]; regions(b, xs, 4)
    line(b, xs, E2["T"], BLUE, "Teacher"); line(b, xs, E2["S"], ORANGE, "Student", marker="s")
    b.set_ylabel("Fraction ending with the\noriginal incorrect answer"); b.set_title("(b) Original incorrect answer", loc="left")
    for x_ in ax:
        x_.set_ylim(0, 1); x_.set_xlim(-0.5, 8.5); x_.set_xticks(xs); x_.set_xticklabels(BINS)
        x_.set_xlabel("Cut point relative to the closing point")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig_window"); plt.close(fig)


def plot_fig3(E4, B6, E7):
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.45), gridspec_kw={"wspace": 0.45, "width_ratios": [1.1, 1.15, 0.95]})
    # (a) 同一批状态:teacher 续写成功率降,critic 判断正确率升
    a = ax[0]; rel = [-2, -1, 0, 1, 2, 3, 4]
    a.axvspan(-2.5, -0.5, color=WIN, zorder=0, lw=0); a.axvspan(-0.5, 4.5, color=POST, zorder=0, lw=0)
    a.axvline(0, color="#555555", ls=":", lw=1)
    line(a, rel, [E4[r][1] for r in rel], BLUE, "Teacher continues", marker="o", ls="--")
    line(a, rel, [E4[r][0] for r in rel], "#1B3F73", "Critique correct", marker="D")
    a.set_ylim(0, 1); a.set_xlim(-2.5, 4.5); a.set_xticks(rel); a.set_xticklabels([f"{r:+d}".replace("-", "−") if r else "0" for r in rel])
    a.set_xlabel("Cut point relative to the closing point"); a.set_ylabel("Rate"); a.set_title("(a) Truncated solutions", loc="left")
    # (b) 方法断点处的五种续写方式
    c = ax[1]
    cols = [BLUE, ORANGE, LIGHT_OR, ORANGE, BLUE]; hat = [None, None, None, "///", "///"]
    for i, (lab, (m, lo, hi_), n) in enumerate(B6):
        c.bar(i, m, color=cols[i], width=0.68, hatch=hat[i], edgecolor="white", linewidth=0.6)
        c.errorbar(i, m, yerr=[[m - lo], [hi_ - m]], color="#333333", lw=0.9, capsize=2)
        c.text(i, hi_ + 0.02, f"{m:.2f}", ha="center", fontsize=7)
    c.set_xticks(range(len(B6))); c.set_xticklabels([b_[0].replace("\n", " ") for b_ in B6], fontsize=6.8, rotation=35, ha="right", rotation_mode="anchor")
    c.set_ylim(0, 1); c.set_ylabel("Success rate from the state"); c.set_title("(b) At the cut point", loc="left")
    # (c) Δ:被丢弃的尾部 vs 带反馈的重写
    d = ax[2]; qs = [25, 50, 75]
    d.axhline(0, color="#555555", lw=0.8)
    line(d, qs, [E7[q / 100]["tail"] for q in qs], TAIL, "Discarded tail", marker="s", ls="--")
    line(d, qs, [E7[q / 100]["rewrite_train"] for q in qs], REWRITE, "Rewrite", marker="o")
    d.set_xticks(qs); d.set_xticklabels([f"{q}%" for q in qs]); d.set_xlim(15, 85)
    d.set_xlabel("Position after the cut point"); d.set_ylabel("Δ = V$^T$ − V$^S$"); d.set_title("(c) Teachability gap", loc="left")
    h = a.get_legend_handles_labels(); h2 = d.get_legend_handles_labels()
    fig.legend(h[0] + h2[0], h[1] + h2[1], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7.5, handlelength=2.2)
    save(fig, "fig_critique"); plt.close(fig)


def plot_fig3_alt(E4B, B6, E7):
    """图 3 的备选版本:(a) 换成「teacher 续写成功率」与「批改列出的错误中属实的比例」(Qwen3-8B 逐条判定)。"""
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.45), gridspec_kw={"wspace": 0.45, "width_ratios": [1.1, 1.15, 0.95]})
    a = ax[0]; rel = [-2, -1, 0, 1, 2, 3, 4]
    a.axvspan(-2.5, -0.5, color=WIN, zorder=0, lw=0); a.axvspan(-0.5, 4.5, color=POST, zorder=0, lw=0)
    a.axvline(0, color="#555555", ls=":", lw=1)
    line(a, rel, [E4B[r]["VT"] for r in rel], BLUE, "Teacher continues", marker="o")
    line(a, rel, [E4B[r]["precision"] for r in rel], "#1B3F73", "Flagged errors that are real", marker="D")
    a.set_ylim(0, 1); a.set_xlim(-2.5, 4.5); a.set_xticks(rel); a.set_xticklabels([f"{r:+d}".replace("-", "−") if r else "0" for r in rel])
    a.set_xlabel("Cut point relative to the closing point"); a.set_ylabel("Rate"); a.set_title("(a) Truncated solutions", loc="left")
    c = ax[1]
    cols = [BLUE, ORANGE, LIGHT_OR, ORANGE, BLUE]; hat = [None, None, None, "///", "///"]
    for i, (lab, (m, lo, hi_), n) in enumerate(B6):
        c.bar(i, m, color=cols[i], width=0.68, hatch=hat[i], edgecolor="white", linewidth=0.6)
        c.errorbar(i, m, yerr=[[m - lo], [hi_ - m]], color="#333333", lw=0.9, capsize=2)
        c.text(i, hi_ + 0.02, f"{m:.2f}", ha="center", fontsize=7)
    c.set_xticks(range(len(B6))); c.set_xticklabels([b_[0].replace("\n", " ") for b_ in B6], fontsize=6.8, rotation=35, ha="right", rotation_mode="anchor")
    c.set_ylim(0, 1); c.set_ylabel("Success rate from the state"); c.set_title("(b) At the cut point", loc="left")
    d = ax[2]; qs = [25, 50, 75]                 # (c) 两者的绝对水平:尾部(虚线)vs 重写(实线,训练条件)
    for key, col, mk, lab in (("VT_tail", BLUE, "o", "Teacher, discarded tail"), ("VS_tail", ORANGE, "s", "Student, discarded tail")):
        line(d, qs, [E7[q / 100][key] for q in qs], col, lab, marker=mk, ls="--")
    for key, col, mk, lab in (("VT_rw_fb", BLUE, "o", "Teacher, rewrite"), ("VS_rw_nofb", ORANGE, "s", "Student, rewrite")):
        line(d, qs, [E7[q / 100][key] for q in qs], col, lab, marker=mk)
    d.set_xticks(qs); d.set_xticklabels([f"{q}%" for q in qs]); d.set_xlim(15, 85); d.set_ylim(0, 1)
    d.set_xlabel("Position after the cut point"); d.set_ylabel("Success rate from the state"); d.set_title("(c) Tail vs. rewrite", loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=6.6, handlelength=1.8)
    d.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=2.0, labelspacing=0.3)
    save(fig, "fig_critique_alt"); plt.close(fig)


def plot_appendix(cats, E7, pts, dev_first, dev_last, N):
    plt = style()
    # 续写结果的构成
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True)
    order = [("correct", "#4C9A6A", "Correct"), ("original", "#C0504D", "Original incorrect answer"),
             ("other", "#9A9A9A", "Other incorrect answer"), ("none", "#D9D9D9", "No answer")]
    for a, k, t in ((ax[0], "T", "Teacher"), (ax[1], "S", "Student")):
        bottom = np.zeros(len(BINS))
        for key, col, lab in order:
            v = np.array([np.mean([c == key for c in cats[k][b]]) if cats[k][b] else 0 for b in range(len(BINS))])
            a.bar(range(len(BINS)), v, bottom=bottom, color=col, width=0.8, edgecolor="white", linewidth=0.6, label=lab)
            bottom += v
        a.set_xticks(range(len(BINS))); a.set_xticklabels(BINS); a.set_title(t, loc="left"); a.set_ylim(0, 1)
        a.set_xlabel("Cut point relative to the closing point")
    ax[0].set_ylabel("Fraction of continuations")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    save(fig, "fig_app_outcomes"); plt.close(fig)
    # E5:第一条错误 vs 关闭区间(散点 + 偏差直方图;最后一条错误的偏差同画)
    if pts:
        fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw={"wspace": 0.3})
        b = ax[0]
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts], dtype=float)
        jit = RNG.uniform(-0.15, 0.15, size=len(x)); hi = max(x.max(), y.max()) + 2
        b.fill_between([0, hi], [-1, hi - 1], [1, hi + 1], color=BAND, alpha=0.7, lw=0)
        b.plot([0, hi], [0, hi], color="#555555", lw=0.8)
        b.scatter(x + jit, y + jit, s=10, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
        b.set_xlim(0, hi); b.set_ylim(0, hi)
        b.set_xlabel("Closing interval (paragraph)"); b.set_ylabel("First flagged paragraph")
        b.text(0.04, 0.96, f"within ±1: {100 * N['e5_full_nogt']['match_first'][0]:.0f}%", transform=b.transAxes, va="top", fontsize=7.5)
        h = ax[1]; bins = np.arange(-10.5, 11.5, 1)
        h.hist(np.clip(dev_first, -10, 10), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5, label="First error")
        h.hist(np.clip(dev_last, -10, 10), bins=bins, histtype="step", color=ORANGE, lw=1.4, label="Last error")
        h.set_xlabel("Flagged paragraph minus closing interval"); h.set_ylabel("Rollouts"); h.legend(frameon=False, fontsize=7)
        save(fig, "fig_app_localization"); plt.close(fig)
    # E7 分项:V^T 与 V^S 在各条件下
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.2), sharey=True, gridspec_kw={"wspace": 0.12}); qs = [25, 50, 75]
    for a, (title, kt, ks) in zip(ax, (("Discarded tail", "VT_tail", "VS_tail"), ("Rewrite, both see feedback", "VT_rw_fb", "VS_rw_fb"),
                                       ("Rewrite, no feedback", "VT_rw_nofb", "VS_rw_nofb"))):
        line(a, qs, [E7[q / 100][kt] for q in qs], BLUE, "Teacher"); line(a, qs, [E7[q / 100][ks] for q in qs], ORANGE, "Student", marker="s")
        a.set_title(title, loc="left", fontsize=8); a.set_xticks(qs); a.set_xticklabels([f"{q}%" for q in qs]); a.set_ylim(0, 1); a.set_xlim(15, 85)
        a.set_xlabel("Position after the cut point")
    ax[0].set_ylabel("Success rate from the state")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig_app_rewrite"); plt.close(fig)


if __name__ == "__main__":
    main()
