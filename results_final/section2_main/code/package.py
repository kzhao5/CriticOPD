#!/usr/bin/env python3
"""把 Section 2 的结果打包到 results_final/section2/:图、numbers.json、中文结果汇总、精简数据、代码快照,再打成 tar.gz。
用法: DIAG_OUT=<out 或 out_all> python3 package.py [目标目录名]"""
import glob, json, os, shutil, subprocess, sys
import numpy as np

SRC = os.environ.get("DIAG_OUT", "/home/kzhao2/Relay-OPD/diag2/out")
NAME = sys.argv[1] if len(sys.argv) > 1 else "section2"
DST = f"/home/kzhao2/Relay-OPD/results_final/{NAME}"
HERE = "/home/kzhao2/Relay-OPD/diag2"
for d in ("figs", "data", "code"): os.makedirs(f"{DST}/{d}", exist_ok=True)


def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def jw(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- 图与数字
for f in glob.glob(f"{SRC}/figs/*"): shutil.copy(f, f"{DST}/figs/")
shutil.copy(f"{SRC}/numbers.json", DST)
N = json.load(open(f"{SRC}/numbers.json"))

# ---- 精简数据(不含续写文本)
R = jl(f"{SRC}/rollouts.jsonl")
jw(f"{DST}/data/rollouts.jsonl", [{k: r[k] for k in ("rollout_id", "problem_id", "problem", "reference_answer", "response", "final_answer",
                                                    "correct", "capped", "cut_points", "cut_tokens")} | {"n_tokens": len(r["resp_ids"]),
                                                    "n_segments": len(r["segments"])} for r in R])
def rates(job, extra=()):
    out = []
    for x in jl(f"{SRC}/graded_{job}.jsonl"):
        row = {k: x.get(k) for k in ("rid", "actor", "cut", "cond", "ntok", "q", "rw", "nrw") if k in x}
        row["n"] = len(x["samples"]); row["success"] = float(np.mean([s["correct"] for s in x["samples"]]))
        row["truncated"] = float(np.mean([s["truncated"] for s in x["samples"]]))
        if "matches_original_answer" in x["samples"][0]:
            row["original_answer"] = float(np.mean([bool(s["matches_original_answer"]) for s in x["samples"]]))
        out.append(row)
    jw(f"{DST}/data/{job}_rates.jsonl", out)
for job in ("e1", "e6", "e7"): rates(job)
for f in ["closing.json", "method_cut.json"] + [os.path.basename(p) for p in glob.glob(f"{SRC}/critic_*.jsonl") if not p.split("_")[-1].split(".")[0].isdigit()] \
         + ["judge.jsonl", "judge_partial.jsonl"]:
    if os.path.exists(f"{SRC}/{f}"): shutil.copy(f"{SRC}/{f}", f"{DST}/data/")
if not os.path.exists(f"{SRC}/judge_partial.jsonl"):
    jw(f"{DST}/data/judge_partial.jsonl", [x for p in sorted(glob.glob(f"{SRC}/judge_partial_[0-9]*.jsonl")) for x in jl(p)])

# ---- 代码快照
for f in ["diag.py", "report.py", "merge_out.py", "analyze_loc.py", "package.py", "run_gpu.sl", "pipeline.sh", "pipeline_ext.sh", "pipeline_ext2.sh"]:
    if os.path.exists(f"{HERE}/{f}"): shutil.copy(f"{HERE}/{f}", f"{DST}/code/")
shutil.copy("/home/kzhao2/Relay-OPD/relay-opd/verl/experimental/agent_loop/critic_opd_agent_loop.py", f"{DST}/code/")


# ---- 中文汇总
def ci(v, pct=False, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if isinstance(v, str): v = json.loads(v.replace("(", "[").replace(")", "]").replace("nan", "NaN"))
    if isinstance(v, (int, float)): v = (v, None, None)
    m, lo, hi = v
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:+.3f}" if sign else f"{x:.3f}")
    return f(m) if lo is None or (isinstance(lo, float) and np.isnan(lo)) else f"{f(m)} [{f(lo)}, {f(hi)}]"


L = []
w = L.append
w(f"# Section 2 诊断实验结果(设置 B)\n")
w(f"数据目录:`{SRC}`。学生 = opd_1p7b@80(Qwen3-1.7B 非思考),teacher = Qwen3-4B-Instruct-2507,题目 = DAPO-Math-17k。")
w("所有均值先在轨迹内平均再对轨迹平均;方括号为按轨迹 bootstrap 的 95% 置信区间。\n")
w("## 样本\n")
w(f"- 正确轨迹 {N['n_correct']} 条,错误轨迹(写完的){N['n_incorrect']} 条;写到长度上限的轨迹占全部 rollout 的 {ci(N.get('raw_capped_share'), pct=True)},不进入。")
w(f"- 错误轨迹分类:A(有关闭点){N['class_A']},B(teacher 始终 ≥0.5){N['class_B']},C(空前缀即 <0.5){N['class_C']}。\n")
w("## 2.1 教学窗口(图 2)\n")
w("| 指标 | 数值 |\n|---|---|")
for k, lab in (("correct_delta", "正确轨迹前缀上的 Δ"), ("correct_VT", "正确轨迹 V^T"), ("correct_VS", "正确轨迹 V^S"),
               ("window_delta", "关闭点之前(窗口)的 Δ"), ("post_delta", "关闭点之后的 Δ"), ("post_VT", "关闭点之后 V^T"),
               ("post_VS", "关闭点之后 V^S"), ("A_earliest_VT", "A 类空前缀 V^T"), ("empty_prefix_delta", "全部轨迹空前缀 Δ"),
               ("post_original_answer_T", "关闭点之后 teacher 续写落回原错误答案"), ("post_original_answer_S", "关闭点之后学生续写落回原错误答案")):
    w(f"| {lab} | {ci(N.get(k), sign='delta' in k)} |")
w(f"| 正确轨迹中曾出现 V^T<0.5 的条数 | {N.get('correct_ever_VT_below_half')} |")
if "q1_states" in N:
    w("\n**问题 1:崩溃区间 = teacher 和学生都低**(按状态统计):\n")
    w("| 区域 | 状态数 | 两者都 <0.5 | 两者都 <0.25 | V^T 中位数 | V^S 中位数 |\n|---|---|---|---|---|---|")
    for k, lab in (("window", "窗口内(关闭点之前)"), ("post", "关闭点之后")):
        d = N["q1_states"][k]
        w(f"| {lab} | {d['n']} | {d['both_below_0.5']:.0%} | {d['both_below_0.25']:.0%} | {d['VT_median']:.2f} | {d['VS_median']:.2f} |")
w("\nE3(token 与 advantage):\n")
w("| 指标 | 数值 |\n|---|---|")
for k, lab in (("post_token_share_of_incorrect", "错误轨迹中关闭点之后的 token 占比"), ("incorrect_token_share_of_batch", "训练 batch 中错误轨迹的 token 占比"),
               ("post_token_share_of_batch", "训练 batch 中关闭点之后的 token 占比"), ("adv_window", "窗口内每 token advantage 均值"),
               ("adv_post", "关闭点之后每 token advantage 均值"), ("adv_position_matched_diff", "位置匹配后 窗口 − 之后")):
    v = N.get(k)
    w(f"| {lab} | {ci(v, pct='share' in k, sign='adv' in k)} |")
w("\n## 2.2 批改能力(图 3)\n")
w("critic = 最终方法的多错误提示词,不给参考答案(给参考答案的版本见 E5 GT 行);定位只看列出的第一条错误是否落在关闭区间 ±1 段。\n")
w("**E4 截断解答**(关闭点之前以 NONE 为正确,之后以第一条错误落在关闭区间 ±1 为正确):\n")
w("| 相对关闭点 | critic 判断正确 | teacher 续写成功率 | n |\n|---|---|---|---|")
for r, d in N["e4_by_rel"].items():
    w(f"| {r} | {ci(d['critic_correct'])} | {ci(d['teacher_continues'])} | {d['n']} |")
w(f"\n误报率(关闭点之前列出错误){ci(N['e4_false_alarm'], pct=True)},定位率(之后){ci(N['e4_localization'], pct=True)},漏报(之后回答 NONE){ci(N['e4_miss'], pct=True)}。\n")
if "e4_judged_by_rel" in N:
    w("**E4 + 逐条判定**(Qwen3-8B 思考模式,给参考答案;图 3(a) 备选版本用的就是这组):\n")
    w("| 相对关闭点 | 批改列出至少一条错误 | 列出的错误属实比例 | 第一条属实 | teacher 续写成功率 |\n|---|---|---|---|---|")
    for r, d in N["e4_judged_by_rel"].items():
        w(f"| {r} | {ci(d['flag'], pct=True)} | {ci(d['precision'], pct=True)} | {ci(d['first_genuine'], pct=True)} | {ci(d['VT'])} |")
w("\n**E5 完整解答**:\n")
w("| 版本 | 第一条错误在关闭区间±1 | 平均列出错误数 | 泄露过滤去掉 | 输出被截断 | 最后一条错误在区间之后 |\n|---|---|---|---|---|---|")
for k, lab in (("e5_full_nogt", "不给参考答案(主结果)"), ("e5_full_gt", "给参考答案(=训练用 critic)")):
    d = N[k]
    w(f"| {lab} | {ci(d['match_first'], pct=True)} | {d['n_errors_listed_mean']:.2f} | {ci(d['leak_drop_rate'], pct=True)} | {d['truncated']}/{d['n']} | {ci(d['last_error_after_interval'], pct=True)} |")
if "single_e5_match" in N:
    w(f"| 对照:附录 C 原单错误提示词 | {ci(N['single_e5_match'], pct=True)} | 1 | — | — | — |")
    w(f"\n单错误提示词第一条错误的位置分布 {N['single_e5_position']};E4 按相对位置 {json.dumps({k: round(v, 2) for k, v in N['single_e4_by_rel'].items()})}。\n")
if "e5_by_position" in N:
    w("**E5 × E8**:列出的错误按相对关闭区间的位置分组,是否属实,以及第一条错误之后 teacher 续写成功率:\n")
    w("| 版本 | 位置 | 错误条数 | 属实 | 第一条错误条数 | 第一条属实 | 第一条之后 V^T |\n|---|---|---|---|---|---|---|")
    for kind, lab in (("full_nogt", "不给答案"), ("full_gt", "给答案")):
        for g, gl in (("before", "区间之前"), ("in", "区间 ±1"), ("after", "区间之后")):
            d = N["e5_by_position"][kind][g]
            w(f"| {lab} | {gl} | {d['n_errors']} | {ci(d['genuine'], pct=True)} | {d['n_first']} | {ci(d['first_genuine'], pct=True)} | {ci(d['VT_after_first'])} |")
if "e8" in N:
    w("\n**E8 列出的错误整体属实率**(Qwen3-8B 思考模式):")
    for k, d in N["e8"].items():
        w(f"- {k}:{d['n_errors']} 条错误,属实 {ci(d['genuine_rate'], pct=True)};按位置 {json.dumps(d['by_position'], ensure_ascii=False)}")
mc = N["method_cut"]
w(f"\n**方法断点**:可用 {mc['usable']} 条(排除 {mc['excluded']}),平均 {mc['n_err_mean']:.2f} 条错误;断点在关闭点及之后 {ci(mc['at_or_after_closing_point'], pct=True)};"
  f"断点处 teacher 直接写 <0.5 的比例 {ci(N.get('e6_teacher_direct_below_half'), pct=True)}。\n")
w("**E6 从方法断点出发的五种续写**(各 8 次):\n")
w("| 条件 | 成功率 | n |\n|---|---|---|")
for lab, (m, n) in N["e6"].items():
    w(f"| {lab} | {ci(m)} | {n} |")
w(f"\n- 学生:看反馈 − 通用提示 {ci(N['e6_feedback_minus_generic'], sign=True)};看反馈 − 直接写 {ci(N['e6_feedback_minus_direct_student'], sign=True)}")
w(f"- teacher:看反馈 − 直接写 {ci(N['e6_teacher_feedback_minus_direct'], sign=True)}")
for name, d in N.get("e6_split", {}).items():
    w(f"- 拆分 {name}(n={d['n']}):" + ";".join(f"{k} {ci(v)}" for k, v in d.items() if k not in ("n",) and "minus" not in k)
      + f";teacher 看反馈−直接写 {ci(d['teacher_feedback_minus_direct'], sign=True)};学生 看反馈−通用 {ci(d['student_feedback_minus_generic'], sign=True)}")
w("\n**E7 可教性差 Δ**(图 3(c) 主图:尾部 = 两者都不看反馈;重写 = 训练条件,teacher 看反馈、学生不看):\n")
w("| 位置 | 被丢弃的尾部 Δ | 重写 Δ(训练条件) | 重写 Δ(两者都看反馈) | 重写 Δ(两者都不看) | 尾部 n | 重写 n |\n|---|---|---|---|---|---|---|")
for q, d in N["e7"].items():
    w(f"| {float(q):.0%} | {ci(d['tail'], sign=True)} | {ci(d['rewrite_train'], sign=True)} | {ci(d['rewrite_both_fb'], sign=True)} | {ci(d['rewrite_no_fb'], sign=True)} | {d['n_tail']} | {d['n_rewrite']} |")
w("\n**E7 两者的绝对水平**(图 3 备选版 (c)):\n")
w("| 位置 | 尾部 V^T | 尾部 V^S | 重写 V^T(看反馈) | 重写 V^S(不看反馈) | V^T 提升(配对) | V^S 提升(配对) | 两者都<0.5:尾部 → 重写 |\n|---|---|---|---|---|---|---|---|")
for q, d in N["e7"].items():
    pp = N.get("e7_paired", {}).get(q, {})
    w(f"| {float(q):.0%} | {ci(d['VT_tail'])} | {ci(d['VS_tail'])} | {ci(d['VT_rw_fb'])} | {ci(d['VS_rw_nofb'])} | {ci(pp.get('VT_rewrite_minus_tail'), sign=True)} | "
      f"{ci(pp.get('VS_rewrite_minus_tail'), sign=True)} | {pp.get('both_low_tail', float('nan')):.0%} → {pp.get('both_low_rewrite', float('nan')):.0%} |")
if "e10" in N:
    w("\n## 问题 3:崩溃区间与错误区间的关系(E9 + E10)\n")
    w("每个锚点在错误段之前和之后各截一刀,teacher 与学生各续写 16 次。分类:**包含** = 错误之后 teacher 仍 ≥0.5(崩溃在更后面,中间是教学窗口);"
      "**重合** = 错误之前 ≥0.5、之后 <0.5;**错位** = 错误之前 teacher 已 <0.5(崩溃先于这个错误)。\n")
    if "judge_first_found" in N["e10"]:
        d = N["e10"]["judge_first_found"]; w(f"E9 独立标注:{d['n']} 条中 {d['with_first']} 条给出第一个错误,{d['none']} 条回答 NONE。\n")
    w("| 锚点 | n | 包含 | 重合 | 错位 | 错误前 V^T | 错误后 V^T | 错误前 V^S | 错误后 V^S | 相对关闭区间的位置 |\n|---|---|---|---|---|---|---|---|---|---|")
    for k, lab in (("judge_first", "独立标注的第一个错误"), ("critic_first", "critic 第一条错误"), ("critic_last", "critic 最后一条错误(方法断点)")):
        if k not in N["e10"]: continue
        d = N["e10"][k]; c = d["classes"]
        w(f"| {lab} | {d['n']} | {c['contain']:.0%} | {c['coincide']:.0%} | {c['precede']:.0%} | {ci(d['VT_before'])} | {ci(d['VT_after'])} | "
          f"{ci(d['VS_before'])} | {ci(d['VS_after'])} | {d['position_vs_closing_interval']} |")
w("\n## 文件\n")
w("- `figs/fig_window.*`:图 2;`figs/fig_critique.*`:图 3(按原方案,(a) 为定位);`figs/fig_critique_alt.*`:图 3 备选((a) 为列出错误的属实率)")
w("- `figs/fig_app_*`:附录图(续写结果构成、E5 定位散点与偏差直方图、E7 分项)")
w("- `numbers.json`:全部数字;`data/`:精简数据(各状态成功率、批改与判定原文、关闭点、方法断点);`code/`:生成这些结果的代码")
open(f"{DST}/SUMMARY.md", "w").write("\n".join(L) + "\n")
subprocess.run(["tar", "czf", f"{DST}.tar.gz", "-C", os.path.dirname(DST), NAME], check=True)
print(f"打包完成: {DST}  ({subprocess.run(['du', '-sh', DST], capture_output=True, text=True).stdout.split()[0]}),{DST}.tar.gz")
