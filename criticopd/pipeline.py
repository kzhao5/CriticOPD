#!/usr/bin/env python
"""CriticOPD 离线流水线:学生 rollout -> 判错 -> teacher 定位首错并给反馈 -> 学生带反馈重写 -> 判分。

为什么先做离线:算法说明里有三件事没定 —— critic prompt、分段方式、两个重写选项选哪个。
这三件都是训练数据的生成参数,定不下来就没法写训练代码;而它们用一次离线采集就能同时测出来,
比开五个训练 arm 便宜两个数量级。产物同时就是训练要用的修复数据。

stages:
  rollout  学生在 DAPO 第 3200 行之后的题上生成(上限 4096),判分,留下"写完了但答错"的
  critic   teacher 读题目+分段后的 rollout,指出第一处出错的段落并给反馈(不准解题、不准给答案)
  repair   两个拼接选项各重写 k 次,判分
  report   修复率/成功率/解析失败率/反馈泄漏率/分段统计
"""
import argparse, json, os, re, random, sys
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")

TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
STUDENT0 = "/home/kzhao2/OPD/model/Qwen3-1.7B"
CKPT = "/home/kzhao2/Relay-OPD/outputs/checkpoints"
TRAIN = "/home/kzhao2/OPD/datasets/dapo-math-17k.parquet"
OUT = os.environ.get("CRITIC_OPD_OUT", "/home/kzhao2/Relay-OPD/criticopd/out")
STOP_IDS = [151643, 151645]
KEEP_CAPPED = os.environ.get("CRITIC_KEEP_CAPPED", "1") == "1"
ROLLOUT_CAP = 12288         # 规格书给的 4096 是他们模型上的:我们这个学生在 DAPO 上 4096 只覆盖 7.8%,
                            # 800 条里 738 条顶上限,可用样本只剩 4 条。实测 OPD 在 aime24 上平均 8462 token,
                            # 12288 才够。放宽后"答错率"也会回到真实值 —— 能在 4096 内写完的本来就是简单题。
REPAIR_CAP = 8192
SEED = 42


def model_for(run, step):
    return STUDENT0 if step == 0 else f"{CKPT}/{run}/global_step_{step}/actor/huggingface"


def _grader():
    sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
    os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
    from opd.reward.math_reward import compute_score
    def g(resp, gt):
        try:
            v = compute_score(solution_str=resp, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False
    return g


# ---------------------------------------------------------------- 分段
_SEP = re.compile(r"\n\s*\n")           # 空行
# 引导句/标题/分隔线:按形态判断,不按长度。只按"短"判断会把正常的短内容块(一行公式、
# 一句结论)也并掉,那样测出来的定位质量反映的是分段器而不是 critic。
_LEADIN = re.compile(r"^\s*(?:#{1,6}\s|[-*_=]{3,}\s*$|\*\*[^*]{0,40}\*\*\s*:?\s*$"
                     r"|(?:So|Now|Then|Next|Thus|Hence|Therefore|Step\s*\d+|Case\s*\d+)\b[^.\n]{0,30}:\s*$)",
                     re.I)


def _is_leadin(p):
    return bool(_LEADIN.match(p)) or (len(p) <= 12 and p.rstrip().endswith(":"))


def segment(text, mode="para_merge"):
    """把 rollout 切成语义单元。

    para       : 只按空行切
    para_merge : 按空行切,再把过短的单元(标题、分隔线、"So:" 这类引导句)并入后一段。
                 规格书指出约 1/3 的案例 critic 指到了这种短单元旁边,切粗能减少歧义。
    sent       : 按句号/换行切(更细,作为对照)
    """
    if mode == "sent":
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
        return parts
    parts = [p.strip() for p in _SEP.split(text) if p.strip()]
    if mode == "para":
        return parts
    merged, buf = [], []
    for p in parts:
        if _is_leadin(p):
            buf.append(p)                     # 引导句本身不成段,并入它后面的块
            continue
        merged.append("\n\n".join(buf + [p]) if buf else p)
        buf = []
    if buf: merged.append("\n\n".join(buf))  # 结尾残留的引导句单独成段
    return merged or parts


# ---------------------------------------------------------------- critic prompt
_CRITIC_HEAD = {
    # 写完了但答错
    "finished_wrong":
        "You are a meticulous math teacher reviewing a student's solution. "
        "The student's FINAL ANSWER IS WRONG. Your job is to find the FIRST place where the "
        "reasoning goes wrong, and say what should have been done instead.",
    # 没写完就被截断:不能说"最终答案错了"—— 它根本没有最终答案。
    "capped":
        "You are a meticulous math teacher reviewing a student's PARTIAL solution. "
        "The student never reached an answer -- the reasoning ran on without converging. "
        "Your job is to find the FIRST place where it went off track, and say what should "
        "have been done instead.",
    # GT 模式:把"你已知正确答案"写进任务定义,并给出定位方法(而不是当附加规则硬塞)
    "finished_wrong_gt":
        "You are a meticulous math teacher. A student has submitted a complete solution whose "
        "FINAL ANSWER IS WRONG. You are given the correct answer. Use it as a reference to "
        "trace the student's work forward and find the EARLIEST step after which the correct "
        "answer can no longer be reached.",
    "capped_gt":
        "You are a meticulous math teacher. A student's solution ran on without ever reaching "
        "an answer. You are given the correct answer. Use it as a reference to trace the "
        "student's work forward and find the EARLIEST step that sent the reasoning off track.",
}

CRITIC_SYS = (
    "{head}\n\n"
    "How to locate the error:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this "
    "step still correct, and does it still allow the problem to be solved? The FIRST segment "
    "where the answer is no is the error.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later "
    "abandons or corrects; a longer route than necessary; or sloppy notation that does not change "
    "the mathematics. Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say what is wrong and what to do from that point instead. The student will "
    "resume writing immediately after this point, so phrase it as guidance they can act on. "
    "Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that "
    "evaluates to it, do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format:\n"
    "SEGMENT: <index, or NONE>\n"
    "QUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>"
)


GIVE_GT = os.environ.get("CRITIC_OPD_GIVE_GT", "0") == "1"
_GT_RULE = ("\n5. You are told the correct answer only so that you can locate the student's error. "
            "NEVER write it, quote it, or hint at its value. Your feedback must describe the mistake "
            "and the correct method, not the answer.")


def critic_prompt(question, segs, kind="finished_wrong", gt=None):
    body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
    if GIVE_GT and gt is not None:
        kind = kind + "_gt"
    sys_msg = CRITIC_SYS.format(head=_CRITIC_HEAD[kind])
    user = f"Problem:\n{question}\n\n"
    if GIVE_GT and gt is not None:
        user += (f"Correct answer (reference only -- never reveal it): {gt}\n\n")
    user += f"Student's solution, split into numbered segments:\n\n{body}"
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


_RE_SEG = re.compile(r"SEGMENT:\s*\[?(\d+)", re.I)
_RE_Q = re.compile(r"QUOTE:\s*(.+?)(?=\n\s*FEEDBACK:|\Z)", re.I | re.S)
_RE_FB = re.compile(r"FEEDBACK:\s*(.+)", re.I | re.S)


def parse_critic(text, segs):
    """解析 critic 输出。优先用 QUOTE 在原文里定位(规格书:不要只靠编号),编号作为兜底。"""
    mq, mf, ms = _RE_Q.search(text), _RE_FB.search(text), _RE_SEG.search(text)
    # critic 明确说「找不到错」时直接放弃该样本:错误的定位比没有定位更糟,
    # 它会让学生从一个本来正确的位置重写。不能让 QUOTE 的模糊匹配把这个逃生口绕过去。
    if re.search(r"SEGMENT:\s*\[?\s*(NONE|N/A|-)\b", text, re.I):
        return {"seg": None, "quote": None, "feedback": None, "located_by": None,
                "parse_ok": False, "no_error_found": True}
    fb = mf.group(1).strip() if mf else None
    quote = mq.group(1).strip().strip('"') if mq else None
    idx, how = None, None
    if quote and len(quote) >= 8:
        # 先精确子串,再退到归一化空白后的匹配(critic 常把换行抄成空格)
        norm = lambda t: re.sub(r"\s+", " ", t).strip()
        nq = norm(quote)[:80]
        for i, s in enumerate(segs):
            if quote[:60] in s or nq in norm(s):
                idx, how = i, "quote"; break
    if idx is None and ms:
        j = int(ms.group(1))
        if 0 <= j < len(segs): idx, how = j, "index"
    return {"seg": idx, "quote": quote, "feedback": fb, "located_by": how,
            "parse_ok": idx is not None and bool(fb)}


# 反馈里泄漏最终答案的检测:规格书说约 13% 的反馈把标准答案算了出来
def leaks_answer(fb, gt):
    if not fb or gt is None: return False
    g = str(gt).strip()
    if not g: return False
    nums = re.findall(r"-?\d+(?:\.\d+)?", g)
    return any(n in fb for n in nums if len(n) >= 2) or (len(g) >= 3 and g in fb)


FEEDBACK_TMPL = ("\n\n[Teacher feedback] {fb}\n\nUsing this feedback, continue the solution from "
                 "here. Put your final answer in \\boxed{{}}.\n\n")


# ---------------------------------------------------------------- stage: rollout
def stage_rollout(run, step, n_prob, k, seg_mode):
    """学生在 DAPO 第 3200 行之后的题上生成;只留下"写完了但答错"的。

    没写完的(顶到 4096)不要:critic 对一段没有结论的推理无从判"第一处错误",
    而且它们对应的是另一个失效模式(不收敛),不是这个算法针对的对象。
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    grade = _grader()
    df = pd.read_parquet(TRAIN).iloc[3200:].reset_index(drop=True)
    idx = list(range(len(df))); random.Random(SEED).shuffle(idx); idx = idx[:n_prob]
    tok = AutoTokenizer.from_pretrained(STUDENT0, trust_remote_code=True)
    llm = LLM(model=model_for(run, step), dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=ROLLOUT_CAP + 2048, enable_prefix_caching=True)
    prompts, meta = [], []
    for i in idx:
        r = df.iloc[i]
        msg = r["prompt"].tolist() if hasattr(r["prompt"], "tolist") else list(r["prompt"])
        # apply_chat_template(tokenize=True) 返回 BatchEncoding 而不是 int 列表,
        # 索引它得到 Encoding 对象,会一路带到 vLLM 才报 str>int 的类型错误。
        pid = tok.encode(tok.apply_chat_template(msg, add_generation_prompt=True,
                                                 tokenize=False), add_special_tokens=False)
        gt = r["reward_model"]["ground_truth"]
        for rep in range(k):
            prompts.append({"prompt_token_ids": pid}); meta.append((int(i), pid, str(gt), rep,
                                                                   msg[-1]["content"]))
    print(f"[rollout] {len(prompts)} 次生成 = {n_prob} 题 x {k}", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=ROLLOUT_CAP,
                                                stop_token_ids=STOP_IDS, seed=SEED))
    rows, stat = [], {"total": 0, "finished": 0, "correct": 0, "wrong_finished": 0, "capped": 0}
    for (i, pid, gt, rep, q), o in zip(meta, outs):
        txt = o.outputs[0].text; capped = o.outputs[0].finish_reason == "length"
        ok = (not capped) and grade(txt, gt)
        stat["total"] += 1; stat["capped"] += capped; stat["finished"] += (not capped)
        stat["correct"] += ok
        if ok: continue                      # 答对的不要
        # 截断的也留下来。理由:首错的中位位置是 token 1050,70% 在前 3000 token 内,
        # 而截断轨迹长度 >= ROLLOUT_CAP —— 前面那一段是完整可读的,结论的缺失不影响它。
        # 丢掉截断样本等于丢掉 49% 的 rollout,而那正是"不收敛"这个失效模式。
        if capped and not KEEP_CAPPED: continue
        stat["wrong_finished"] += (not capped)
        segs = segment(txt, seg_mode)
        rows.append({"prob": i, "rep": rep, "question": q, "gt": gt, "text": txt,
                     "kind": "capped" if capped else "finished_wrong",
                     "prompt_ids": list(pid), "resp_ids": list(o.outputs[0].token_ids),
                     "n_tok": len(o.outputs[0].token_ids), "n_seg": len(segs)})
    with open(f"{OUT}/wrong.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    stat["seg_mode"] = seg_mode
    stat["mean_seg"] = round(sum(r["n_seg"] for r in rows) / max(len(rows), 1), 1)
    stat["mean_tok"] = round(sum(r["n_tok"] for r in rows) / max(len(rows), 1), 1)
    json.dump(stat, open(f"{OUT}/rollout_stat.json", "w"), indent=1)
    print(f"[rollout] {json.dumps(stat, ensure_ascii=False)}")
    print(f"[rollout] -> {OUT}/wrong.jsonl  ({len(rows)} 条可用于 critic)")


# ---------------------------------------------------------------- stage: critic
def stage_critic(seg_mode, max_items):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(f"{OUT}/wrong.jsonl")][:max_items]
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=16384, enable_prefix_caching=True)
    prompts, segs_all = [], []
    for r in rows:
        segs = segment(r["text"], seg_mode); segs_all.append(segs)
        prompts.append(tok.apply_chat_template(
            critic_prompt(r["question"], segs, r.get("kind", "finished_wrong"), gt=r["gt"]),
            add_generation_prompt=True, tokenize=False))
    print(f"[critic] {len(prompts)} 条", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512, seed=SEED))
    res, st = [], {"n": 0, "parse_ok": 0, "by_quote": 0, "by_index": 0, "leak": 0,
                   "seg_pos_sum": 0.0, "crit_tok": 0}
    per_kind = {}
    for r, segs, o in zip(rows, segs_all, outs):
        p = parse_critic(o.outputs[0].text, segs)
        st["n"] += 1; st["crit_tok"] += len(o.outputs[0].token_ids)
        st["parse_ok"] += p["parse_ok"]
        st["by_quote"] += p["located_by"] == "quote"; st["by_index"] += p["located_by"] == "index"
        leak = leaks_answer(p["feedback"], r["gt"]); st["leak"] += leak
        if p["seg"] is not None: st["seg_pos_sum"] += p["seg"] / max(len(segs), 1)
        k = r.get("kind", "finished_wrong")
        d = per_kind.setdefault(k, {"n": 0, "ok": 0, "quote": 0, "leak": 0, "pos": 0.0})
        d["n"] += 1; d["ok"] += p["parse_ok"]; d["quote"] += p["located_by"] == "quote"
        d["leak"] += leak
        if p["seg"] is not None: d["pos"] += p["seg"] / max(len(segs), 1)
        res.append({**r, "segs": segs, "critic_raw": o.outputs[0].text, **p, "leak": leak})
    with open(f"{OUT}/critic.jsonl", "w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    n = max(st["n"], 1); ok = max(st["parse_ok"], 1)
    summ = {"n": st["n"], "parse_ok_rate": round(st["parse_ok"]/n, 3),
            "located_by_quote": round(st["by_quote"]/ok, 3),
            "located_by_index": round(st["by_index"]/ok, 3),
            "leak_rate": round(st["leak"]/n, 3),
            "mean_error_pos_rel": round(st["seg_pos_sum"]/ok, 3),
            "mean_critic_tokens": round(st["crit_tok"]/n, 1), "seg_mode": seg_mode,
            "by_kind": {k: {"n": v["n"], "parse_ok": round(v["ok"]/max(v["n"],1), 3),
                            "by_quote": round(v["quote"]/max(v["ok"],1), 3),
                            "leak": round(v["leak"]/max(v["n"],1), 3),
                            "mean_pos": round(v["pos"]/max(v["ok"],1), 3)}
                        for k, v in per_kind.items()}}
    json.dump(summ, open(f"{OUT}/critic_stat.json", "w"), indent=1)
    print(f"[critic] {json.dumps(summ, ensure_ascii=False)}")


# ---------------------------------------------------------------- stage: repair
def stage_repair(run, step, k):
    """两个拼接选项各重写 k 次。

    A1 = 截到出错段之前(不含出错段)+ 反馈 -> 学生从这里重写
         学生学的是"在这一步不要犯这个错"
    A2 = 保留出错段 + 反馈放在它后面 -> 学生反思后继续
         学生学的是"犯错之后自己发现并纠正"

    两者的差别只在错误内容是否留在上下文里,所以必须同一批样本、同样的采样预算对比。
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    grade = _grader()
    rows = [json.loads(l) for l in open(f"{OUT}/critic.jsonl") if json.loads(l)["parse_ok"]]
    tok = AutoTokenizer.from_pretrained(STUDENT0, trust_remote_code=True)
    llm = LLM(model=model_for(run, step), dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=ROLLOUT_CAP + REPAIR_CAP + 2048, enable_prefix_caching=True)
    reqs, meta = [], []
    for r in rows:
        segs, e = r["segs"], r["seg"]
        # A1/A2 = 两个拼接选项;NOFB/FRESH 是必需的对照:
        #   NOFB  从同一个首错处截断但不给反馈 -> 区分"反馈起作用" vs "只是切掉了后面的循环"
        #   FRESH 同一道题从头重采 -> 区分"修复有用" vs "多采一次就能中"
        # 没有这两个对照,修复成功率本身说明不了任何事。
        for opt in ("A1", "A2", "NOFB", "FRESH"):
            if opt == "FRESH":
                ids = list(r["prompt_ids"])
            else:
                keep = segs[:e] if opt == "A1" else segs[:e + 1]
                prefix = "\n\n".join(keep)
                spliced = prefix + (FEEDBACK_TMPL.format(fb=r["feedback"]) if opt in ("A1", "A2")
                                    else "\n\n")
                ids = r["prompt_ids"] + tok.encode(spliced, add_special_tokens=False)
            for rep in range(k):
                reqs.append({"prompt_token_ids": ids})
                meta.append((r, opt, rep, len(r["prompt_ids"]), len(ids)))
    print(f"[repair] {len(reqs)} 次 = {len(rows)} 条 x 4 分支(A1/A2/NOFB/FRESH) x {k}", flush=True)
    outs = llm.generate(reqs, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=REPAIR_CAP,
                                             stop_token_ids=STOP_IDS, seed=SEED))
    res = []
    for (r, opt, rep, n_p, n_ctx), o in zip(meta, outs):
        txt = o.outputs[0].text; capped = o.outputs[0].finish_reason == "length"
        res.append({"prob": r["prob"], "rollout_rep": r["rep"], "opt": opt, "rep": rep,
                    "gt": r["gt"], "seg": r["seg"], "n_seg": len(r["segs"]), "leak": r["leak"],
                    "ctx_tokens": n_ctx - n_p, "new_tokens": len(o.outputs[0].token_ids),
                    "capped": capped, "correct": bool((not capped) and grade(txt, r["gt"])),
                    "text": txt})
    with open(f"{OUT}/repair.jsonl", "w") as f:
        for x in res: f.write(json.dumps(x) + "\n")
    print(f"[repair] -> {OUT}/repair.jsonl ({len(res)} 条)")


# ---------------------------------------------------------------- stage: report
def stage_report():
    from statistics import mean
    import collections
    ro = json.load(open(f"{OUT}/rollout_stat.json"))
    cs = json.load(open(f"{OUT}/critic_stat.json"))
    rp = [json.loads(l) for l in open(f"{OUT}/repair.jsonl")]
    print("=" * 74)
    print("CriticOPD 离线流水线")
    print("=" * 74)
    print(f"\n[1] 学生 rollout(上限 {ROLLOUT_CAP})")
    print(f"    共 {ro['total']} 次 | 写完 {100*ro['finished']/ro['total']:.1f}%"
          f" | 顶到上限 {100*ro['capped']/ro['total']:.1f}% | 正确 {100*ro['correct']/ro['total']:.1f}%")
    print(f"    可用(写完但错) {ro['wrong_finished']} 条 = {100*ro['wrong_finished']/ro['total']:.1f}%")
    print(f"    分段模式 {ro['seg_mode']}: 平均 {ro['mean_seg']} 段 / {ro['mean_tok']} token")
    print(f"\n[2] critic 定位")
    print(f"    解析成功 {100*cs['parse_ok_rate']:.1f}%  (靠引用定位 {100*cs['located_by_quote']:.0f}%,"
          f" 靠编号 {100*cs['located_by_index']:.0f}%)")
    print(f"    反馈泄漏答案 {100*cs['leak_rate']:.1f}%   首错相对位置 {cs['mean_error_pos_rel']:.2f}"
          f"   critic 平均 {cs['mean_critic_tokens']:.0f} token")
    if "by_kind" in cs:
        print(f"\n[2b] 按轨迹类型分(finished_wrong = 写完但答错, capped = 没写完被截断)")
        print(f"    {'类型':16s}{'n':>6s}{'解析成功':>10s}{'引用定位':>10s}{'泄漏':>8s}{'首错相对位置':>14s}")
        for k, v in cs["by_kind"].items():
            print(f"    {k:16s}{v['n']:6d}{100*v['parse_ok']:9.1f}%{100*v['by_quote']:9.0f}%"
                  f"{100*v['leak']:7.1f}%{v['mean_pos']:14.2f}")
    print(f"\n[3] 两个拼接选项(同一批样本,同样采样预算)")
    print(f"    {'选项':6s}{'n':>6s}{'修复成功率':>12s}{'顶上限':>9s}{'平均新token':>12s}{'上下文token':>12s}")
    for opt in ("A1", "A2", "NOFB", "FRESH"):
        v = [x for x in rp if x["opt"] == opt]
        if not v: continue
        print(f"    {opt:6s}{len(v):6d}{100*mean(x['correct'] for x in v):11.1f}%"
              f"{100*mean(x['capped'] for x in v):8.1f}%{mean(x['new_tokens'] for x in v):12.0f}"
              f"{mean(x['ctx_tokens'] for x in v):12.0f}")
    # 题级配对:同一条错误 rollout 上 A1 vs A2 谁赢
    by = collections.defaultdict(dict)
    for x in rp: by[(x["prob"], x["rollout_rep"])].setdefault(x["opt"], []).append(x["correct"])
    pair = [(mean(v["A1"]), mean(v["A2"])) for v in by.values() if "A1" in v and "A2" in v]
    if pair:
        d = [a - b for a, b in pair]
        print(f"\n    配对比较(n={len(pair)} 条 rollout): A1-A2 = {100*mean(d):+.1f} 分"
              f"   A1 赢 {sum(1 for x in d if x>0)} / A2 赢 {sum(1 for x in d if x<0)}"
              f" / 平 {sum(1 for x in d if x==0)}")
    kinds = sorted({x.get("kind", "finished_wrong") for x in rp})
    if len(kinds) > 1:
        print(f"\n    按轨迹类型拆开:")
        print(f"      {'类型':16s}{'选项':6s}{'n':>6s}{'修复成功率':>12s}{'顶上限':>9s}")
        for k in kinds:
            for opt in ("A1", "A2", "NOFB", "FRESH"):
                v = [x for x in rp if x["opt"] == opt and x.get("kind", "finished_wrong") == k]
                if v: print(f"      {k:16s}{opt:6s}{len(v):6d}{100*mean(x['correct'] for x in v):11.1f}%"
                            f"{100*mean(x['capped'] for x in v):8.1f}%")
    nl = [x for x in rp if not x["leak"]]
    if nl and len(nl) < len(rp):
        print(f"\n    剔除反馈泄漏答案的样本后({len(rp)-len(nl)} 条被剔除):")
        for opt in ("A1", "A2"):
            v = [x for x in nl if x["opt"] == opt]
            if v: print(f"      {opt}: {100*mean(x['correct'] for x in v):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rollout", "critic", "repair", "report"])
    ap.add_argument("--run", default="opd_1p7b")
    ap.add_argument("--step", type=int, default=80)
    ap.add_argument("--n-prob", type=int, default=400)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--repair-k", type=int, default=3)
    ap.add_argument("--seg-mode", default="para_merge", choices=["para", "para_merge", "sent"])
    ap.add_argument("--max-items", type=int, default=400)
    a = ap.parse_args()
    if a.stage == "rollout": stage_rollout(a.run, a.step, a.n_prob, a.k, a.seg_mode)
    elif a.stage == "critic": stage_critic(a.seg_mode, a.max_items)
    elif a.stage == "repair": stage_repair(a.run, a.step, a.repair_k)
    else: stage_report()
