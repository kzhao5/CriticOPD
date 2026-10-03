#!/usr/bin/env python3
"""离线检验:给 critic 一份 teacher 写的参考解答,能不能让修复更容易成功。

对比三个 critic(同一批错误 rollout、同一个学生、同样的重写预算):
  A   现在的 R4GT:题目 + 正确答案 + 学生分段解答            (与训练逐字一致)
  B   A + 参考解答(teacher 自己做对的一份完整解答)
  B2  B + 允许 critic 先写一小段 ANALYSIS 再给结论

与训练保持一致的地方 —— 分段、critic 提示词(A)、解析、按 token 拼接、反馈模板、重写预算,
全部直接 import 训练用的 critic_opd_agent_loop;学生 prompt 用 enable_thinking=False。

和旧的 pipeline.py 不同:每个采样请求用各自的 seed。旧管线对所有请求传同一个
SamplingParams(seed=42),同一上下文的多次采样因此大量重复(1664 对修复里 363 对逐字相同)。

用法: python3 ref_probe.py {rollout|ref|critic|repair|report}
"""
import argparse, glob, json, os, random, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")

import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402
from pipeline import leaks_answer                                # noqa: E402

OUT_BASE = os.environ.get("PROBE_OUT", "/home/kzhao2/Relay-OPD/criticopd/out_ref")
# 按题目切分到多张卡:每个分片独立跑 rollout -> ref -> critic -> repair,写进 OUT_BASE/shard_k,
# report 阶段把所有分片合并。题目之间互不依赖,切分不改变结果。
SHARD = int(os.environ.get("PROBE_SHARD", "0"))
NSH = int(os.environ.get("PROBE_NSHARDS", "1"))
OUT = f"{OUT_BASE}/shard_{SHARD}" if NSH > 1 else OUT_BASE
STUDENT = "/home/kzhao2/Relay-OPD/outputs/checkpoints/criticopd_R4GT_pt17b/global_step_20/actor/huggingface"
TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
TOKENIZER = "/home/kzhao2/OPD/model/Qwen3-1.7B"
TRAIN = "/home/kzhao2/OPD/datasets/dapo-math-17k.parquet"
STOP_IDS = [151643, 151645]
RLEN = 16384            # 训练时的 max_response_length
SEED = 1234
N_PROB, K_ROLL = int(os.environ.get("PROBE_N_PROB", "500")), 2
N_REF, REF_MAXTOK = 4, 8192
N_REPAIR = 2
N_FP = int(os.environ.get("PROBE_N_FP", "150"))             # 答对的 rollout 抽多少条做「误报」检验

# ---------------------------------------------------------------- 参考解答版 critic 提示词
_REF_RULES = (
    "How to locate the error:\n"
    "Read the student's numbered segments in order. Check the mathematics of each one, using the "
    "reference solution as a guide to what is true. The FIRST segment containing a genuinely wrong "
    "step -- a false statement, a wrong computation, an invalid deduction, or a wrong setup -- is "
    "the error.\n\n"
    "The student may solve the problem differently from the reference. A different approach is NOT "
    "an error: flag a segment only if its mathematics is actually wrong, never merely because it "
    "differs from the reference. Also do NOT flag an exploratory attempt the student later abandons "
    "or corrects, a longer route than necessary, or sloppy notation that does not change the "
    "mathematics.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say concretely what is wrong at that step and what the correct reasoning for THAT "
    "step is (you may use the reference to state the correct method or fact for that step). The "
    "student will resume writing immediately after this point. Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not describe later steps of the reference and do not copy "
    "it.\n"
    "5. If every step looks correct to you, reply SEGMENT: NONE. Do not guess -- a wrong location is "
    "worse than none.\n\n")
_FMT = ("Reply in exactly this format:\n"
        "SEGMENT: <index, or NONE>\n"
        "QUOTE: <the exact wrong sentence, copied verbatim>\n"
        "FEEDBACK: <what is wrong and what to do instead>")
_HEAD = ("You are a meticulous math teacher. A student has submitted a complete solution whose FINAL "
         "ANSWER IS WRONG. You are given the correct final answer and a correct reference solution. "
         "Use them to find the EARLIEST step where the student's work goes wrong.\n\n")
CRITIC_SYS_REF = _HEAD + _REF_RULES + _FMT
CRITIC_SYS_REF_ANALYSIS = (
    _HEAD + _REF_RULES +
    "Before answering, write a short comparison of the student's work against the reference, at most "
    "150 words, starting with 'ANALYSIS:'. Then give the three lines below.\n\n" + _FMT)
VARIANTS = {"A": (None, 512), "B": (CRITIC_SYS_REF, 512), "B2": (CRITIC_SYS_REF_ANALYSIS, 1024)}


def jl_write(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def jl_read(path):
    return [json.loads(l) for l in open(path)]


def load(name):
    """report 用:有分片就合并所有分片,否则读单一目录。"""
    shards = sorted(glob.glob(f"{OUT_BASE}/shard_*/{name}"))
    return [x for f in shards for x in jl_read(f)] if shards else jl_read(f"{OUT_BASE}/{name}")


def grader():
    from opd.reward.math_reward import compute_score
    def g(text, gt):
        try:
            v = compute_score(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False
    return g


def sp_list(n, base, **kw):
    """每个请求各自的 seed —— 同 seed + 同 prompt 在 vLLM 里会得到逐字相同的输出。"""
    from vllm import SamplingParams
    return [SamplingParams(seed=base + i, **kw) for i in range(n)]


def located_by_quote(text, segs):
    """和训练里 parse_critic 的引用匹配逻辑一致,只用于统计「靠引用定位」的比例。"""
    mq = M._RE_Q.search(text)
    quote = mq.group(1).strip().strip('"') if mq else None
    if not quote or len(quote) < 8:
        return False
    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    nq = norm(quote)[:80]
    return any(quote[:60] in s or nq in norm(s) for s in segs)


# ---------------------------------------------------------------- stage: rollout
def stage_rollout():
    import pandas as pd
    from vllm import LLM
    from transformers import AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    g = grader()
    df = pd.read_parquet(TRAIN).iloc[3200:].reset_index(drop=True)
    idx = list(range(len(df))); random.Random(SEED).shuffle(idx); idx = idx[:N_PROB][SHARD::NSH]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    reqs, meta = [], []
    for i in idx:
        r = df.iloc[i]
        msgs = r["prompt"].tolist() if hasattr(r["prompt"], "tolist") else list(r["prompt"])
        msgs = [dict(m) for m in msgs]
        pid = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False), add_special_tokens=False)
        for k in range(K_ROLL):
            reqs.append({"prompt_token_ids": pid})
            meta.append({"prob": int(i), "rep": k, "messages": msgs, "question": msgs[-1]["content"],
                         "gt": str(r["reward_model"]["ground_truth"]), "prompt_ids": pid})
    llm = LLM(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.88,
              max_model_len=RLEN + 2048, enable_prefix_caching=True)
    outs = llm.generate(reqs, sp_list(len(reqs), SEED, temperature=1.0, top_p=1.0,
                                      max_tokens=RLEN, stop_token_ids=STOP_IDS))
    rows = []
    for m, o in zip(meta, outs):
        ids = list(o.outputs[0].token_ids); txt = o.outputs[0].text
        capped = len(ids) >= RLEN
        rows.append({**m, "resp_ids": ids, "text": txt, "capped": capped,
                     "correct": bool((not capped) and g(txt, m["gt"]))})
    jl_write(f"{OUT}/rollouts.jsonl", rows)
    n = len(rows)
    print(f"[rollout] {n} 条: 答对 {sum(r['correct'] for r in rows)}  写完答错 "
          f"{sum((not r['correct']) and (not r['capped']) for r in rows)}  写不完 {sum(r['capped'] for r in rows)}"
          f"  重复文本 {n - len({(r['prob'], r['text']) for r in rows})}")


def select_items(rows):
    """错误样本 = 写完但答错(训练中 critic 的主要来源);误报样本 = 答对的,每题最多 1 条。"""
    wrong = [r for r in rows if (not r["correct"]) and (not r["capped"])]
    seen, fp = set(), []
    for r in sorted((r for r in rows if r["correct"]), key=lambda r: (r["prob"], r["rep"])):
        if r["prob"] in seen: continue
        seen.add(r["prob"]); fp.append(r)
    random.Random(SEED).shuffle(fp)
    return wrong, fp[:-(-N_FP // NSH)]          # 误报样本按分片均分


# ---------------------------------------------------------------- stage: ref
def stage_ref():
    from vllm import LLM
    from transformers import AutoTokenizer
    g = grader()
    wrong, fp = select_items(jl_read(f"{OUT}/rollouts.jsonl"))
    probs = {}
    for r in wrong + fp: probs[r["prob"]] = r
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    reqs, meta = [], []
    for p, r in probs.items():
        ids = tok.encode(tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False),
                         add_special_tokens=False)
        for k in range(N_REF):
            reqs.append({"prompt_token_ids": ids}); meta.append(p)
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.88,
              max_model_len=REF_MAXTOK + 4096, enable_prefix_caching=True)
    outs = llm.generate(reqs, sp_list(len(reqs), SEED + 100000, temperature=0.7, top_p=0.95,
                                      max_tokens=REF_MAXTOK, stop_token_ids=STOP_IDS))
    cand = {}
    for p, o in zip(meta, outs):
        txt = o.outputs[0].text; ok = o.outputs[0].finish_reason != "length" and g(txt, probs[p]["gt"])
        cand.setdefault(p, []).append((ok, len(o.outputs[0].token_ids), txt))
    refs = []
    for p, c in cand.items():
        good = sorted((x for x in c if x[0]), key=lambda x: x[1])
        refs.append({"prob": p, "n_ok": len(good), "ref": good[0][2] if good else None,
                     "ref_tokens": good[0][1] if good else None})
    jl_write(f"{OUT}/refs.jsonl", refs)
    cov = sum(r["ref"] is not None for r in refs)
    print(f"[ref] {len(refs)} 题,teacher 4 次内做对(有参考解答) {cov} = {100*cov/len(refs):.1f}%;"
          f" 参考解答长度中位 {statistics.median([r['ref_tokens'] for r in refs if r['ref']]):.0f} token")


# ---------------------------------------------------------------- stage: critic
def critic_messages(variant, r, segs, ref):
    body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
    head = f"Problem:\n{r['question']}\n\nCorrect answer (reference only -- never reveal it): {r['gt']}\n\n"
    if variant == "A":          # 训练里 R4GT 的 critic,逐字一致
        return [{"role": "system", "content": M.CRITIC_SYS_GT},
                {"role": "user", "content": head + f"Student's solution, split into numbered segments:\n\n{body}"}]
    return [{"role": "system", "content": VARIANTS[variant][0]},
            {"role": "user", "content": head + "Reference solution (correct; the student may use a different "
                                               f"valid approach):\n{ref}\n\n"
                                               f"Student's solution, split into numbered segments:\n\n{body}"}]


def stage_critic():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    wrong, fp = select_items(jl_read(f"{OUT}/rollouts.jsonl"))
    refs = {r["prob"]: r["ref"] for r in jl_read(f"{OUT}/refs.jsonl")}
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    MAXLEN = 40960
    reqs, sps, meta = [], [], []
    for kind, items in (("wrong", wrong), ("fp", fp)):
        for r in items:
            segs = M.segment(r["text"]); ref = refs.get(r["prob"])
            for v, (_, maxtok) in VARIANTS.items():
                if v != "A" and not ref: continue
                ids = tok.encode(tok.apply_chat_template(critic_messages(v, r, segs, ref), add_generation_prompt=True,
                                                         tokenize=False), add_special_tokens=False)
                if len(ids) + maxtok > MAXLEN:
                    continue
                reqs.append({"prompt_token_ids": ids}); sps.append(SamplingParams(temperature=0.0, max_tokens=maxtok))
                meta.append((kind, r, v, segs, ref is not None))
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=MAXLEN,
              enable_prefix_caching=True)
    outs = llm.generate(reqs, sps)
    res = []
    for (kind, r, v, segs, has_ref), o in zip(meta, outs):
        txt = o.outputs[0].text
        idx, fb = M.parse_critic(txt, segs)
        none = bool(M._RE_NONE.search(txt))
        res.append({"kind": kind, "prob": r["prob"], "rep": r["rep"], "variant": v, "has_ref": has_ref,
                    "n_seg": len(segs), "seg": idx, "feedback": fb, "none": none,
                    "parsed": idx is not None and bool(fb), "by_quote": located_by_quote(txt, segs),
                    "leak": bool(fb) and leaks_answer(fb, r["gt"]), "critic_tokens": len(o.outputs[0].token_ids),
                    "critic_text": txt})
    jl_write(f"{OUT}/critic.jsonl", res)
    print(f"[critic] {len(res)} 次 critic(错误 {sum(x['kind']=='wrong' for x in res)},误报检验 {sum(x['kind']=='fp' for x in res)})")


# ---------------------------------------------------------------- stage: repair
def stage_repair():
    from vllm import LLM
    from transformers import AutoTokenizer
    g = grader()
    rolls = {(r["prob"], r["rep"]): r for r in jl_read(f"{OUT}/rollouts.jsonl")}
    crit = [c for c in jl_read(f"{OUT}/critic.jsonl") if c["kind"] == "wrong" and c["parsed"]]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    MAXLEN = RLEN + 4096
    reqs, meta = [], []
    for c in crit:
        r = rolls[(c["prob"], c["rep"])]
        segs = M.segment(r["text"])
        kept_txt = "\n\n".join(segs[:c["seg"] + 1])                      # SPLICE=after,与 R4GT 一致
        k = M._tok_prefix_for_text(tok, r["resp_ids"], kept_txt)
        kept_ids = r["resp_ids"][:k]
        kept_txt = tok.decode(kept_ids, skip_special_tokens=True)
        ctx_ids = tok.encode(kept_txt + M.FEEDBACK_TMPL.format(fb=c["feedback"]), add_special_tokens=False)
        ids = r["prompt_ids"] + ctx_ids
        budget = min(max(256, RLEN - len(kept_ids)), MAXLEN - len(ids) - 1)
        if budget < 256:
            continue
        for j in range(N_REPAIR):
            reqs.append({"prompt_token_ids": ids})
            meta.append((c, j, kept_txt, len(kept_ids), budget))
    from vllm import SamplingParams
    sps = [SamplingParams(seed=SEED + 200000 + i, temperature=1.0, top_p=1.0, max_tokens=m[4],
                          stop_token_ids=STOP_IDS) for i, m in enumerate(meta)]
    llm = LLM(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=MAXLEN,
              enable_prefix_caching=True)
    outs = llm.generate(reqs, sps)
    res = []
    for (c, j, kept_txt, n_kept, budget), o in zip(meta, outs):
        txt = o.outputs[0].text; n_new = len(o.outputs[0].token_ids); capped = n_new >= budget
        res.append({"prob": c["prob"], "rep": c["rep"], "variant": c["variant"], "has_ref": c["has_ref"], "j": j,
                    "n_kept": n_kept, "new_tokens": n_new, "capped": capped,
                    "correct": bool((not capped) and g(kept_txt + txt, rolls[(c["prob"], c["rep"])]["gt"])),
                    "text": txt})
    jl_write(f"{OUT}/repair.jsonl", res)
    print(f"[repair] {len(res)} 次重写")


# ---------------------------------------------------------------- stage: report
def boot_diff(xs, ys, n=10000, seed=0):
    rng = random.Random(seed); d = [x - y for x, y in zip(xs, ys)]; m = len(d)
    bs = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, bs[int(.025 * n)], bs[int(.975 * n)]


def stage_report():
    rolls = load("rollouts.jsonl"); refs = load("refs.jsonl")
    crit = load("critic.jsonl"); rep = load("repair.jsonl")
    wrong, fp = select_items(rolls)
    cov = sum(r["ref"] is not None for r in refs)
    print(f"学生 R4GT@20 rollout:{len(rolls)} 条,答对 {sum(r['correct'] for r in rolls)},写完答错 {len(wrong)},"
          f"写不完 {sum(r['capped'] for r in rolls)}")
    print(f"参考解答覆盖:{cov}/{len(refs)} 题 = {100*cov/len(refs):.1f}%\n")

    C = {(c["kind"], c["prob"], c["rep"], c["variant"]): c for c in crit}
    R = {}
    for x in rep: R.setdefault((x["prob"], x["rep"], x["variant"]), []).append(x)
    # 公平比较:只看三个 variant 都评到了的错误样本(都有参考解答)
    keys = sorted({(p, r) for (k, p, r, v) in C if k == "wrong"})
    common = [kr for kr in keys if all(("wrong",) + kr + (v,) in C for v in VARIANTS)]

    def succ(kr, v):            # 每条错误 rollout 的修复成功率;critic 没给出可用定位 = 0
        xs = R.get(kr + (v,), [])
        return sum(x["correct"] for x in xs) / N_REPAIR if xs else 0.0

    print(f"三个 critic 都评到的错误 rollout:{len(common)} 条(都有参考解答)")
    print(f"| critic | 给出定位 | NONE | 靠引用定位 | 泄露答案 | 首错相对位置 | **修复成功率** | 修复长度中位 | 修复写不完 | critic 输出 token |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    S = {}
    for v in VARIANTS:
        cs = [C[("wrong",) + kr + (v,)] for kr in common]
        xs = [x for kr in common for x in R.get(kr + (v,), [])]
        S[v] = [succ(kr, v) for kr in common]
        pos = [c["seg"] / max(c["n_seg"] - 1, 1) for c in cs if c["parsed"]]
        f = lambda a: f"{100*a:.1f}%"
        print(f"| {v} | {f(sum(c['parsed'] for c in cs)/len(cs))} | {f(sum(c['none'] for c in cs)/len(cs))} "
              f"| {f(sum(c['by_quote'] for c in cs if c['parsed'])/max(1,sum(c['parsed'] for c in cs)))} "
              f"| {f(sum(c['leak'] for c in cs if c['parsed'])/max(1,sum(c['parsed'] for c in cs)))} "
              f"| {statistics.mean(pos):.2f} | **{f(statistics.mean(S[v]))}** "
              f"| {statistics.median([x['new_tokens'] for x in xs]) if xs else 0:.0f} "
              f"| {f(sum(x['capped'] for x in xs)/max(1,len(xs)))} "
              f"| {statistics.mean(c['critic_tokens'] for c in cs):.0f} |")
    print()
    for v in ("B", "B2"):
        m, lo, hi = boot_diff(S[v], S["A"])
        print(f"  {v} − A 修复成功率: {100*m:+.1f} 个百分点  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]  (按 rollout 配对 bootstrap)")
    # 修对的修复里,长度有没有变短
    for v in VARIANTS:
        ok = [x["new_tokens"] for kr in common for x in R.get(kr + (v,), []) if x["correct"]]
        if ok: print(f"  {v}: 修对的修复长度中位 {statistics.median(ok):.0f}(n={len(ok)})")
    # 没参考解答的题:只能用 A
    noref = [kr for kr in keys if kr not in set(common) and ("wrong",) + kr + ("A",) in C]
    if noref:
        print(f"\n  没有参考解答的错误 rollout {len(noref)} 条,A 的修复成功率 {100*statistics.mean(succ(kr,'A') for kr in noref):.1f}%"
              f"(这部分在训练里只能退回 A)")
    # 误报:把答对的解答交给 critic(提示词里同样说「答案是错的」),看它多常硬找一个错
    print(f"\n误报检验(答对的 rollout,critic 仍被告知「答案错了」;看相对差异):")
    for v in VARIANTS:
        cs = [c for (k, p, r, vv), c in C.items() if k == "fp" and vv == v]
        if cs: print(f"  {v}: {len(cs)} 条,判出错误 {100*sum(c['parsed'] for c in cs)/len(cs):.1f}%  回答 NONE {100*sum(c['none'] for c in cs)/len(cs):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["rollout", "ref", "critic", "repair", "report"])
    a = ap.parse_args()
    {"rollout": stage_rollout, "ref": stage_ref, "critic": stage_critic,
     "repair": stage_repair, "report": stage_report}[a.stage]()
