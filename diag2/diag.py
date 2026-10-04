#!/usr/bin/env python3
"""Section 2 诊断实验(设置 B:与主实验一致)。
学生 = 主表 OPD 的 checkpoint(opd_1p7b 第 80 步,Qwen3-1.7B 非思考);teacher = Qwen3-4B-Instruct-2507;题目 = DAPO-Math-17k。
续写:teacher 与学生都在同一个 assistant 回复里接着写(各自的对话模板 + 学生前缀的原 token),温度 1.0、top-p 1.0,
预算 16384 - 前缀长度。判分与训练相同(opd/reward/math_reward.py,子进程运行)。分段与 CriticOPD 相同(按空行)。

critic 与最终方法一致(R4GTKL2K):多错误提示词(按顺序列出全部错误,至多 5 个),贪心解码,上限 2048,
被截断时去掉最后一条没写完的错误,去掉泄露答案的错误。区别只有一处:Section 2 默认不给参考答案
(提示词去掉「You are given the correct answer...」那句,user 消息去掉参考答案那行);给参考答案的
版本(与训练逐字一致)只用于 E5 附录与 E8。

阶段:
  rollouts <shard> <nsh>     学生在 N 道题上各做 1 次
  select                     判分,选 150 对 / 150 错(错的只取写完的,写满上限的没有原错误答案),定截断点
  cont <actor> <job> <shard> <nsh>
       job = e1                每个截断点续写 16 次
       job = e6                方法断点(最后一个错误所在段之后):teacher / 学生直接写,学生 + 通用提示,
                               学生 + 全部反馈,teacher + 全部反馈,各 8 次
       job = e7                带反馈重写的 25/50/75% 处(teacher 带/不带反馈,学生带/不带反馈),
                               与原轨迹在断点之后被丢弃的尾部 25/50/75% 处(都不带反馈),各 8 次
  grade <job>                判分(子进程)
  closing                    关闭点与 A/B/C 分类                               -> closing.json
  score <actor> <shard> <nsh>  E3:原轨迹逐 token logprob(teacher 用学生的对话模板,与训练打分相同)
  critic <shard> <nsh>       E4 截断解答 / E5 完整解答(不给与给参考答案)
  merge_critic               合并 critic 分片,算方法断点                       -> method_cut.json
  judge <shard> <nsh>        E8:Qwen3-8B(思考模式)逐条判断列出的错误是否属实
"""
import asyncio, collections, glob, json, math, os, random, re, statistics, sys, zlib
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402

OUT = os.environ.get("DIAG_OUT", "/home/kzhao2/Relay-OPD/diag2/out")
STUDENT = "/home/kzhao2/Relay-OPD/outputs/checkpoints/opd_1p7b/global_step_80/actor/huggingface"
TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
JUDGE = "/home/kzhao2/OPD/model/Qwen3-8B"
DATA = "/home/kzhao2/OPD/datasets/dapo-math-17k.parquet"
N_PROB, N_CORRECT, N_WRONG = (int(os.environ.get(k, d)) for k, d in (("DIAG_NPROB", 900), ("DIAG_NC", 150), ("DIAG_NW", 150)))
RLEN, MAXLEN = 16384, 20480
CRITIC_TOKENS, CRITIC_MAXLEN = 2048, 34817            # 与训练相同:CRITIC_OPD_CRITIC_TOKENS=2048,critic_maxlen=34817
STOP = [151643, 151645]
QS = (0.25, 0.5, 0.75)
# 通用提示:与方法同一个反馈模板(FEEDBACK_TMPL),只放一条不含内容的反馈
GENERIC = "Your solution contains an error. Check it and continue."

# ---- critic 提示词:最终方法的多错误提示词(CRITIC_SYS_MULTI),只去掉参考答案那句 ----
_GT_SENT = "You are given the correct answer. Use it as a reference to trace"
assert _GT_SENT in M.CRITIC_SYS_MULTI
CRITIC_FULL_GT = M.CRITIC_SYS_MULTI
CRITIC_FULL_NOGT = M.CRITIC_SYS_MULTI.replace(_GT_SENT, "Trace")
CRITIC_PARTIAL_NOGT = CRITIC_FULL_NOGT.replace(
    "A student has submitted a complete solution whose FINAL ANSWER IS WRONG.",
    "A student has written part of a solution, and the solution is not finished yet.").replace(
    "find the steps that contain genuine errors.\n\n",
    "find the steps that contain genuine errors. If every step so far is correct, reply SEGMENT: NONE.\n\n")
assert CRITIC_PARTIAL_NOGT.count("not finished yet") == 1 and CRITIC_PARTIAL_NOGT.count("every step so far") == 1


def jl(p): return [json.loads(l) for l in open(p)]
def jw(p, rows):
    with open(p + ".tmp", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)
def jlg(pat): return [x for f in sorted(glob.glob(pat)) for x in jl(f)]


def need_shards(pat, n):
    """正式运行(pipeline.sh 写了 out/expect.json)时核对分片是否齐全;缺片直接失败,下游 afterok 不会运行。"""
    if not os.path.exists(f"{OUT}/expect.json"): return
    miss = [k for k in range(n) if not os.path.exists(pat.format(k=k))]
    if miss: raise SystemExit(f"缺分片 {pat} {miss}")


def toks():
    from transformers import AutoTokenizer
    return (AutoTokenizer.from_pretrained(STUDENT), AutoTokenizer.from_pretrained(TEACHER))


def prompt_ids(tok, msgs, student):
    kw = {"enable_thinking": False} if student else {}
    return tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False)


def final_answer(text):
    """学生原轨迹的最终答案:最后一个 \\boxed{...}(括号配平),否则最后一个 Answer: 行。"""
    i = text.rfind("\\boxed{")
    if i >= 0:
        j, d = i + 7, 1
        while j < len(text) and d:
            d += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if d == 0: return text[i + 7:j - 1].strip()
    m = re.findall(r"Answer:\s*(.+)", text)
    return m[-1].strip().strip("$").strip() if m else None


def llm(path, n_seq=256, maxlen=MAXLEN):
    from vllm import LLM
    return LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=maxlen,
               enable_prefix_caching=True, max_num_seqs=n_seq)


# ------------------------------------------------------------------ rollouts / select
def stage_rollouts(shard, nsh):
    import pandas as pd
    from vllm import SamplingParams
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(DATA)
    rows = random.Random(2026).sample(range(len(df)), N_PROB)[shard::nsh]
    ts, _ = toks()
    reqs, meta = [], []
    for r in rows:
        msgs = [dict(m) for m in df.iloc[r]["prompt"]]
        ids = prompt_ids(ts, msgs, True)
        reqs.append({"prompt_token_ids": ids})
        meta.append({"row": int(r), "problem": msgs[-1]["content"], "messages": msgs,
                     "reference_answer": str(df.iloc[r]["reward_model"]["ground_truth"]), "s_prompt_ids": ids})
    sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=RLEN, stop_token_ids=STOP, seed=1000 * shard + i)
           for i in range(len(reqs))]
    outs = llm(STUDENT).generate(reqs, sps)
    res = []
    for m, o in zip(meta, outs):
        c = o.outputs[0]
        ids = [t for t in c.token_ids if t not in STOP]
        res.append({**m, "resp_ids": ids, "response": c.text, "capped": c.finish_reason == "length"})
    jw(f"{OUT}/raw_rollouts_{shard}.jsonl", res)


async def _grade_many(pairs, timeout=30):
    pool = M._GraderPool.get()
    sem = asyncio.Semaphore(256)
    async def one(t, g):
        if g is None: return None
        async with sem:
            try: return bool(await asyncio.wait_for(pool.grade(t, g, timeout), timeout=240))
            except Exception: return False
    return await asyncio.gather(*[one(t, g) for t, g in pairs])


def grade_many(pairs):
    return asyncio.run(_grade_many(pairs))


def cut_points(tok, resp_ids, text, n_cut=7):
    segs = M.segment(text)
    if len(segs) < 2: return segs, [(0, 0)]
    ends, pos = [], 0
    for i, s in enumerate(segs[:-1]):
        j = text.find(s, pos); pos = j + len(s) if j >= 0 else pos + len(s); ends.append((i + 1, pos))
    L = len(text); chosen = []
    for q in range(1, n_cut + 1):
        tgt = L * q / (n_cut + 1)
        k = min(ends, key=lambda e: abs(e[1] - tgt))[0]
        if k not in chosen: chosen.append(k)
    cuts = [(0, 0)]
    for k in sorted(chosen):
        nt = M._tok_prefix_for_text(tok, resp_ids, "\n\n".join(segs[:k]))
        if nt > cuts[-1][1] and nt < len(resp_ids): cuts.append((k, nt))
    return segs, cuts


def stage_select():
    ts, tt = toks()
    ext = os.environ.get("DIAG_EXTEND_FROM")            # 扩充批次:从主批次的同一批 rollout 里取主批次没用到的错误轨迹
    if os.path.exists(f"{OUT}/expect.json"): need_shards(f"{ext or OUT}/raw_rollouts_{{k}}.jsonl", json.load(open(f"{OUT}/expect.json"))["rollouts"])
    raw = jlg(f"{ext or OUT}/raw_rollouts_*.jsonl")
    if ext:
        used = {r["problem_id"] for r in jl(f"{ext}/rollouts.jsonl")}
        raw = [r for r in raw if r["row"] not in used]
    ok = grade_many([(r["response"], r["reference_answer"]) for r in raw])
    for r, c in zip(raw, ok): r["correct"] = bool(c) and not r["capped"]
    json.dump({str(r["row"]): r["correct"] for r in raw}, open(f"{OUT}/raw_correct.json", "w"))   # 估计训练 batch 里错误 token 的占比
    json.dump({str(r["row"]): r["capped"] for r in raw}, open(f"{OUT}/raw_capped.json", "w"))
    rng = random.Random(7)
    cor = [r for r in raw if r["correct"]]
    wro = [r for r in raw if not r["correct"] and not r["capped"]]      # 写满上限的没有原错误答案,图 2(b) 对它们无定义
    pick = rng.sample(cor, min(N_CORRECT, len(cor))) + rng.sample(wro, min(N_WRONG, len(wro)))
    res = []
    off = 1000 if ext else 0                             # 扩充批次的 rollout_id 从 1000 起,合并时不冲突
    for i, r in enumerate(pick):
        segs, cuts = cut_points(ts, r["resp_ids"], r["response"])
        res.append({"rollout_id": off + i, "problem_id": r["row"], "problem": r["problem"], "messages": r["messages"],
                    "reference_answer": r["reference_answer"], "response": r["response"], "segments": segs,
                    "final_answer": final_answer(r["response"]), "correct": r["correct"], "capped": r["capped"],
                    "cut_points": [k for k, _ in cuts], "cut_tokens": [n for _, n in cuts], "resp_ids": r["resp_ids"],
                    "s_prompt_ids": r["s_prompt_ids"], "t_prompt_ids": prompt_ids(tt, r["messages"], False)})
    jw(f"{OUT}/rollouts.jsonl", res)
    print(f"[select] 生成 {len(raw)} 条:对 {len(cor)},错且写完 {len(wro)},写满上限 {sum(r['capped'] for r in raw)}(不进入);"
          f"选 {sum(r['correct'] for r in res)} 对 + {sum(not r['correct'] for r in res)} 错;平均截断点 "
          f"{statistics.mean(len(r['cut_points']) for r in res):.1f}")


# ------------------------------------------------------------------ 续写
def rewrites():
    """E6 里学生带全部反馈的重写(每条轨迹取前 2 次),{rid: [ids, ids]}。"""
    rw = {}
    for x in jlg(f"{OUT}/cont_student_e6_*.jsonl"):
        if x["cond"] == "critique": rw[x["rid"]] = [s["ids"] for s in x["samples"][:2]]
    return rw


def items_for(job):
    R = jl(f"{OUT}/rollouts.jsonl")
    if job == "e1":
        return [{"rid": r["rollout_id"], "cut": c, "cond": "direct", "ntok": r["cut_tokens"][c], "fb": None, "n": 16}
                for r in R for c in range(len(r["cut_points"]))]
    MC = {int(k): v for k, v in json.load(open(f"{OUT}/method_cut.json")).items() if v.get("fb")}
    if job == "e6":
        its = []
        for rid, m in sorted(MC.items()):
            base = {"rid": rid, "ntok": m["keep_tok"], "n": 8}
            its += [{**base, "cond": "direct", "fb": None, "actor": "teacher"},
                    {**base, "cond": "direct", "fb": None, "actor": "student"},
                    {**base, "cond": "generic", "fb": GENERIC, "actor": "student"},
                    {**base, "cond": "critique", "fb": m["fb"], "actor": "student"},
                    {**base, "cond": "critique", "fb": m["fb"], "actor": "teacher"}]
        return its
    if job == "e7":
        Rr = {r["rollout_id"]: r for r in R}
        RW = rewrites()
        its = []
        for rid, m in sorted(MC.items()):
            for j, ids in enumerate(RW.get(rid, [])):
                for q in QS:
                    nrw = int(len(ids) * q)
                    if nrw == 0: continue
                    base = {"rid": rid, "ntok": m["keep_tok"], "rw": j, "nrw": nrw, "q": q, "n": 8}
                    # 带反馈:生成时的状态;不带反馈:训练序列(保留前缀 + 重写,反馈去掉)
                    its += [{**base, "cond": "rw_fb", "fb": m["fb"], "actor": "teacher"},
                            {**base, "cond": "rw_fb", "fb": m["fb"], "actor": "student"},
                            {**base, "cond": "rw_nofb", "fb": None, "actor": "teacher"},
                            {**base, "cond": "rw_nofb", "fb": None, "actor": "student"}]
            n0, L = m["keep_tok"], len(Rr[rid]["resp_ids"])
            if L - n0 >= 16:                                                # 对照:断点之后被丢弃的原轨迹尾部
                for q in QS:
                    base = {"rid": rid, "ntok": n0 + int((L - n0) * q), "q": q, "fb": None, "n": 8, "cond": "tail"}
                    its += [{**base, "actor": "teacher"}, {**base, "actor": "student"}]
        return its
    if job == "e10":                                                       # 以错误为锚:每个不同的截断位置续写 16 次
        R = {r["rollout_id"]: r for r in R}
        its = []
        for rid, an in sorted(json.load(open(f"{OUT}/e10_anchors.json")).items(), key=lambda kv: int(kv[0])):
            rid = int(rid); L = len(R[rid]["resp_ids"])
            for nt in sorted({v[side] for v in an.values() for side in ("before", "after")}):
                if nt < L: its.append({"rid": rid, "cut": None, "cond": "anchor", "ntok": nt, "fb": None, "n": 16})
        return its
    raise ValueError(job)


def state_ids(ts, r, it, RW):
    """状态 = 学生原轨迹前 ntok 个 token [+ 反馈模板] [+ 重写的前 nrw 个 token];反馈的拼法与训练逐字相同。"""
    ids = list(r["resp_ids"][:it["ntok"]])
    if it["fb"]:
        ids = ts.encode(ts.decode(ids, skip_special_tokens=True) + M.FEEDBACK_TMPL.format(fb=it["fb"]), add_special_tokens=False)
    if it.get("rw") is not None:
        ids = ids + list(RW[it["rid"]][it["rw"]][:it["nrw"]])
    return ids


def stage_cont(actor, job, shard, nsh):
    from vllm import SamplingParams
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    ts, tt = toks()
    RW = rewrites() if job == "e7" else {}
    its = [it for it in items_for(job) if it.get("actor", actor) == actor][shard::nsh]
    reqs, sps, keep = [], [], []
    for i, it in enumerate(its):
        r = R[it["rid"]]
        p = r["s_prompt_ids"] if actor == "student" else r["t_prompt_ids"]
        ids = state_ids(ts, r, it, RW)
        budget = max(256, RLEN - it["ntok"] - (it.get("nrw") or 0))       # 与训练相同:重写预算 = 16384 - 保留前缀
        budget = min(budget, MAXLEN - len(p) - len(ids) - 8)
        if budget < 64: continue
        reqs.append({"prompt_token_ids": p + ids}); keep.append(it)
        sps.append(SamplingParams(n=it["n"], temperature=1.0, top_p=1.0, max_tokens=budget, stop_token_ids=STOP,
                                  seed=zlib.crc32(f"{job}|{actor}|{shard}|{i}".encode()) % (2 ** 31)))
    outs = llm(STUDENT if actor == "student" else TEACHER).generate(reqs, sps)
    res = []
    for it, o in zip(keep, outs):
        ss = [{"text": c.text, "truncated": c.finish_reason == "length"} for c in o.outputs]
        if job == "e6" and actor == "student" and it["cond"] == "critique":      # 只有 E7 要用的重写才存 token
            for s, c in zip(ss[:2], o.outputs[:2]): s["ids"] = [t for t in c.token_ids if t not in STOP]
        res.append({**it, "actor": actor, "samples": ss})
    jw(f"{OUT}/cont_{actor}_{job}_{shard}.jsonl", res)
    print(f"[cont {actor} {job}] shard {shard}: {len(res)} 个状态")


def stage_grade(job):
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    from transformers import AutoTokenizer
    ts = AutoTokenizer.from_pretrained(STUDENT)
    if os.path.exists(f"{OUT}/expect.json"):
        for actor, n in json.load(open(f"{OUT}/expect.json"))[job].items(): need_shards(f"{OUT}/cont_{actor}_{job}_{{k}}.jsonl", n)
    RW = rewrites() if job == "e7" else {}
    rows = jlg(f"{OUT}/cont_*_{job}_[0-9]*.jsonl")
    pairs = []
    for x in rows:
        r = R[x["rid"]]
        pre = ts.decode(state_ids(ts, r, {**x, "fb": None}, RW), skip_special_tokens=True)   # 判分文本不含反馈
        for s in x["samples"]:
            body = pre + s["text"]
            pairs.append((body, r["reference_answer"]))
            if job == "e1": pairs.append((body, r["final_answer"]))
    res = grade_many(pairs)
    k = 0
    for x in rows:
        for s in x["samples"]:
            s["correct"] = bool(res[k]) and not s["truncated"]; k += 1
            if job == "e1": s["matches_original_answer"] = res[k]; k += 1
    jw(f"{OUT}/graded_{job}.jsonl", rows)
    print(f"[grade {job}] {len(rows)} 个状态,{sum(len(x['samples']) for x in rows)} 次续写")


# ------------------------------------------------------------------ 关闭点
def stage_closing():
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    if os.path.exists(f"{OUT}/expect.json"):                   # 每条轨迹的每个截断点都要有 teacher 和学生的续写
        have = {(x["rid"], x["actor"], x["cut"]) for x in jl(f"{OUT}/graded_e1.jsonl")}
        miss = [(rid, a, c) for rid, r in R.items() for a in ("teacher", "student") for c in range(len(r["cut_points"])) if (rid, a, c) not in have]
        if miss: raise SystemExit(f"E1 缺 {len(miss)} 个状态,例如 {miss[:3]}")
    G = jl(f"{OUT}/graded_e1.jsonl")
    V = collections.defaultdict(dict)
    for x in G: V[x["rid"]][(x["actor"], x["cut"])] = [s["correct"] for s in x["samples"]]
    A, B, Cc = {}, [], []
    for rid, r in R.items():
        if r["correct"]: continue
        n = len(r["cut_points"])
        vt8 = [statistics.mean(V[rid][("teacher", c)][:8]) for c in range(n)]
        close = next((c for c in range(n) if vt8[c] < 0.5), None)
        if close is None: B.append(rid)
        elif close == 0: Cc.append(rid)
        else:
            prev = close - 1
            A[rid] = {"close": close, "close_tok": r["cut_tokens"][close], "close_seg": r["cut_points"][close],
                      "prev_seg": r["cut_points"][prev], "vt8": vt8}
    json.dump({"A": A, "B": B, "C": Cc}, open(f"{OUT}/closing.json", "w"))
    print(f"[closing] 错误轨迹 {sum(not r['correct'] for r in R.values())} 条:A {len(A)},B {len(B)},C {len(Cc)}")


# ------------------------------------------------------------------ E3 打分
def stage_score(actor, shard, nsh):
    """逐 token logprob。用 transformers 直接前向(bf16),lm_head 分块做 log-softmax:
    vLLM 的 prompt_logprobs 在 1.6 万 token 的序列上会把引擎撑死(EngineDeadError)。"""
    import torch
    from transformers import AutoModelForCausalLM
    R = jl(f"{OUT}/rollouts.jsonl")
    rows = [r for r in R if not r["correct"]][shard::nsh]
    path = STUDENT if actor == "student" else TEACHER
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    res = []
    with torch.no_grad():
        for r in rows:
            # 两边都用学生的对话模板:训练里 teacher 就是在学生的 prompt_ids + response_ids 上打分的
            P, n = len(r["s_prompt_ids"]), len(r["resp_ids"])
            ids = torch.tensor([r["s_prompt_ids"] + r["resp_ids"]], device="cuda")
            h = model.model(input_ids=ids).last_hidden_state[0, P - 1:P + n - 1]      # 第 P-1.. 个位置预测回复的每个 token
            tgt = ids[0, P:P + n]
            lp = []
            for i in range(0, n, 2048):
                lg = model.lm_head(h[i:i + 2048]).float()
                lp.append(torch.log_softmax(lg, -1).gather(1, tgt[i:i + 2048, None])[:, 0])
            res.append({"rid": r["rollout_id"], "lp": [round(x, 4) for x in torch.cat(lp).tolist()]})
            del h, ids
    jw(f"{OUT}/score_{actor}_{shard}.jsonl", res)
    print(f"[score {actor}] shard {shard}: {len(res)} 条")


# ------------------------------------------------------------------ E4 / E5 critic
def numbered(segs): return "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))


def critic_user(problem, segs, gt=None):
    """与训练的 user 消息逐字一致;不给参考答案时只去掉参考答案那一行。"""
    return (f"Problem:\n{problem}\n\n" + (f"Correct answer (reference only -- never reveal it): {gt}\n\n" if gt else "")
            + f"Student's solution, split into numbered segments:\n\n{numbered(segs)}")


def process_critic(text, ntok, segs, gt, stu):
    """与训练相同的后处理:截断 -> 去掉没写完的最后一块;解析多错误;去掉泄露答案的错误。"""
    trunc = ntok >= CRITIC_TOKENS
    if trunc: text = M.drop_incomplete_tail(text)
    raw = M.parse_critic_multi(text, segs)
    errs = M.drop_leaky(raw, gt, stu)
    none = not raw and bool(M._RE_NONE.search(text))
    return {"truncated": trunc, "none": none, "valid": none or bool(raw),
            "errs_raw": [list(e) for e in raw], "errs": [list(e) for e in errs]}


def stage_critic(shard, nsh):
    from vllm import SamplingParams
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    A = {int(k): v for k, v in json.load(open(f"{OUT}/closing.json"))["A"].items()}
    _, tt = toks()
    jobs = []
    for rid, a in sorted(A.items()):
        r = R[rid]; segs = r["segments"]
        for rel in (-2, -1, 0, 1, 2, 3, 4):
            c = a["close"] + rel
            if c < 1 or c >= len(r["cut_points"]): continue
            k = r["cut_points"][c]
            jobs.append(("partial", rid, rel, segs[:k], CRITIC_PARTIAL_NOGT, None))
        jobs.append(("full_nogt", rid, None, segs, CRITIC_FULL_NOGT, None))
        jobs.append(("full_gt", rid, None, segs, CRITIC_FULL_GT, r["reference_answer"]))
    jobs = jobs[shard::nsh]
    reqs = []
    for kind, rid, rel, ss, sysp, gt in jobs:
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": critic_user(R[rid]["problem"], ss, gt)}]
        reqs.append({"prompt_token_ids": tt.encode(tt.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                                                   add_special_tokens=False)})
    outs = llm(TEACHER, maxlen=CRITIC_MAXLEN).generate(reqs, SamplingParams(temperature=0.0, max_tokens=CRITIC_TOKENS))
    res = collections.defaultdict(list)
    for (kind, rid, rel, ss, _, _), o in zip(jobs, outs):
        r = R[rid]; c = o.outputs[0]
        shown = "\n\n".join(ss)
        res[kind].append({"rid": rid, "rel": rel, "n_seg_shown": len(ss), "text": c.text, "ntok": len(c.token_ids),
                          **process_critic(c.text, len(c.token_ids), ss, r["reference_answer"], shown)})
    for k in ("partial", "full_nogt", "full_gt"): jw(f"{OUT}/critic_{k}_{shard}.jsonl", res.get(k, []))


# ---- 对照:附录 C 原来的单错误提示词(找「从哪一步起正确答案再也到不了」),不给参考答案;只用于 E4/E5 的定位对照
_RULES1 = M._GT_RULES
SINGLE_FULL_NOGT = ("You are a meticulous math teacher. A student has submitted a complete solution whose FINAL ANSWER IS "
                    "WRONG. Trace the student's work forward and find the EARLIEST step after which the correct answer can "
                    "no longer be reached.\n\n" + _RULES1)
SINGLE_PARTIAL_NOGT = ("You are a meticulous math teacher. A student has written part of a solution, and the solution is not "
                       "finished yet. Trace the student's work forward and find the EARLIEST step after which the correct "
                       "answer can no longer be reached. If every step so far is correct and the correct answer can still be "
                       "reached, reply SEGMENT: NONE.\n\n" + _RULES1)


def stage_critic_single(shard, nsh):
    from vllm import SamplingParams
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    A = {int(k): v for k, v in json.load(open(f"{OUT}/closing.json"))["A"].items()}
    _, tt = toks()
    jobs = []
    for rid, a in sorted(A.items()):
        r = R[rid]; segs = r["segments"]
        for rel in (-2, -1, 0, 1, 2, 3, 4):
            c = a["close"] + rel
            if c < 1 or c >= len(r["cut_points"]): continue
            jobs.append(("single_partial", rid, rel, segs[:r["cut_points"][c]], SINGLE_PARTIAL_NOGT))
        jobs.append(("single_full", rid, None, segs, SINGLE_FULL_NOGT))
    jobs = jobs[shard::nsh]
    reqs = [{"prompt_token_ids": tt.encode(tt.apply_chat_template(
        [{"role": "system", "content": j[4]}, {"role": "user", "content": critic_user(R[j[1]]["problem"], j[3])}],
        add_generation_prompt=True, tokenize=False), add_special_tokens=False)} for j in jobs]
    outs = llm(TEACHER, maxlen=CRITIC_MAXLEN).generate(reqs, SamplingParams(temperature=0.0, max_tokens=512))
    res = collections.defaultdict(list)
    for (kind, rid, rel, ss, _), o in zip(jobs, outs):
        t = o.outputs[0].text; none = bool(M._RE_NONE.search(t))
        idx, fb = (None, None) if none else M.parse_critic(t, ss)
        res[kind].append({"rid": rid, "rel": rel, "text": t, "none": none, "valid": none or idx is not None,
                          "errs_raw": [[idx, "", fb]] if idx is not None else [], "errs": [[idx, "", fb]] if idx is not None else []})
    for k in ("single_partial", "single_full"): jw(f"{OUT}/critic_{k}_{shard}.jsonl", res.get(k, []))


def stage_merge_critic():
    if os.path.exists(f"{OUT}/expect.json"): need_shards(f"{OUT}/critic_full_nogt_{{k}}.jsonl", json.load(open(f"{OUT}/expect.json"))["critic"])
    for k in ("partial", "full_nogt", "full_gt"):
        jw(f"{OUT}/critic_{k}.jsonl", jlg(f"{OUT}/critic_{k}_[0-9]*.jsonl"))
    # 方法断点:与训练相同 —— 全部(泄露过滤后的)错误,在最后一个错误所在段之后断开,反馈按 kerr_feedback 编号拼接
    ts, _ = toks()
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    MC = {}
    for x in jl(f"{OUT}/critic_full_nogt.jsonl"):
        r = R[x["rid"]]
        if not x["errs"]:
            MC[x["rid"]] = {"fb": None, "reason": "none" if x["none"] else ("leak" if x["errs_raw"] else "invalid")}
            continue
        sel = M.kerr_select([tuple(e) for e in x["errs"]], "last")
        last = sel[-1][0]
        MC[x["rid"]] = {"fb": M.kerr_feedback(sel), "first": sel[0][0], "first_raw": x["errs_raw"][0][0], "last": last,
                        "n_err": len(sel), "keep_seg": last + 1,
                        "keep_tok": M._tok_prefix_for_text(ts, r["resp_ids"], "\n\n".join(r["segments"][:last + 1]))}
    json.dump(MC, open(f"{OUT}/method_cut.json", "w"))
    print(f"[merge_critic] 方法断点 {sum(1 for v in MC.values() if v['fb'])} 条;没有可用错误 "
          f"{collections.Counter(v['reason'] for v in MC.values() if not v['fb'])}")


# ------------------------------------------------------------------ E8 逐条判断列出的错误是否属实
JUDGE_SYS = (
    "You are an expert mathematics grader. You are given a math problem, its correct answer, a student's "
    "solution split into numbered segments, and a list of errors that a teacher claims to have found in the "
    "solution, in order of appearance. For every listed error, decide whether the claim is right.\n\n"
    "Labels:\n"
    "NEW_ERROR - the flagged step contains a genuine mistake of its own (a wrong formula, wrong logic, an "
    "arithmetic slip, a misread condition, an unjustified claim) that would still be wrong even if every "
    "earlier step were correct.\n"
    "PROPAGATED - the reasoning in the flagged step is valid in itself; it is wrong only because it uses a "
    "value, expression or conclusion produced by an earlier mistake.\n"
    "RESTATEMENT - the flagged step only restates or reports a wrong result that was already reached earlier "
    "(for example the final answer line or a summary), with no new reasoning.\n"
    "NOT_AN_ERROR - the flagged step is actually correct.\n\n"
    "Reply with exactly one line per listed error and nothing else, in this format:\n"
    "ERROR 1: <LABEL>\nERROR 2: <LABEL>")
JLABELS = ("NEW_ERROR", "PROPAGATED", "RESTATEMENT", "NOT_AN_ERROR")


def judge_items():
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    kinds = os.environ.get("DIAG_JUDGE_KINDS", "full_nogt,full_gt").split(",")
    its = []
    for kind in kinds:
        for x in jl(f"{OUT}/critic_{kind}.jsonl"):
            if not x["errs_raw"]: continue
            r = R[x["rid"]]
            segs = r["segments"][:x["n_seg_shown"]] if kind == "partial" else r["segments"]
            what = ("Student's partial solution (not finished yet), split into numbered segments" if kind == "partial"
                    else "Student's solution, split into numbered segments")
            errs = "\n\n".join(f"ERROR {i + 1}\nSEGMENT: {e[0]}\nQUOTE: {e[1]}\nTEACHER'S NOTE: {e[2]}"
                               for i, e in enumerate(x["errs_raw"]))
            user = (f"Problem:\n{r['problem']}\n\nCorrect answer: {r['reference_answer']}\n\n"
                    f"{what}:\n\n{numbered(segs)}\n\nErrors claimed by the teacher:\n\n{errs}")
            its.append({"kind": kind, "rid": x["rid"], "rel": x.get("rel"), "n": len(x["errs_raw"]), "user": user})
    return its


# ---- E9:不看 critic 的输出,由判定模型独立找出第一个错误(和全部错误),错误区间的起点因此不依赖被检验的 critic
JUDGE_FIRST_SYS = (
    "You are an expert mathematics grader. You are given a math problem, its correct answer, and a student's "
    "solution split into numbered segments. The student's final answer is wrong. Read the segments in order and "
    "find every segment that contains a genuine mistake of its own: a wrong formula, wrong logic, an arithmetic slip, "
    "a misread condition, or an unjustified claim. Do not flag an exploratory attempt that the student later abandons "
    "or corrects, a longer route than necessary, or notation that does not change the mathematics. Do not flag a step "
    "only because it reuses a wrong value produced by an earlier mistake.\n\n"
    "Reply with exactly two lines and nothing else:\n"
    "FIRST: <index of the first segment with a genuine mistake, or NONE>\n"
    "ALL: <indices of all segments with a genuine mistake, in increasing order, separated by commas, or NONE>")


def stage_judge_first(shard, nsh):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tj = AutoTokenizer.from_pretrained(JUDGE)
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    A = sorted(int(k) for k in json.load(open(f"{OUT}/closing.json"))["A"])[shard::nsh]
    reqs, keep = [], []
    for rid in A:
        r = R[rid]
        user = (f"Problem:\n{r['problem']}\n\nCorrect answer: {r['reference_answer']}\n\n"
                f"Student's solution, split into numbered segments:\n\n{numbered(r['segments'])}")
        ids = tj.encode(tj.apply_chat_template([{"role": "system", "content": JUDGE_FIRST_SYS}, {"role": "user", "content": user}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=True), add_special_tokens=False)
        if len(ids) > 40960 - 4096: continue
        reqs.append({"prompt_token_ids": ids}); keep.append((rid, min(16384, 40960 - len(ids) - 16)))
    sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mt, seed=11) for _, mt in keep]
    outs = llm(JUDGE, 128, 40960).generate(reqs, sps)
    res = []
    for (rid, _), o in zip(keep, outs):
        ans = o.outputs[0].text.split("</think>")[-1]
        n = len(R[rid]["segments"])
        mf = re.search(r"FIRST:\s*(\d+|NONE)", ans, re.I); ma = re.search(r"ALL:\s*([^\n]*)", ans, re.I)
        first = int(mf.group(1)) if mf and mf.group(1).isdigit() and int(mf.group(1)) < n else None
        alls = sorted({int(t) for t in re.findall(r"\d+", ma.group(1)) if int(t) < n}) if ma else []
        if first is None and alls: first = alls[0]
        res.append({"rid": rid, "first": first, "all": alls, "none": bool(mf and mf.group(1).upper() == "NONE"),
                    "truncated": o.outputs[0].finish_reason == "length", "answer": ans[-600:]})
    jw(f"{OUT}/judge_first_{shard}.jsonl", res)


def stage_anchors():
    """E10 的锚点:独立标注的第一个错误、critic 列出的第一条和最后一条错误,各在错误段之前和之后截断。"""
    ts, _ = toks()
    need_shards(f"{OUT}/judge_first_{{k}}.jsonl", 2)
    R = {r["rollout_id"]: r for r in jl(f"{OUT}/rollouts.jsonl")}
    JF = {x["rid"]: x for x in jlg(f"{OUT}/judge_first_[0-9]*.jsonl")}
    CR = {x["rid"]: x for x in jl(f"{OUT}/critic_full_nogt.jsonl")}
    AN = {}
    for rid in sorted(int(k) for k in json.load(open(f"{OUT}/closing.json"))["A"]):
        r = R[rid]; segs = r["segments"]; an = {}
        tok = lambda k: M._tok_prefix_for_text(ts, r["resp_ids"], "\n\n".join(segs[:k])) if k > 0 else 0
        for name, seg in (("judge_first", JF.get(rid, {}).get("first")),
                          ("critic_first", CR[rid]["errs_raw"][0][0] if rid in CR and CR[rid]["errs_raw"] else None),
                          ("critic_last", CR[rid]["errs"][-1][0] if rid in CR and CR[rid]["errs"] else None)):
            if seg is None: continue
            an[name] = {"seg": seg, "before": tok(seg), "after": tok(seg + 1)}
        AN[rid] = an
    json.dump(AN, open(f"{OUT}/e10_anchors.json", "w"))
    print(f"[anchors] {len(AN)} 条;有独立第一个错误 {sum('judge_first' in a for a in AN.values())}")


def stage_judge(shard, nsh):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tj = AutoTokenizer.from_pretrained(JUDGE)
    its = judge_items()[shard::nsh]
    reqs, keep = [], []
    for it in its:
        msgs = [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": it["user"]}]
        ids = tj.encode(tj.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True),
                        add_special_tokens=False)
        if len(ids) > 40960 - 4096: continue
        reqs.append({"prompt_token_ids": ids}); keep.append((it, min(16384, 40960 - len(ids) - 16)))
    sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mt, seed=7) for _, mt in keep]   # Qwen3 思考模式推荐参数
    outs = llm(JUDGE, 128, 40960).generate(reqs, sps)
    res = []
    for (it, _), o in zip(keep, outs):
        t = o.outputs[0].text; ans = t.split("</think>")[-1]
        lab = {int(i): l for i, l in re.findall(r"ERROR\s*(\d+)\s*:\s*(" + "|".join(JLABELS) + r")", ans)}
        res.append({"kind": it["kind"], "rid": it["rid"], "rel": it.get("rel"), "n": it["n"], "labels": [lab.get(i + 1) for i in range(it["n"])],
                    "truncated": o.outputs[0].finish_reason == "length", "answer": ans[-2000:]})
    jw(f"{OUT}/judge{os.environ.get('DIAG_JUDGE_TAG', '')}_{shard}.jsonl", res)


if __name__ == "__main__":
    a = sys.argv[1:]
    st = a[0]
    if st == "rollouts": stage_rollouts(int(a[1]), int(a[2]))
    elif st == "select": stage_select()
    elif st == "cont": stage_cont(a[1], a[2], int(a[3]), int(a[4]))
    elif st == "grade": stage_grade(a[1])
    elif st == "closing": stage_closing()
    elif st == "score": stage_score(a[1], int(a[2]), int(a[3]))
    elif st == "critic": stage_critic(int(a[1]), int(a[2]))
    elif st == "merge_critic": stage_merge_critic()
    elif st == "critic_single": stage_critic_single(int(a[1]), int(a[2]))
    elif st == "merge_single":
        for k in ("single_partial", "single_full"): jw(f"{OUT}/critic_{k}.jsonl", jlg(f"{OUT}/critic_{k}_[0-9]*.jsonl"))
    elif st == "judge": stage_judge(int(a[1]), int(a[2]))
    elif st == "judge_first": stage_judge_first(int(a[1]), int(a[2]))
    elif st == "anchors": stage_anchors()
    elif st == "merge_judge":
        if os.path.exists(f"{OUT}/expect.json"): need_shards(f"{OUT}/judge_{{k}}.jsonl", json.load(open(f"{OUT}/expect.json"))["judge"])
        jw(f"{OUT}/judge.jsonl", jlg(f"{OUT}/judge_[0-9]*.jsonl"))
    else: raise SystemExit(f"未知阶段 {st}")
    if st in ("rollouts", "cont", "score", "critic", "critic_single", "judge", "judge_first"):
        sys.stdout.flush(); os._exit(0)
