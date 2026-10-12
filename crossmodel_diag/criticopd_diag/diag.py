#!/usr/bin/env python3
"""Diagnosis of teacher reliability for on-policy distillation (Figures 3, 4 and 5(a) of the CriticOPD paper).

The logic is ported from the script that produced the paper figures; only paths, models and model-specific
settings (chat-template arguments, stop tokens, length budgets) come from a config file. Every stage reads and
writes files in `out_dir`, so stages can be rerun or sharded across GPUs.

Stages (run in this order; <shard> <nsh> split the work, use `0 1` on a single GPU):
  rollouts <shard> <nsh>          the student solves each problem once
  select                          grade, pick n_correct correct + n_wrong incorrect finished rollouts (main batch, Figures
                                  3 and 4), add all other incorrect finished rollouts (extension batch, Figure 5(a) only,
                                  as in the paper), set cut points
  cont <actor> e1 <shard> <nsh>   teacher and student each continue every prefix e1_samples times
  grade e1                        grade the continuations
  closing                         classify incorrect rollouts by where the teacher's success drops below 0.5
  critic <shard> <nsh>            the teacher critiques the student: truncated prefixes (no reference answer) and
                                  full solutions (without and with the reference answer)
  merge_critic                    merge critic shards and build the method cut used for Figure 5(a)
  judge <shard> <nsh>             the judge model labels every listed error of the truncated-prefix critiques
  cont <actor> e6 <shard> <nsh>   continuations at the method cut (direct, generic hint, critique)
  grade e6
  cont <actor> e7 <shard> <nsh>   continuations along the rewrite and along the discarded original tail
  grade e7
  plots                           write fig3/fig4/fig5a (.pdf/.png) and numbers.json into out_dir/figs
  check                           print the sample sizes and key numbers that should be reported
  pack                            out_dir/return.tar.gz: the results without continuation texts, to send back
  inspect                         (before everything, CPU only) print the prompts, stop ids and context lengths

Usage: python -m criticopd_diag.diag --config configs/<file>.yaml <stage> [args]
"""
import argparse, asyncio, collections, glob, json, os, random, re, shutil, statistics, sys, tarfile, time, zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import criticopd_diag.method as M   # noqa: E402

CFG = {}
QS = (0.25, 0.5, 0.75)
GENERIC = "Your solution contains an error. Check it and continue."

# ---- critic prompts: the multi-error prompt of the final method (M.CRITIC_SYS_MULTI); the no-answer versions only
# drop the reference-answer sentence, exactly as in the paper.
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


# ------------------------------------------------------------------ config and io
def load_config(path):
    if path.endswith(".json"):
        c = json.load(open(path))
    else:
        import yaml
        c = yaml.safe_load(open(path))
    d = {"judge": "Qwen/Qwen3-8B", "problems": "data/diag_problems_900.jsonl", "limit_problems": None,
         "student_chat_kwargs": {}, "teacher_chat_kwargs": {}, "judge_chat_kwargs": {"enable_thinking": True},
         "student_stop_token_ids": None, "teacher_stop_token_ids": None,
         "response_budget": 16384, "max_model_len": 20480, "critic_tokens": 2048, "critic_max_model_len": 34817,
         "judge_max_model_len": 40960, "judge_max_tokens": 16384,
         "n_correct": 150, "n_wrong": 150, "extend_incorrect": True, "n_cut": 7, "e1_samples": 16, "e6_samples": 8, "e7_samples": 8,
         "method_cut": {"critique": "full_gt", "append_answer": True, "filter_leaks": False},
         "judge_kinds": ["partial"], "grader_procs": 8,
         "vllm": {"dtype": "bfloat16", "gpu_memory_utilization": 0.85, "tensor_parallel_size": 1, "max_num_seqs": 256}}
    d.update(c)
    for k in ("student", "teacher", "out_dir"):
        assert d.get(k), f"config needs '{k}'"
    if not os.path.isabs(d["problems"]):
        d["problems"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d["problems"])
    os.environ.setdefault("CRITIC_OPD_GRADER_PROCS", str(d["grader_procs"]))
    return d


def OUT(): return CFG["out_dir"]


def say(msg):
    print(msg, flush=True)
    with open(f"{OUT()}/summary.log", "a") as f: f.write(f"{time.strftime('%F %T')} {msg}\n")
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def jlg(pat): return [x for f in sorted(glob.glob(pat)) for x in jl(f)]


def jw(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def toks():
    from transformers import AutoTokenizer
    ts, tt = AutoTokenizer.from_pretrained(CFG["student"]), AutoTokenizer.from_pretrained(CFG["teacher"])
    if ts.get_vocab() != tt.get_vocab():    # the teacher continues the student's token ids, as in OPD training
        raise SystemExit("the student and the teacher must share one tokenizer (same vocabulary), as OPD requires")
    return ts, tt


def stop_ids(which):
    """Stop tokens of the student or teacher: the config value if given, else eos ids of the tokenizer and of
    generation_config.json. They are stripped from stored token ids, exactly as the paper's script strips them."""
    if CFG.get(f"{which}_stop_token_ids"):
        return list(CFG[f"{which}_stop_token_ids"])
    from transformers import AutoTokenizer, GenerationConfig
    path = CFG[which]; ids = set()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.eos_token_id is not None: ids.add(tok.eos_token_id)
    try:
        e = GenerationConfig.from_pretrained(path).eos_token_id
        ids.update(e if isinstance(e, list) else [e] if e is not None else [])
    except Exception:
        pass
    return sorted(ids)


def prompt_ids(tok, msgs, student):
    kw = CFG["student_chat_kwargs"] if student else CFG["teacher_chat_kwargs"]
    return tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw), add_special_tokens=False)


def final_answer(text):
    """Final answer of the student's rollout: the last \\boxed{...} (balanced), else the last 'Answer:' line."""
    i = text.rfind("\\boxed{")
    if i >= 0:
        j, d = i + 7, 1
        while j < len(text) and d:
            d += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if d == 0: return text[i + 7:j - 1].strip()
    m = re.findall(r"Answer:\s*(.+)", text)
    return m[-1].strip().strip("$").strip() if m else None


def llm(path, n_seq=None, maxlen=None):
    from vllm import LLM
    v = CFG["vllm"]
    return LLM(model=path, dtype=v["dtype"], gpu_memory_utilization=v["gpu_memory_utilization"],
               tensor_parallel_size=v["tensor_parallel_size"], max_model_len=maxlen or CFG["max_model_len"],
               enable_prefix_caching=True, max_num_seqs=n_seq or v["max_num_seqs"])


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


# ------------------------------------------------------------------ rollouts / select
def problems():
    P = jl(CFG["problems"])
    return P[:CFG["limit_problems"]] if CFG["limit_problems"] else P


def stage_rollouts(shard, nsh):
    from vllm import SamplingParams
    ts, _ = toks(); stop = stop_ids("student")
    reqs, meta = [], []
    for p in problems()[shard::nsh]:
        msgs = p["messages"]; ids = prompt_ids(ts, msgs, True)
        reqs.append({"prompt_token_ids": ids})
        meta.append({"row": p["row"], "problem": msgs[-1]["content"], "messages": msgs,
                     "reference_answer": p["ground_truth"], "s_prompt_ids": ids})
    sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=CFG["response_budget"], stop_token_ids=stop,
                          seed=1000 * shard + i) for i in range(len(reqs))]
    outs = llm(CFG["student"]).generate(reqs, sps)
    res = []
    for m, o in zip(meta, outs):
        c = o.outputs[0]
        res.append({**m, "resp_ids": [t for t in c.token_ids if t not in stop], "response": c.text,
                    "capped": c.finish_reason == "length"})
    jw(f"{OUT()}/raw_rollouts_{shard}.jsonl", res)
    say(f"[rollouts] shard {shard}: {len(res)} rollouts, stop ids {stop}")


def cut_points(tok, resp_ids, text, n_cut):
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
    raw = jlg(f"{OUT()}/raw_rollouts_*.jsonl")
    assert len(raw) == len(problems()), f"rollouts incomplete: {len(raw)} of {len(problems())} (run every rollouts shard)"
    ok = grade_many([(r["response"], r["reference_answer"]) for r in raw])
    for r, c in zip(raw, ok): r["correct"] = bool(c) and not r["capped"]
    json.dump({str(r["row"]): r["correct"] for r in raw}, open(f"{OUT()}/raw_correct.json", "w"))
    json.dump({str(r["row"]): r["capped"] for r in raw}, open(f"{OUT()}/raw_capped.json", "w"))
    rng = random.Random(7)
    cor = [r for r in raw if r["correct"]]
    wro = [r for r in raw if not r["correct"] and not r["capped"]]      # capped rollouts have no original wrong answer
    pick = rng.sample(cor, min(CFG["n_correct"], len(cor))) + rng.sample(wro, min(CFG["n_wrong"], len(wro)))
    pick = [(i, "main", r) for i, r in enumerate(pick)]
    if CFG["extend_incorrect"]:      # extension batch: the remaining incorrect finished rollouts, ids from 1000
        used = {r["row"] for _, _, r in pick}
        rest = [r for r in wro if r["row"] not in used]
        pick += [(1000 + i, "ext", r) for i, r in enumerate(random.Random(7).sample(rest, len(rest)))]
    res = []
    for i, batch, r in pick:
        segs, cuts = cut_points(ts, r["resp_ids"], r["response"], CFG["n_cut"])
        res.append({"rollout_id": i, "batch": batch, "problem_id": r["row"], "problem": r["problem"], "messages": r["messages"],
                    "reference_answer": r["reference_answer"], "response": r["response"], "segments": segs,
                    "final_answer": final_answer(r["response"]), "correct": r["correct"], "capped": r["capped"],
                    "cut_points": [k for k, _ in cuts], "cut_tokens": [n for _, n in cuts], "resp_ids": r["resp_ids"],
                    "s_prompt_ids": r["s_prompt_ids"], "t_prompt_ids": prompt_ids(tt, r["messages"], False)})
    jw(f"{OUT()}/rollouts.jsonl", res)
    say(f"[select] {len(raw)} rollouts: {len(cor)} correct, {len(wro)} incorrect and finished, "
          f"{sum(r['capped'] for r in raw)} capped; picked {sum(r['correct'] for r in res)} correct + "
          f"{sum(not r['correct'] and r['batch'] == 'main' for r in res)} incorrect (main batch), "
          f"{sum(r['batch'] == 'ext' for r in res)} incorrect (extension batch); mean number of cuts {statistics.mean(len(r['cut_points']) for r in res):.1f}")


# ------------------------------------------------------------------ continuations
def rewrites():
    """The student's rewrites with the critique at the method cut (first 2 per rollout), {rid: [ids, ids]}."""
    rw = {}
    for x in jlg(f"{OUT()}/cont_student_e6_*.jsonl"):
        if x["cond"] == "critique": rw[x["rid"]] = [s["ids"] for s in x["samples"][:2]]
    return rw


def items_for(job):
    R = jl(f"{OUT()}/rollouts.jsonl")
    if job == "e1":
        return [{"rid": r["rollout_id"], "cut": c, "cond": "direct", "ntok": r["cut_tokens"][c], "fb": None, "n": CFG["e1_samples"]}
                for r in R for c in range(len(r["cut_points"]))]
    MC = {int(k): v for k, v in json.load(open(f"{OUT()}/method_cut.json")).items() if v.get("fb")}
    if job == "e6":
        its = []
        for rid, m in sorted(MC.items()):
            base = {"rid": rid, "ntok": m["keep_tok"], "n": CFG["e6_samples"]}
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
                    base = {"rid": rid, "ntok": m["keep_tok"], "rw": j, "nrw": nrw, "q": q, "n": CFG["e7_samples"]}
                    its += [{**base, "cond": "rw_fb", "fb": m["fb"], "actor": "teacher"},
                            {**base, "cond": "rw_fb", "fb": m["fb"], "actor": "student"},
                            {**base, "cond": "rw_nofb", "fb": None, "actor": "teacher"},
                            {**base, "cond": "rw_nofb", "fb": None, "actor": "student"}]
            n0, L = m["keep_tok"], len(Rr[rid]["resp_ids"])
            if L - n0 >= 16:                                     # control: the original tail discarded after the cut
                for q in QS:
                    base = {"rid": rid, "ntok": n0 + int((L - n0) * q), "q": q, "fb": None, "n": CFG["e7_samples"], "cond": "tail"}
                    its += [{**base, "actor": "teacher"}, {**base, "actor": "student"}]
        return its
    raise ValueError(job)


def state_ids(ts, r, it, RW):
    """State = the first ntok tokens of the student's rollout [+ feedback template] [+ the first nrw rewrite tokens];
    the feedback is spliced exactly as in training (M.FEEDBACK_TMPL)."""
    ids = list(r["resp_ids"][:it["ntok"]])
    if it["fb"]:
        ids = ts.encode(ts.decode(ids, skip_special_tokens=True) + M.FEEDBACK_TMPL.format(fb=it["fb"]), add_special_tokens=False)
    if it.get("rw") is not None:
        ids = ids + list(RW[it["rid"]][it["rw"]][:it["nrw"]])
    return ids


def stage_cont(actor, job, shard, nsh):
    from vllm import SamplingParams
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    ts, tt = toks(); stop = stop_ids(actor)
    RW = rewrites() if job == "e7" else {}
    its = [it for it in items_for(job) if it.get("actor", actor) == actor][shard::nsh]
    reqs, sps, keep = [], [], []
    for i, it in enumerate(its):
        r = R[it["rid"]]
        p = r["s_prompt_ids"] if actor == "student" else r["t_prompt_ids"]
        ids = state_ids(ts, r, it, RW)
        budget = max(256, CFG["response_budget"] - it["ntok"] - (it.get("nrw") or 0))   # as in training: budget minus the kept prefix
        budget = min(budget, CFG["max_model_len"] - len(p) - len(ids) - 8)
        if budget < 64: continue
        reqs.append({"prompt_token_ids": p + ids}); keep.append(it)
        sps.append(SamplingParams(n=it["n"], temperature=1.0, top_p=1.0, max_tokens=budget, stop_token_ids=stop,
                                  seed=zlib.crc32(f"{job}|{actor}|{shard}|{i}".encode()) % (2 ** 31)))
    outs = llm(CFG["student"] if actor == "student" else CFG["teacher"]).generate(reqs, sps)
    res = []
    for it, o in zip(keep, outs):
        ss = [{"text": c.text, "truncated": c.finish_reason == "length"} for c in o.outputs]
        if job == "e6" and actor == "student" and it["cond"] == "critique":      # keep token ids of the rewrites used by e7
            for s, c in zip(ss[:2], o.outputs[:2]): s["ids"] = [t for t in c.token_ids if t not in stop]
        res.append({**it, "actor": actor, "samples": ss})
    jw(f"{OUT()}/cont_{actor}_{job}_{shard}.jsonl", res)
    say(f"[cont {actor} {job}] shard {shard}: {len(res)} states")


def stage_grade(job):
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    ts, _ = toks()
    RW = rewrites() if job == "e7" else {}
    rows = jlg(f"{OUT()}/cont_*_{job}_[0-9]*.jsonl")
    pairs = []
    for x in rows:
        r = R[x["rid"]]
        pre = ts.decode(state_ids(ts, r, {**x, "fb": None}, RW), skip_special_tokens=True)   # graded text excludes the feedback
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
    jw(f"{OUT()}/graded_{job}.jsonl", rows)
    say(f"[grade {job}] {len(rows)} states, {sum(len(x['samples']) for x in rows)} continuations")


def stage_closing():
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    have = {(x["rid"], x["actor"], x["cut"]) for x in jl(f"{OUT()}/graded_e1.jsonl")}
    miss = [(rid, a, c) for rid, r in R.items() for a in ("teacher", "student") for c in range(len(r["cut_points"])) if (rid, a, c) not in have]
    if miss: raise SystemExit(f"e1 is missing {len(miss)} states, e.g. {miss[:3]} (run every cont shard, then grade e1)")
    V = collections.defaultdict(dict)
    for x in jl(f"{OUT()}/graded_e1.jsonl"): V[x["rid"]][(x["actor"], x["cut"])] = [s["correct"] for s in x["samples"]]
    A, B, Cc = {}, [], []
    for rid, r in R.items():
        if r["correct"]: continue
        n = len(r["cut_points"])
        vt8 = [statistics.mean(V[rid][("teacher", c)][:8]) for c in range(n)]
        close = next((c for c in range(n) if vt8[c] < 0.5), None)
        if close is None: B.append(rid)
        elif close == 0: Cc.append(rid)
        else:
            A[rid] = {"close": close, "close_tok": r["cut_tokens"][close], "close_seg": r["cut_points"][close],
                      "prev_seg": r["cut_points"][close - 1], "vt8": vt8}
    json.dump({"A": A, "B": B, "C": Cc}, open(f"{OUT()}/closing.json", "w"))
    for b in ("main", "ext"):
        inb = lambda rid: R[rid].get("batch", "main") == b
        say(f"[closing] {b} batch: {sum(not r['correct'] and r.get('batch', 'main') == b for r in R.values())} incorrect rollouts: "
            f"A {sum(map(inb, A))}, B {sum(map(inb, B))}, C {sum(map(inb, Cc))}")


# ------------------------------------------------------------------ critic
def numbered(segs): return "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))


def critic_user(problem, segs, gt=None):
    """The user message of training; without the reference answer only that line is removed."""
    return (f"Problem:\n{problem}\n\n" + (f"Correct answer (reference only -- never reveal it): {gt}\n\n" if gt else "")
            + f"Student's solution, split into numbered segments:\n\n{numbered(segs)}")


def process_critic(text, ntok, segs, gt, stu):
    """Post-processing of training: drop the unfinished last block if truncated; parse all errors; flag leaks."""
    trunc = ntok >= CFG["critic_tokens"]
    if trunc: text = M.drop_incomplete_tail(text)
    raw = M.parse_critic_multi(text, segs)
    errs = M.drop_leaky(raw, gt, stu)
    none = not raw and bool(M._RE_NONE.search(text))
    return {"truncated": trunc, "none": none, "valid": none or bool(raw),
            "errs_raw": [list(e) for e in raw], "errs": [list(e) for e in errs]}


def stage_critic(shard, nsh):
    from vllm import SamplingParams
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    A = {int(k): v for k, v in json.load(open(f"{OUT()}/closing.json"))["A"].items()}
    _, tt = toks()
    jobs = []
    for rid, a in sorted(A.items()):
        r = R[rid]; segs = r["segments"]
        if r.get("batch", "main") == "ext":            # extension batch: only the critique that sets the method cut
            jobs.append(("full_gt", rid, None, segs, CRITIC_FULL_GT, r["reference_answer"])); continue
        for rel in (-2, -1, 0, 1, 2, 3, 4):
            c = a["close"] + rel
            if c < 1 or c >= len(r["cut_points"]): continue
            jobs.append(("partial", rid, rel, segs[:r["cut_points"][c]], CRITIC_PARTIAL_NOGT, None))
        jobs.append(("full_nogt", rid, None, segs, CRITIC_FULL_NOGT, None))
        jobs.append(("full_gt", rid, None, segs, CRITIC_FULL_GT, r["reference_answer"]))
    jobs = jobs[shard::nsh]
    reqs, keep, skipped = [], [], 0
    for job in jobs:
        kind, rid, rel, ss, sysp, gt = job
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": critic_user(R[rid]["problem"], ss, gt)}]
        ids = tt.encode(tt.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **CFG["teacher_chat_kwargs"]),
                        add_special_tokens=False)
        if len(ids) + CFG["critic_tokens"] > CFG["critic_max_model_len"]:   # does not fit the teacher's context
            skipped += 1; continue
        reqs.append({"prompt_token_ids": ids}); keep.append(job)
    jobs = keep
    if skipped: say(f"[critic] shard {shard}: skipped {skipped} critiques whose prompt does not fit critic_max_model_len")
    outs = llm(CFG["teacher"], maxlen=CFG["critic_max_model_len"]).generate(
        reqs, SamplingParams(temperature=0.0, max_tokens=CFG["critic_tokens"]))
    res = collections.defaultdict(list)
    for (kind, rid, rel, ss, _, _), o in zip(jobs, outs):
        r = R[rid]; c = o.outputs[0]
        res[kind].append({"rid": rid, "rel": rel, "n_seg_shown": len(ss), "text": c.text, "ntok": len(c.token_ids),
                          **process_critic(c.text, len(c.token_ids), ss, r["reference_answer"], "\n\n".join(ss))})
    for k in ("partial", "full_nogt", "full_gt"): jw(f"{OUT()}/critic_{k}_{shard}.jsonl", res.get(k, []))
    say(f"[critic] shard {shard}: " + ", ".join(f"{k} {len(v)}" for k, v in res.items()))


def stage_merge_critic():
    for k in ("partial", "full_nogt", "full_gt"):
        jw(f"{OUT()}/critic_{k}.jsonl", jlg(f"{OUT()}/critic_{k}_[0-9]*.jsonl"))
    stage_method_cut()


def stage_method_cut():
    """Method cut for Figure 5(a): all listed errors of the full-solution critique, cut after the paragraph of the
    last error, feedback numbered by kerr_feedback; with append_answer the feedback ends with the reference answer
    and no leak filtering (the final method of the paper)."""
    mc = CFG["method_cut"]
    ts, _ = toks()
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    MC = {}
    for x in jl(f"{OUT()}/critic_{mc['critique']}.jsonl"):
        r = R[x["rid"]]
        errs = x["errs"] if mc["filter_leaks"] else x["errs_raw"]
        if not errs:
            MC[x["rid"]] = {"fb": None, "reason": "none" if x["none"] else ("leak" if x["errs_raw"] else "invalid")}
            continue
        sel = M.kerr_select([tuple(e) for e in errs], "last")
        last = sel[-1][0]
        fb = M.kerr_feedback(sel)
        if mc["append_answer"]:
            fb = f"{fb}\nThe correct final answer is {r['reference_answer']}."
        MC[x["rid"]] = {"fb": fb, "first": sel[0][0], "first_raw": x["errs_raw"][0][0], "last": last,
                        "n_err": len(sel), "keep_seg": last + 1,
                        "keep_tok": M._tok_prefix_for_text(ts, r["resp_ids"], "\n\n".join(r["segments"][:last + 1]))}
    json.dump(MC, open(f"{OUT()}/method_cut.json", "w"))
    say(f"[method_cut] {sum(1 for v in MC.values() if v['fb'])} cuts; without usable errors: "
          f"{collections.Counter(v['reason'] for v in MC.values() if not v['fb'])}")


# ------------------------------------------------------------------ judge
def judge_items():
    R = {r["rollout_id"]: r for r in jl(f"{OUT()}/rollouts.jsonl")}
    its = []
    for kind in CFG["judge_kinds"]:
        for x in jl(f"{OUT()}/critic_{kind}.jsonl"):
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


def stage_judge(shard, nsh):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tj = AutoTokenizer.from_pretrained(CFG["judge"])
    ML = CFG["judge_max_model_len"]
    its = judge_items()[shard::nsh]
    reqs, keep = [], []
    for it in its:
        msgs = [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": it["user"]}]
        ids = tj.encode(tj.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **CFG["judge_chat_kwargs"]),
                        add_special_tokens=False)
        if len(ids) > ML - 4096: continue
        reqs.append({"prompt_token_ids": ids}); keep.append((it, min(CFG["judge_max_tokens"], ML - len(ids) - 16)))
    sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mt, seed=7) for _, mt in keep]   # Qwen3 thinking-mode settings
    outs = llm(CFG["judge"], 128, ML).generate(reqs, sps)
    res = []
    for (it, _), o in zip(keep, outs):
        t = o.outputs[0].text; ans = t.split("</think>")[-1]
        lab = {int(i): l for i, l in re.findall(r"ERROR\s*(\d+)\s*:\s*(" + "|".join(JLABELS) + r")", ans)}
        res.append({"kind": it["kind"], "rid": it["rid"], "rel": it.get("rel"), "n": it["n"],
                    "labels": [lab.get(i + 1) for i in range(it["n"])],
                    "truncated": o.outputs[0].finish_reason == "length", "answer": ans[-2000:]})
    for kind in CFG["judge_kinds"]:
        jw(f"{OUT()}/judge_{kind}_{shard}.jsonl", [x for x in res if x["kind"] == kind])
    say(f"[judge] shard {shard}: {len(res)} critiques judged")


# ------------------------------------------------------------------ inspect / pack
def stage_inspect():
    """Print what each model sees, to check a new model family before any GPU job (needs only the tokenizers)."""
    from transformers import AutoConfig
    ts, tt = toks()
    p = problems()[0]
    for who, tok in (("student", ts), ("teacher", tt)):
        st = stop_ids(who)
        print(f"{who} stop ids {st} = {[tok.decode([i]) for i in st]}")
    for who, path, need in (("student", CFG["student"], CFG["max_model_len"]), ("teacher", CFG["teacher"], CFG["critic_max_model_len"]),
                            ("judge", CFG["judge"], CFG["judge_max_model_len"])):
        n = getattr(AutoConfig.from_pretrained(path), "max_position_embeddings", None)
        print(f"{who} max_position_embeddings {n}; this config needs {need}" + ("  <-- TOO LONG" if n and need > n else ""))
    print("\n=== student prompt (rollouts, student continuations)\n" + ts.decode(prompt_ids(ts, p["messages"], True)))
    print("=== teacher prompt (teacher continuations)\n" + tt.decode(prompt_ids(tt, p["messages"], False)))
    segs = ["First step of a solution.", "Second step, which contains the error.", "Answer: 42"]
    msgs = [{"role": "system", "content": CRITIC_FULL_GT}, {"role": "user", "content": critic_user(p["messages"][-1]["content"], segs, p["ground_truth"])}]
    txt = tt.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **CFG["teacher_chat_kwargs"])
    print("=== critic prompt (full solution, with the reference answer)\n" + txt)
    if CRITIC_FULL_GT[:60] not in txt: print("WARNING: the critic system prompt does not appear in the rendered template")
    ids = ts.encode("\n\n".join(segs), add_special_tokens=False)
    r = {"resp_ids": ids}
    st = state_ids(ts, r, {"ntok": len(ts.encode("\n\n".join(segs[:2]), add_special_tokens=False)), "fb": "Paragraph 1: example critique.", "rid": 0}, {})
    print("=== state after the feedback (what the student and teacher continue at the method cut)\n"
          + ts.decode(prompt_ids(ts, p["messages"], True) + st))


def stage_pack():
    """Copy of out_dir without continuation texts and token ids (everything the figures need), as out_dir/return.tar.gz."""
    src = OUT(); dst = f"{src}/return"
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    drop = ("resp_ids", "s_prompt_ids", "t_prompt_ids")
    jw(f"{dst}/rollouts.jsonl", [{**{k: v for k, v in r.items() if k not in drop}, "resp_len": len(r["resp_ids"])}
                                 for r in jl(f"{src}/rollouts.jsonl")])
    for job in ("e1", "e6", "e7"):
        rows = jl(f"{src}/graded_{job}.jsonl")
        for x in rows:
            x["samples"] = [{k: s[k] for k in ("correct", "truncated", "matches_original_answer") if k in s} for s in x["samples"]]
        if rows: jw(f"{dst}/graded_{job}.jsonl", rows)
    keep = ["closing.json", "method_cut.json", "raw_correct.json", "raw_capped.json", "summary.log"]
    keep += [f"critic_{k}.jsonl" for k in ("partial", "full_nogt", "full_gt")]
    keep += [os.path.basename(f) for f in glob.glob(f"{src}/judge_*_[0-9]*.jsonl")]
    for f in keep:
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", dst)
    json.dump({k: v for k, v in CFG.items()}, open(f"{dst}/config_used.json", "w"), indent=1)
    from criticopd_diag import plots
    plots.run(dst)
    with tarfile.open(f"{src}/return.tar.gz", "w:gz") as t: t.add(dst, arcname="return")
    print(f"wrote {src}/return.tar.gz ({os.path.getsize(f'{src}/return.tar.gz') / 1e6:.1f} MB)")


# ------------------------------------------------------------------ main
def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("stage"); ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    CFG = load_config(a.config)
    os.makedirs(OUT(), exist_ok=True)
    s, x = a.stage, a.args
    if s == "rollouts": stage_rollouts(int(x[0]), int(x[1]))
    elif s == "select": stage_select()
    elif s == "cont": stage_cont(x[0], x[1], int(x[2]), int(x[3]))
    elif s == "grade": stage_grade(x[0])
    elif s == "closing": stage_closing()
    elif s == "critic": stage_critic(int(x[0]), int(x[1]))
    elif s == "merge_critic": stage_merge_critic()
    elif s == "method_cut": stage_method_cut()
    elif s == "judge": stage_judge(int(x[0]), int(x[1]))
    elif s == "inspect": stage_inspect()
    elif s == "pack": stage_pack()
    elif s in ("plots", "check"):
        from criticopd_diag import plots
        plots.run(OUT(), check_only=(s == "check"))
    else:
        raise SystemExit(f"unknown stage {s}")


if __name__ == "__main__":
    main()
