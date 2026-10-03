#!/usr/bin/env python3
"""离线检验「错题集」:跨题目复用批改得来的教训,能不能帮学生少犯错。

复用 ref_probe 的数据(out_ref_sh:R4GT@20 学生在 500 道题上的解答 + 批语)。
  label   teacher 给每条批语标:错误类别(10 选 1)、一句可推广的教训、是否会在别的题上重复
  build   按题目哈希一分为二:「以前的题」建错题集,「新题」做检验;为每道新题检索 3 条教训
  gen     学生生成(按任务分片):
            检验一 新题从头做,三种条件各 4 次:不给教训 / 检索到的 3 条 / 随机 3 条(对照)
            检验二 新题上的错误解答按批语重写,两种条件各 2 次:只有批语 / 批语 + 3 条教训
  report  类别分布 + 两个检验的配对比较
每个采样请求都有独立的种子;学生 prompt 用 enable_thinking=False,与训练一致。
用法: python3 notebook_probe.py {label|build|gen|report} [shard nshards]
"""
import collections, glob, hashlib, json, math, os, random, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402

SRC = "/home/kzhao2/Relay-OPD/criticopd/out_ref_sh"
OUT = os.environ.get("NB_OUT", "/home/kzhao2/Relay-OPD/criticopd/out_notebook")
STUDENT = "/home/kzhao2/Relay-OPD/outputs/checkpoints/criticopd_R4GT_pt17b/global_step_20/actor/huggingface"
TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
TOKENIZER = "/home/kzhao2/OPD/model/Qwen3-1.7B"
STOP_IDS = [151643, 151645]
RLEN, SEED = 16384, 777
K_FRESH, K_REPAIR, N_LESSON = 4, 2, 3

CATS = ["arithmetic slip", "algebra or sign error", "misread or ignored a condition", "missed a case",
        "overcounting or undercounting", "wrong formula or theorem", "unjustified assumption",
        "domain or boundary error", "flawed overall approach", "other"]
LABEL_SYS = (
    "You analyze a student's mistake in a math solution. You are given the problem, the sentence where "
    "the student first went wrong, and a teacher's feedback on that sentence.\n\n"
    "1. CATEGORY: choose exactly one of: " + "; ".join(CATS) + ".\n"
    "2. LESSON: one sentence that would help a student avoid this kind of mistake on a DIFFERENT problem. "
    "Do not mention any number, variable, or object that is specific to this problem.\n"
    "3. REUSABLE: YES if a mistake of this kind could plausibly recur on other problems, NO if it is a "
    "one-off slip specific to this problem.\n\n"
    "Reply in exactly this format:\nCATEGORY: <category>\nLESSON: <one sentence>\nREUSABLE: <YES or NO>")
LESSON_BLOCK = "Before solving, keep in mind these common mistakes:\n{items}\n\n"
FB_WITH_LESSONS = ("\n\n[Teacher feedback] {fb}\n\nCommon mistakes to avoid:\n{items}\n\n"
                   "Using this feedback, continue the solution from here. Put your final answer in \\boxed{{}}.\n\n")


def jl(path): return [json.loads(l) for l in open(path)]
def jw(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def src(name): return [x for f in sorted(glob.glob(f"{SRC}/shard_*/{name}")) for x in jl(f)]
def is_new(prob): return int(hashlib.sha1(str(prob).encode()).hexdigest(), 16) % 2 == 1


def grader():
    from opd.reward.math_reward import compute_score
    def g(text, gt):
        try:
            v = compute_score(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False
    return g


def best_critiques():
    """每条写完答错的解答取一条可用批语:有参考解答版(B)就用 B,否则用 A。"""
    best = {}
    for c in src("critic.jsonl"):
        if c["kind"] != "wrong" or not c["parsed"]: continue
        k = (c["prob"], c["rep"])
        if c["variant"] == "B" or (c["variant"] == "A" and k not in best): best[k] = c
    return best


# ---------------------------------------------------------------- label
def stage_label():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    best = best_critiques()
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    reqs, meta = [], []
    for k, c in best.items():
        mq = M._RE_Q.search(c["critic_text"]); quote = mq.group(1).strip() if mq else ""
        msg = [{"role": "system", "content": LABEL_SYS},
               {"role": "user", "content": f"Problem:\n{rolls[k]['question']}\n\nStudent's erroneous sentence:\n"
                                           f"{quote}\n\nTeacher feedback:\n{c['feedback']}"}]
        reqs.append({"prompt_token_ids": tok.encode(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False),
                                                    add_special_tokens=False)}); meta.append((k, c))
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=16384)
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=200))
    rows = []
    for (k, c), o in zip(meta, outs):
        t = o.outputs[0].text
        cat = (re.search(r"CATEGORY:\s*(.+)", t) or [None, ""])[1].strip().lower().rstrip(".")
        cat = next((x for x in CATS if x in cat), "other")
        lesson = (re.search(r"LESSON:\s*(.+)", t) or [None, ""])[1].strip()
        reuse = bool(re.search(r"REUSABLE:\s*YES", t, re.I))
        rows.append({"prob": k[0], "rep": k[1], "variant": c["variant"], "category": cat, "lesson": lesson,
                     "reusable": reuse and bool(lesson), "raw": t})
    jw(f"{OUT}/labels.jsonl", rows)
    print(f"[label] {len(rows)} 条批语已标注")


# ---------------------------------------------------------------- build
_W = re.compile(r"[a-z]{3,}")
def bow(text): return collections.Counter(_W.findall(text.lower()))


def stage_build():
    labels = jl(f"{OUT}/labels.jsonl")
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    qs = {r["prob"]: r for r in rolls.values()}
    bank = [l for l in labels if not is_new(l["prob"]) and l["reusable"]]
    new_probs = sorted(p for p in qs if is_new(p))
    # 题目文本的 TF-IDF 相似度:教训来自与新题相近的旧题
    docs = {p: bow(qs[p]["question"]) for p in qs}
    df = collections.Counter(w for d in docs.values() for w in d)
    N = len(docs)
    def vec(p): return {w: c * math.log(N / df[w]) for w, c in docs[p].items()}
    def cos(a, b):
        s = sum(a[w] * b.get(w, 0.0) for w in a); na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
        return s / (na * nb) if na and nb else 0.0
    V = {p: vec(p) for p in qs}
    freq = collections.Counter(l["category"] for l in bank)
    tasks = []
    for p in new_probs:
        cand = sorted(bank, key=lambda l: (-cos(V[p], V[l["prob"]]), -freq[l["category"]]))
        pick, used = [], set()
        for l in cand:                                   # 每个类别最多一条,同题的教训不取(bank 里本就没有新题)
            if l["category"] in used: continue
            pick.append(l["lesson"]); used.add(l["category"])
            if len(pick) == N_LESSON: break
        rng = random.Random(SEED + p)
        rand = [l["lesson"] for l in rng.sample(bank, N_LESSON)]
        tasks.append({"type": "fresh", "prob": p, "retrieved": pick, "random": rand})
    best = best_critiques(); lab = {(l["prob"], l["rep"]): l for l in labels}
    for k, c in best.items():
        if not is_new(k[0]): continue
        p = k[0]
        t = next(t for t in tasks if t["prob"] == p)
        tasks.append({"type": "repair", "prob": p, "rep": k[1], "seg": c["seg"], "feedback": c["feedback"],
                      "lessons": t["retrieved"]})
    jw(f"{OUT}/tasks.jsonl", tasks)
    print(f"[build] 错题集 {len(bank)} 条可复用教训(来自 {len({l['prob'] for l in bank})} 道旧题);"
          f"新题 {len(new_probs)} 道;重写任务 {sum(t['type']=='repair' for t in tasks)} 条")


# ---------------------------------------------------------------- gen
def stage_gen(shard, nsh):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    g = grader()
    tasks = jl(f"{OUT}/tasks.jsonl")
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    qs = {r["prob"]: r for r in rolls.values()}
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    items = lambda ls: "\n".join(f"{i+1}. {x}" for i, x in enumerate(ls))
    reqs, meta = [], []
    for ti, t in enumerate(tasks):
        if ti % nsh != shard: continue
        if t["type"] == "fresh":
            base = qs[t["prob"]]["messages"]
            for cond, ls in (("none", None), ("retrieved", t["retrieved"]), ("random", t["random"])):
                msgs = [dict(m) for m in base]
                if ls: msgs[-1]["content"] = LESSON_BLOCK.format(items=items(ls)) + msgs[-1]["content"]
                ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                         enable_thinking=False), add_special_tokens=False)
                for j in range(K_FRESH):
                    reqs.append((ids, RLEN)); meta.append({**t, "cond": cond, "j": j, "kept": ""})
        else:
            r = rolls[(t["prob"], t["rep"])]
            segs = M.segment(r["text"])
            k = M._tok_prefix_for_text(tok, r["resp_ids"], "\n\n".join(segs[:t["seg"] + 1]))
            kept_ids = r["resp_ids"][:k]; kept = tok.decode(kept_ids, skip_special_tokens=True)
            budget = max(256, RLEN - len(kept_ids))
            for cond in ("feedback", "feedback+lessons"):
                ctx = kept + (M.FEEDBACK_TMPL.format(fb=t["feedback"]) if cond == "feedback"
                              else FB_WITH_LESSONS.format(fb=t["feedback"], items=items(t["lessons"])))
                ids = r["prompt_ids"] + tok.encode(ctx, add_special_tokens=False)
                for j in range(K_REPAIR):
                    reqs.append((ids, budget)); meta.append({**t, "cond": cond, "j": j, "kept": kept})
    sps = [SamplingParams(seed=SEED + 10_000_000 * shard + i, temperature=1.0, top_p=1.0, max_tokens=mt,
                          stop_token_ids=STOP_IDS) for i, (_, mt) in enumerate(reqs)]
    llm = LLM(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=RLEN + 4096,
              enable_prefix_caching=True)
    outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in reqs], sps)
    res = []
    for m, o in zip(meta, outs):
        n = len(o.outputs[0].token_ids); capped = o.outputs[0].finish_reason == "length"
        gt = qs[m["prob"]]["gt"]
        res.append({"type": m["type"], "prob": m["prob"], "rep": m.get("rep"), "cond": m["cond"], "j": m["j"],
                    "new_tokens": n, "capped": capped,
                    "correct": bool((not capped) and g(m["kept"] + o.outputs[0].text, gt))})
    jw(f"{OUT}/gen_{shard}.jsonl", res)
    print(f"[gen] shard {shard}: {len(res)} 次生成")


# ---------------------------------------------------------------- report
def boot(d, n=10000, seed=0):
    rng = random.Random(seed); m = len(d)
    bs = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, bs[int(.025 * n)], bs[int(.975 * n)]


def stage_report():
    labels = jl(f"{OUT}/labels.jsonl")
    res = [x for f in sorted(glob.glob(f"{OUT}/gen_*.jsonl")) for x in jl(f)]
    print(f"一、错误会不会重复({len(labels)} 条批语)")
    c = collections.Counter(l["category"] for l in labels); n = len(labels)
    for cat, k in c.most_common(): print(f"  {cat:<32} {k:>4}  {100*k/n:5.1f}%")
    top = [k for _, k in c.most_common()]
    print(f"  前 3 类覆盖 {100*sum(top[:3])/n:.1f}%,前 5 类覆盖 {100*sum(top[:5])/n:.1f}%")
    print(f"  判为「会在别的题上重复」: {100*sum(l['reusable'] for l in labels)/n:.1f}%")
    print("  教训示例:"); [print(f"    [{l['category']}] {l['lesson']}") for l in random.Random(1).sample([l for l in labels if l['reusable']], 8)]

    print("\n二、检验一:新题从头做(每题每种条件 4 次,按题配对)")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in res:
        if x["type"] == "fresh": acc[x["prob"]][x["cond"]].append(x["correct"])
    P = [p for p in acc if all(len(acc[p][c]) == K_FRESH for c in ("none", "retrieved", "random"))]
    mean = lambda p, c: sum(acc[p][c]) / K_FRESH
    for c in ("none", "retrieved", "random"):
        print(f"  {c:<10} 正确率 {100*statistics.mean(mean(p, c) for p in P):5.1f}%  (题数 {len(P)})")
    for a, b in (("retrieved", "none"), ("random", "none"), ("retrieved", "random")):
        m, lo, hi = boot([mean(p, a) - mean(p, b) for p in P])
        print(f"  {a} − {b}: {100*m:+.1f} 个百分点  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
    for c in ("none", "retrieved", "random"):
        L = [x["new_tokens"] for x in res if x["type"] == "fresh" and x["cond"] == c]
        print(f"  {c:<10} 生成长度中位 {statistics.median(L):.0f}")

    print("\n三、检验二:错误解答按批语重写(每条每种条件 2 次,按解答配对)")
    rp = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in res:
        if x["type"] == "repair": rp[(x["prob"], x["rep"])][x["cond"]].append(x["correct"])
    K = [k for k in rp if all(len(rp[k][c]) == K_REPAIR for c in ("feedback", "feedback+lessons"))]
    for c in ("feedback", "feedback+lessons"):
        print(f"  {c:<18} 修复成功率 {100*statistics.mean(sum(rp[k][c])/K_REPAIR for k in K):5.1f}%  (解答数 {len(K)})")
    m, lo, hi = boot([sum(rp[k]["feedback+lessons"])/K_REPAIR - sum(rp[k]["feedback"])/K_REPAIR for k in K])
    print(f"  加教训 − 只有批语: {100*m:+.1f} 个百分点  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "gen": stage_gen(int(sys.argv[2]), int(sys.argv[3]))
    else: {"label": stage_label, "build": stage_build, "report": stage_report}[st]()
