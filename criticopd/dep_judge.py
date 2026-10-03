#!/usr/bin/env python3
"""离线分析:多错误批改找到的第 2、3、... 个错误,和前面的错误是什么关系。

数据:out_kerr/critic.jsonl(多错误提示词对 540 份写完答错的解答的批改,未做泄露过滤)。
对至少有 2 个错误的解答,让裁判模型判断第 i(≥2)个错误属于:
  PROPAGATED   推理本身没问题,只是沿用了前面某个错误得出的值或结论(前面改对,这一步照做也就对了)
  RESTATEMENT  只是复述/汇报前面已经错了的结果(比如最后一行答案),没有新的推理
  INDEPENDENT  自己有新的错误(公式、逻辑、计算、看错条件),前面全对它也是错的
  NOT_AN_ERROR 这一步其实是对的
两个裁判(Qwen3-4B-Instruct-2507 = 批改老师本身;Qwen3-8B 思考模式)各判一遍,报告一致性。
再把标签和 out_kerr 的重写结果对上:k=1 / k=2 的修复成功率随第 2 个错误的类型怎么变。

用法: python3 dep_judge.py judge {4b|8b}      python3 dep_judge.py report
"""
import collections, glob, json, os, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402

SRC = "/home/kzhao2/Relay-OPD/criticopd/out_ref_sh"
KERR = "/home/kzhao2/Relay-OPD/criticopd/out_kerr"
OUT = os.environ.get("DEP_OUT", "/home/kzhao2/Relay-OPD/criticopd/out_dep")
JUDGES = {"4b": "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507", "8b": "/home/kzhao2/OPD/model/Qwen3-8B"}
LABELS = ["PROPAGATED", "RESTATEMENT", "INDEPENDENT", "NOT_AN_ERROR"]

SYS = (
    "You are an expert mathematics grader. You are given a math problem, its correct answer, a student's "
    "solution split into numbered segments, and a list of errors that a teacher found in the solution, in "
    "order of appearance. Your job is to decide, for every error after the first, how it relates to the "
    "errors listed before it.\n\n"
    "Labels:\n"
    "PROPAGATED - the reasoning in this step is valid in itself; it is wrong only because it uses a value, "
    "expression or conclusion produced by an earlier listed error. Had the earlier errors been fixed, this "
    "step done the same way would be correct.\n"
    "RESTATEMENT - the step only restates or reports a wrong result that was already reached earlier (for "
    "example the final answer line or a summary), with no new reasoning.\n"
    "INDEPENDENT - the step contains a new mistake of its own (a wrong formula, wrong logic, an arithmetic "
    "slip, a misread condition) that would still be wrong even if every earlier step were correct.\n"
    "NOT_AN_ERROR - the step is actually correct.\n\n"
    "For PROPAGATED and RESTATEMENT, also name the earlier error it depends on.\n"
    "Reply with exactly one line per error from ERROR 2 onward and nothing else, in this format:\n"
    "ERROR 2: <LABEL> [on ERROR <j>]\n"
    "ERROR 3: <LABEL> [on ERROR <j>]")


def jl(p): return [json.loads(l) for l in open(p)]
def src(name): return [x for f in sorted(glob.glob(f"{SRC}/shard_*/{name}")) for x in jl(f)]


def items():
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    out = []
    for c in jl(f"{KERR}/critic.jsonl"):
        if len(c["errs"]) < 2: continue
        r = rolls[(c["prob"], c["rep"])]
        segs = M.segment(r["text"])
        body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
        errs = "\n\n".join(f"ERROR {i + 1}\nSEGMENT: {e[0]}\nQUOTE: {e[1]}\nTEACHER'S NOTE: {e[2]}"
                           for i, e in enumerate(c["errs"]))
        user = (f"Problem:\n{r['question']}\n\nCorrect answer: {r['gt']}\n\n"
                f"Student's solution, split into numbered segments:\n\n{body}\n\n"
                f"Errors found by the teacher, in order:\n\n{errs}")
        out.append({"prob": c["prob"], "rep": c["rep"], "n_err": len(c["errs"]), "n_seg": len(segs),
                    "segs_idx": [e[0] for e in c["errs"]], "user": user})
    return out


def parse(text, n_err):
    text = text.split("</think>")[-1]
    lab = {}
    for m in re.finditer(r"ERROR\s*(\d+)\s*:\s*\**\s*(PROPAGATED|RESTATEMENT|INDEPENDENT|NOT_AN_ERROR)\**"
                         r"(?:[^\n]*?ERROR\s*(\d+))?", text, re.I):
        i = int(m.group(1))
        if 2 <= i <= n_err and i not in lab:
            lab[i] = (m.group(2).upper(), int(m.group(3)) if m.group(3) else None)
    return lab


def stage_judge(which):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    its = items()
    tok = AutoTokenizer.from_pretrained(JUDGES[which], trust_remote_code=True)
    think = which == "8b"
    reqs, keep = [], []
    for it in its:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": it["user"]}]
        kw = {"enable_thinking": True} if think else {}
        ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw),
                         add_special_tokens=False)
        if len(ids) + 8192 > 40960: continue
        reqs.append({"prompt_token_ids": ids}); keep.append(it)
    llm = LLM(model=JUDGES[which], dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=40960,
              enable_prefix_caching=True)
    sp = (SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192, seed=7) if think   # Qwen3 思考模式推荐参数
          else SamplingParams(temperature=0.0, max_tokens=512))
    outs = llm.generate(reqs, sp)
    with open(f"{OUT}/judge_{which}.jsonl", "w") as f:
        for it, o in zip(keep, outs):
            t = o.outputs[0].text
            lab = parse(t, it["n_err"])
            f.write(json.dumps({k: it[k] for k in ("prob", "rep", "n_err", "n_seg", "segs_idx")}
                               | {"labels": {str(i): v for i, v in lab.items()}, "complete": len(lab) == it["n_err"] - 1,
                                  "finish": o.outputs[0].finish_reason, "text": t[-1500:]}) + "\n")
    print(f"[judge {which}] {len(keep)} 份")


def stage_report():
    J = {w: {(x["prob"], x["rep"]): x for x in jl(f"{OUT}/judge_{w}.jsonl")} for w in ("4b", "8b")
         if os.path.exists(f"{OUT}/judge_{w}.jsonl")}
    for w, d in J.items():
        n = len(d); comp = sum(x["complete"] for x in d.values())
        print(f"\n===== 裁判 {w}:{n} 份,标签完整 {comp}({100*comp/n:.0f}%) =====")
        c2 = collections.Counter(x["labels"]["2"][0] for x in d.values() if "2" in x["labels"])
        call = collections.Counter(v[0] for x in d.values() for v in x["labels"].values())
        last = collections.Counter(x["labels"][str(x["n_err"])][0] for x in d.values() if str(x["n_err"]) in x["labels"])
        for name, c in (("第 2 个错误", c2), ("第 2 个及以后的全部错误", call), ("最后一个错误", last)):
            t = sum(c.values())
            print(f"  {name}({t}): " + ",".join(f"{l} {100*c[l]/t:.1f}%" for l in LABELS))
        roots = [1 + sum(v[0] == "INDEPENDENT" for v in x["labels"].values()) for x in d.values() if x["complete"]]
        rc = collections.Counter(min(r, 4) for r in roots)
        print(f"  每份解答的独立错误(根因)个数:" + ",".join(f"{k}{'+' if k == 4 else ''} 个 {100*rc[k]/len(roots):.1f}%" for k in sorted(rc))
              + f";只有 1 个根因(后面全是传导/复述/误报)的占 {100*rc[1]/len(roots):.1f}%")
        on1 = [v[1] for x in d.values() for v in x["labels"].values() if v[0] in ("PROPAGATED", "RESTATEMENT") and v[1]]
        if on1:
            print(f"  传导/复述指向第 1 个错误的比例 {100*statistics.mean(j == 1 for j in on1):.0f}%")
    if len(J) == 2:
        both = [k for k in J["4b"] if k in J["8b"] and "2" in J["4b"][k]["labels"] and "2" in J["8b"][k]["labels"]]
        a = [J["4b"][k]["labels"]["2"][0] for k in both]; b = [J["8b"][k]["labels"]["2"][0] for k in both]
        po = statistics.mean(x == y for x, y in zip(a, b))
        ca, cb = collections.Counter(a), collections.Counter(b)
        pe = sum(ca[l] * cb[l] for l in LABELS) / len(a) ** 2
        coarse = lambda l: "INDEP" if l == "INDEPENDENT" else ("DEP" if l in ("PROPAGATED", "RESTATEMENT") else "NA")
        po2 = statistics.mean(coarse(x) == coarse(y) for x, y in zip(a, b))
        print(f"\n两个裁判在第 2 个错误上的一致率 {100*po:.1f}%(Cohen κ = {(po-pe)/(1-pe):.2f});"
              f"只分「独立 / 依赖前面 / 不是错误」三类时一致率 {100*po2:.1f}%")

    # ---- 与 k 消融的重写结果对照(只用两个裁判一致的;且泄露过滤没有动过前两个错误的解答) ----
    rep = [x for f in sorted(glob.glob(f"{KERR}/repair_*.jsonl")) for x in jl(f)]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in rep: by[(x["prob"], x["rep"])][x["k"]].append(x["correct"])
    crit = {(c["prob"], c["rep"]): c for c in jl(f"{KERR}/critic.jsonl")}
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    lab = {}
    for k in (J.get("4b") or {}):
        ls = [J[w][k]["labels"].get("2", (None,))[0] for w in J if k in J[w]]
        if len(ls) == len(J) and len(set(ls)) == 1 and ls[0]:
            raw = [tuple(e) for e in crit[k]["errs"]]; r = rolls[k]
            if M.drop_leaky(raw, r["gt"], r["text"])[:2] == raw[:2]:
                lab[k] = ls[0]
    print(f"\n与重写结果对照(裁判一致且前两个错误未被泄露过滤改动的 {len(lab)} 份):")
    print("  | 第 2 个错误的类型 | 份数 | k=1 成功率 | k=2 成功率 | k=2 − k=1 | 最后一个 − k=1 |")
    for l in LABELS:
        ks = [k for k, v in lab.items() if v == l and all(q in by[k] for q in ("1", "2", "last"))]
        if len(ks) < 5: continue
        s = {q: statistics.mean(statistics.mean(by[k][q]) for k in ks) for q in ("1", "2", "last")}
        print(f"  | {l} | {len(ks)} | {100*s['1']:.1f}% | {100*s['2']:.1f}% | {100*(s['2']-s['1']):+.1f} | {100*(s['last']-s['1']):+.1f} |")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        stage_judge(sys.argv[2]); sys.stdout.flush(); os._exit(0)
    stage_report()
