#!/usr/bin/env python3
"""离线筛选 k 消融:批改老师按顺序列出所有错误,在第 k 个错误处断开(保留出错段)。

数据复用 out_ref_sh(R4GT@20 学生在 500 道题上的解答)中「写完但答错」的解答。
提示词、解析、选第 k 个、批语拼接、按 token 断开,全部直接调用训练代码
(critic_opd_agent_loop),保证离线与训练一致。
  critic  老师用多错误提示词批改(贪心,上限 1024)
  repair  每个 k(1/2/3/last)各重写 2 次,预算 = 16384 - 保留长度(与训练一致),按分片并行
  report  错误数分布、各 k 的修复成功率/写不完比例/重写长度/泄露/实际用到第 k 个错误的比例;
          与旧提示词 A(R4GT 的批改)对照;写出 decision.json
用法: python3 kerr_probe.py {critic|repair <shard> <nshards>|report}
"""
import collections, glob, json, os, random, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402

SRC = "/home/kzhao2/Relay-OPD/criticopd/out_ref_sh"
OUT = os.environ.get("KERR_OUT", "/home/kzhao2/Relay-OPD/criticopd/out_kerr")
STUDENT = "/home/kzhao2/Relay-OPD/outputs/checkpoints/criticopd_R4GT_pt17b/global_step_20/actor/huggingface"
TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
TOKENIZER = "/home/kzhao2/OPD/model/Qwen3-1.7B"
STOP_IDS = [151643, 151645]
RLEN, SEED, K_REPAIR = 16384, 2468, 2
KS = ["1", "2", "3", "last"]


def jl(path): return [json.loads(l) for l in open(path)]
def jw(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def src(name): return [x for f in sorted(glob.glob(f"{SRC}/shard_*/{name}")) for x in jl(f)]
def wrong_rollouts(): return [r for r in src("rollouts.jsonl") if not r["correct"] and not r["capped"]]


def grader():
    from opd.reward.math_reward import compute_score
    def g(text, gt):
        try:
            v = compute_score(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False
    return g


def strict_leak(fb, gt, stu): return M.leaks_answer(fb, gt, stu)


def filtered(c, rolls):
    """与训练一致:去掉反馈泄露答案的错误,再选第 k 个。"""
    r = rolls[(c["prob"], c["rep"])]
    return M.drop_leaky([tuple(e) for e in c["errs"]], r["gt"], r["text"])


def stage_critic():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    rolls = wrong_rollouts()
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    reqs, meta = [], []
    for r in rolls:
        segs = M.segment(r["text"])
        body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
        msg = [{"role": "system", "content": M.CRITIC_SYS_MULTI},          # 与训练里 kerr 分支逐字一致
               {"role": "user", "content": f"Problem:\n{r['question']}\n\n"
                                           f"Correct answer (reference only -- never reveal it): {r['gt']}\n\n"
                                           f"Student's solution, split into numbered segments:\n\n{body}"}]
        ids = tok.encode(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
        if len(ids) + 1024 > 40960: continue
        reqs.append({"prompt_token_ids": ids}); meta.append((r, segs))
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=40960, enable_prefix_caching=True)
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=1024))
    res = []
    for (r, segs), o in zip(meta, outs):
        t = o.outputs[0].text
        errs = M.parse_critic_multi(t, segs)
        res.append({"prob": r["prob"], "rep": r["rep"], "n_seg": len(segs), "errs": errs, "critic_text": t})
    jw(f"{OUT}/critic.jsonl", res)
    print(f"[critic] {len(res)} 条;至少找到 1 个错误 {sum(bool(x['errs']) for x in res)}")


def stage_repair(shard, nsh):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    g = grader()
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    crit = [c for c in jl(f"{OUT}/critic.jsonl") if filtered(c, rolls)]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    MAXLEN = RLEN + 4096
    reqs, sps, meta = [], [], []
    for ci, c in enumerate(crit):
        if ci % nsh != shard: continue
        r = rolls[(c["prob"], c["rep"])]
        segs = M.segment(r["text"])
        errs = filtered(c, rolls)
        for k in KS:
            sel = M.kerr_select(errs, k)
            idx, fb = sel[-1][0], M.kerr_feedback(sel)
            kept_txt = "\n\n".join(segs[:idx + 1])                       # 保留出错段,与训练一致
            kk = M._tok_prefix_for_text(tok, r["resp_ids"], kept_txt)
            kept_ids = r["resp_ids"][:kk]; kept = tok.decode(kept_ids, skip_special_tokens=True)
            ids = r["prompt_ids"] + tok.encode(kept + M.FEEDBACK_TMPL.format(fb=fb), add_special_tokens=False)
            budget = min(max(256, RLEN - len(kept_ids)), MAXLEN - len(ids) - 1)
            if budget < 256: continue
            for j in range(K_REPAIR):
                reqs.append({"prompt_token_ids": ids})
                sps.append(SamplingParams(seed=SEED + shard * 10_000_000 + len(sps), temperature=1.0, top_p=1.0,
                                          max_tokens=budget, stop_token_ids=STOP_IDS))
                meta.append({"prob": c["prob"], "rep": c["rep"], "k": k, "j": j, "k_eff": len(sel), "n_err": len(errs), "n_err_raw": len(c["errs"]),
                             "n_kept": len(kept_ids), "n_resp": len(r["resp_ids"]), "budget": budget,
                             "leak": strict_leak(fb, r["gt"], r["text"]), "kept": kept, "gt": r["gt"]})
    llm = LLM(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=MAXLEN, enable_prefix_caching=True)
    outs = llm.generate(reqs, sps)
    res = []
    for m, o in zip(meta, outs):
        n = len(o.outputs[0].token_ids); capped = n >= m["budget"]
        res.append({**{k: v for k, v in m.items() if k not in ("kept", "gt")}, "new_tokens": n, "capped": capped,
                    "correct": bool((not capped) and g(m["kept"] + o.outputs[0].text, m["gt"]))})
    jw(f"{OUT}/repair_{shard}.jsonl", res)
    print(f"[repair] shard {shard}: {len(res)} 次重写")


def boot(d, n=10000, seed=0):
    rng = random.Random(seed); m = len(d)
    bs = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, bs[int(.025 * n)], bs[int(.975 * n)]


def stage_report():
    crit = jl(f"{OUT}/critic.jsonl")
    rep = [x for f in sorted(glob.glob(f"{OUT}/repair_*.jsonl")) for x in jl(f)]
    rolls = {(r["prob"], r["rep"]): r for r in src("rollouts.jsonl")}
    n_raw = sum(len(c["errs"]) for c in crit); raw_parsed = sum(bool(c["errs"]) for c in crit)
    for c in crit: c["errs"] = filtered(c, rolls)          # 以下全部按过滤后的错误统计(与训练一致)
    n_drop = n_raw - sum(len(c["errs"]) for c in crit)
    W = len(crit); parsed = [c for c in crit if c["errs"]]
    print(f"〇、泄露过滤:{n_raw} 个错误中 {n_drop} 个的反馈写出了标准答案,已去掉;"
          f"至少剩 1 个错误的解答 {len(parsed)}/{W}(过滤前 {raw_parsed})")
    ne = collections.Counter(min(len(c["errs"]), 5) for c in crit)
    print(f"一、批改老师找到的错误数({W} 份写完答错的解答)")
    for k in range(6): print(f"  {k} 个{'(含 5 个以上)' if k == 5 else ''}: {ne[k]:>4}  {100*ne[k]/W:5.1f}%")
    ge2 = sum(len(c["errs"]) >= 2 for c in crit) / W; ge3 = sum(len(c["errs"]) >= 3 for c in crit) / W
    gaps = [(c["errs"][1][0] - c["errs"][0][0]) / max(c["n_seg"], 1) for c in parsed if len(c["errs"]) >= 2]
    print(f"  至少 2 个: {100*ge2:.1f}%,至少 3 个: {100*ge3:.1f}%;第 1、2 个错误相隔(占解答长度)中位 {statistics.median(gaps) if gaps else float('nan'):.2f}")

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in rep: by[(x["prob"], x["rep"])][x["k"]].append(x)
    keys = [k for k in by if all(len(by[k][q]) == K_REPAIR for q in KS)]
    succ = lambda key, q: statistics.mean(x["correct"] for x in by[key][q])
    print(f"\n二、各 k 的表现(同一批 {len(keys)} 份解答,每个 k 重写 {K_REPAIR} 次)")
    print("  | k | 修复成功率 | 对比 k=1 | 写不完 | 重写长度中位 | 保留占原解答 | 严格泄露 | 实际用到第 k 个错误 |")
    S = {}
    for q in KS:
        xs = [x for key in keys for x in by[key][q]]
        S[q] = statistics.mean(succ(key, q) for key in keys)
        d = boot([succ(key, q) - succ(key, "1") for key in keys]) if q != "1" else None
        # 「实际用到」:数字 k 看是否找到了至少 k 个错误;last 看是否至少 2 个(否则与 k=1 完全相同)
        kk = 2 if q == "last" else int(q)
        use = statistics.mean(by[key][q][0]["n_err"] >= kk for key in keys)
        print(f"  | {q} | {100*S[q]:.1f}% | " + (f"{100*d[0]:+.1f} [{100*d[1]:+.1f}, {100*d[2]:+.1f}]" if d else "—")
              + f" | {100*statistics.mean(x['capped'] for x in xs):.1f}% | {statistics.median(x['new_tokens'] for x in xs):.0f}"
              f" | {100*statistics.mean(x['n_kept']/x['n_resp'] for x in xs):.0f}% | {100*statistics.mean(x['leak'] for x in xs):.1f}%"
              f" | {100*use:.0f}% |")
    # 旧提示词 A(R4GT 的批改)在同一批解答上的修复成功率,用来检查「换提示词」本身的影响
    old = collections.defaultdict(list)
    for x in src("repair.jsonl"):
        if x["variant"] == "A": old[(x["prob"], x["rep"])].append(x["correct"])
    ka = [k for k in keys if len(old.get(k, [])) == 2]
    A = statistics.mean(statistics.mean(old[k]) for k in ka) if ka else float("nan")
    K1 = statistics.mean(succ(k, "1") for k in ka) if ka else float("nan")
    print(f"\n三、换提示词的影响:同一批 {len(ka)} 份解答上,旧提示词 A(R4GT){100*A:.1f}%,新提示词 k=1 {100*K1:.1f}%")

    # ---- 自动决定(事先定好的规则) ----
    parse_rate = raw_parsed / W
    leak = {q: statistics.mean(x["leak"] for key in keys for x in by[key][q]) for q in KS}
    # 只拦截「提示词不可用」:解析率太低、泄露、或 k=1 比旧提示词低 10 个点以上。
    # k 的比较在同一个新提示词内部进行(新旧 k=1 相差 ≥3 个点时同时训练新提示词 k=1 作对照),
    # 所以新提示词略弱(部分数据上低 5.6 个点)不影响消融的公平性。
    gate = parse_rate >= 0.85 and (K1 >= A - 0.10) and max(leak.values()) <= 0.02
    cands = ["2", "last"] + (["3"] if ge3 >= 0.30 else [])
    best = max(cands, key=lambda q: S[q])
    dec = {"parse_rate": round(parse_rate, 3), "succ": {q: round(S[q], 4) for q in KS}, "old_A": round(A, 4), "new_k1": round(K1, 4),
           "leak": {q: round(leak[q], 4) for q in KS}, "n_err_dropped_leak": n_drop, "share_ge2": round(ge2, 3), "share_ge3": round(ge3, 3), "candidates": cands, "gate_ok": gate,
           "train": [], "reason": ""}
    if not gate:
        dec["reason"] = (f"未通过合理性检查:解析率 {parse_rate:.2f}(要求 ≥0.85),新提示词 k=1 {K1:.3f} 对旧提示词 {A:.3f}"
                         f"(要求不低于 10 个百分点以上),最大泄露率 {max(leak.values()):.3f}(要求 ≤0.02)")
    else:
        dec["train"].append({"2": "R4GTK2", "3": "R4GTK3", "last": "R4GTKL"}[best])
        if abs(K1 - A) >= 0.03:
            dec["train"].append("R4GTK1")
            dec["reason"] = f"训练 k={best}(候选中修复成功率最高);新旧提示词 k=1 相差 {100*(K1-A):+.1f} 个百分点 ≥3,同时训练新提示词 k=1 作对照"
        else:
            dec["reason"] = f"训练 k={best}(候选中修复成功率最高);新旧提示词 k=1 相差 {100*(K1-A):+.1f} 个百分点 <3,直接与 R4GT 比较"
    json.dump(dec, open(f"{OUT}/decision.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n四、决定:{dec['reason']}\n  要训练: {dec['train']}")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "repair": stage_repair(int(sys.argv[2]), int(sys.argv[3]))
    else: {"critic": stage_critic, "report": stage_report}[st]()
    if st in ("critic", "repair"):
        # 结果已经写完:直接退出,避免 vLLM 收尾时卡住(错题集实验的分片 5 就卡过 1.5 小时)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
