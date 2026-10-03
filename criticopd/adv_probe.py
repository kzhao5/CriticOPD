#!/usr/bin/env python3
"""离线检验:L_fb 的 advantage 在「修复成功」和「修复失败」的重写上有没有差别。

数据复用 out_ref_sh:R4GT@20 学生在 500 道 DAPO 题上的解答;写完答错的解答用 R4GT 的批改提示词
(变体 A)批改,每份重写 2 次,带对错标签。按训练的公式重算每个修复 token 的 advantage
(k1 估计器 + 策略梯度,loss_agg_mode=token-mean,无截断):
    A_t = log pi_T(y_t | 题目, 保留前缀, 反馈, y_<t) - log pi_S(y_t | 题目, 保留前缀, y_<t)
另算两组对照:
    A_plain:teacher 不看反馈时的 advantage;A - A_plain 就是反馈对 teacher 打分的影响
    on-policy:学生原始解答上每个 token 的 advantage(普通 OPD 信号)
打分方式与训练一致:vLLM,max_tokens=1,prompt_logprobs=0(取实际 token 的 logprob)。

用法: python3 adv_probe.py {teacher|student} <shard> <nshards>      python3 adv_probe.py report
"""
import collections, glob, json, os, random, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
import verl.experimental.agent_loop.critic_opd_agent_loop as M   # noqa: E402

SRC = "/home/kzhao2/Relay-OPD/criticopd/out_ref_sh"
OUT = os.environ.get("ADV_OUT", "/home/kzhao2/Relay-OPD/criticopd/out_adv")
STUDENT = "/home/kzhao2/Relay-OPD/outputs/checkpoints/criticopd_R4GT_pt17b/global_step_20/actor/huggingface"
TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
TOKENIZER = "/home/kzhao2/OPD/model/Qwen3-1.7B"
MAXLEN = 16384 + 4096 + 64


def jl(p): return [json.loads(l) for l in open(p)]
def jw(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def src(name): return [x for f in sorted(glob.glob(f"{SRC}/shard_*/{name}")) for x in jl(f)]
def key(x): return (str(x["prob"]), str(x["rep"]))


def build_items():
    """按固定顺序构造所有要打分的序列。与 ref_probe 的重写阶段逐行一致(SPLICE=after)。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    rolls = {key(r): r for r in src("rollouts.jsonl")}
    crit = {key(c): c for c in src("critic.jsonl") if c["variant"] == "A"}
    items, bad = [], collections.Counter()
    for x in src("repair.jsonl"):
        if x["variant"] != "A":
            continue
        k = key(x); c, r = crit[k], rolls[k]
        segs = M.segment(r["text"])
        kept_txt = "\n\n".join(segs[:c["seg"] + 1])
        kk = M._tok_prefix_for_text(tok, r["resp_ids"], kept_txt)
        kept_ids = r["resp_ids"][:kk]
        if len(kept_ids) != x["n_kept"]:
            bad["n_kept 对不上"] += 1; continue
        kept_txt = tok.decode(kept_ids, skip_special_tokens=True)
        ctx_ids = tok.encode(kept_txt + M.FEEDBACK_TMPL.format(fb=c["feedback"]), add_special_tokens=False)
        rep_ids = tok.encode(x["text"], add_special_tokens=False)
        if not rep_ids:
            bad["空重写"] += 1; continue
        if abs(len(rep_ids) - x["new_tokens"]) > 1:
            bad["重写重新分词后长度差 >1(仍保留)"] += 1
        fb_seq = r["prompt_ids"] + ctx_ids + rep_ids
        pl_seq = r["prompt_ids"] + kept_ids + rep_ids
        if len(fb_seq) >= MAXLEN:
            bad["超长"] += 1; continue
        items.append({"key": ["rep", k[0], k[1], x["j"]], "fb": fb_seq, "plain": pl_seq, "n": len(rep_ids)})
    for k in sorted(rolls):
        r = rolls[k]
        items.append({"key": ["roll", k[0], k[1]], "plain": r["prompt_ids"] + r["resp_ids"], "n": len(r["resp_ids"])})
    return items, bad


def score(llm, seqs, ns):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp)
    res = []
    for s, n, o in zip(seqs, ns, outs):
        pl = o.prompt_logprobs
        res.append([round(d[t].logprob, 4) for t, d in zip(s[-n:], pl[-n:])])
    return res


def stage_score(who, shard, nsh):
    from vllm import LLM
    os.makedirs(OUT, exist_ok=True)
    items, bad = build_items()
    if shard == 0:
        print(f"[items] 重写 {sum(i['key'][0] == 'rep' for i in items)},原解答 {sum(i['key'][0] == 'roll' for i in items)};{dict(bad)}")
    mine = items[shard::nsh]
    llm = LLM(model=TEACHER if who == "teacher" else STUDENT, dtype="bfloat16", gpu_memory_utilization=0.70,
              max_model_len=MAXLEN, enable_prefix_caching=False, max_num_batched_tokens=8192)
    if who == "teacher":           # 带反馈(训练里真正用的)和不带反馈两份
        rep = [i for i in mine if i["key"][0] == "rep"]
        lf = score(llm, [i["fb"] for i in rep], [i["n"] for i in rep])
        lp = score(llm, [i["plain"] for i in mine], [i["n"] for i in mine])
        fbm = {json.dumps(i["key"]): v for i, v in zip(rep, lf)}
        rows = [{"key": i["key"], "t_plain": v, **({"t_fb": fbm[json.dumps(i["key"])]} if i["key"][0] == "rep" else {})}
                for i, v in zip(mine, lp)]
    else:
        ls = score(llm, [i["plain"] for i in mine], [i["n"] for i in mine])
        rows = [{"key": i["key"], "s": v} for i, v in zip(mine, ls)]
    jw(f"{OUT}/{who}_{shard}.jsonl", rows)
    print(f"[{who}] shard {shard}: {len(rows)} 条")


# ---------------------------------------------------------------- report
BUCKETS = ((0, 32), (32, 256), (256, 1024), (1024, 10**9))


def summ(a, ap=None):
    """把一条序列的逐 token advantage 压成汇总量;之后的统计和 bootstrap 只用汇总量(几百万 token 逐个算太慢)。"""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    d = {"n": len(a), "sum": float(a.sum()), "pos": int((a > 0).sum()), "mean": float(a.mean()) if len(a) else 0.0,
         "b": [(float(a[lo:hi].sum()), len(a[lo:hi])) for lo, hi in BUCKETS]}
    if ap is not None:
        ap = np.asarray(ap, dtype=np.float64)
        d["sum_p"] = float(ap.sum()); d["bp"] = [float(ap[lo:hi].sum()) for lo, hi in BUCKETS]
    return d


def boot(f, groups, n=4000, seed=0):
    """按解答(rollout)整簇重抽样,返回 f 的 95% 区间。"""
    rng = random.Random(seed); ks = list(groups)
    vals = sorted(f([g for _ in ks for g in groups[ks[rng.randrange(len(ks))]]]) for _ in range(n))
    return vals[int(.025 * n)], vals[int(.975 * n)]


def auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        for t in range(i, j): rank[t] = (i + j + 1) / 2
        i = j
    rs = sum(rank[t] for t, (v, l) in enumerate(allv) if l == 1)
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def stage_report():
    rolls = {key(r): r for r in src("rollouts.jsonl")}
    crit = {key(c): c for c in src("critic.jsonl") if c["variant"] == "A"}
    reps = {json.dumps(["rep", *key(x), x["j"]]): x for x in src("repair.jsonl") if x["variant"] == "A"}
    cut = {}
    for x in reps.values(): cut[key(x)] = x["n_kept"]
    S = {}
    for f in sorted(glob.glob(f"{OUT}/student_*.jsonl")):
        for l in open(f):
            x = json.loads(l); S[json.dumps(x["key"])] = x["s"]
    R, O = [], []
    for f in sorted(glob.glob(f"{OUT}/teacher_*.jsonl")):
        for l in open(f):
            t = json.loads(l); k = json.dumps(t["key"]); s = S.get(k)
            if s is None: continue
            if t["key"][0] == "rep":
                x = reps[k]; r0 = rolls[key(x)]
                a = [u - v for u, v in zip(t["t_fb"], s)]; ap = [u - v for u, v in zip(t["t_plain"], s)]
                grp = "写不完" if x["capped"] else ("修复成功" if x["correct"] else "修复失败")
                R.append({"grp": grp, "roll": key(x), **summ(a, ap),
                          "leak": M.leaks_answer(crit[key(x)]["feedback"], r0["gt"], r0["text"])})
            else:
                kk = (t["key"][1], t["key"][2]); r = rolls[kk]
                a = [u - v for u, v in zip(t["t_plain"], s)]
                if r["correct"]:
                    O.append({"grp": "原解答·答对", **summ(a)})
                elif r["capped"]:
                    O.append({"grp": "原解答·写不完", **summ(a)})
                else:
                    O.append({"grp": "原解答·答错(全部)", **summ(a)})
                    if kk in cut:
                        O.append({"grp": "  其中被保留的前缀", **summ(a[:cut[kk]])})
                        O.append({"grp": "  其中被丢弃的尾部", **summ(a[cut[kk]:])})

    tm = lambda rows: sum(r["sum"] for r in rows) / max(sum(r["n"] for r in rows), 1)
    tmp = lambda rows: sum(r["sum_p"] for r in rows) / max(sum(r["n"] for r in rows), 1)
    sm = lambda rows: statistics.mean(r["mean"] for r in rows if r["n"])
    pos = lambda rows: sum(r["pos"] for r in rows) / max(sum(r["n"] for r in rows), 1)
    print("一、各组 token 的 advantage(修复组:teacher 看反馈,即训练里的 L_fb;原解答组:普通 OPD)")
    print("  | 组 | 条数 | 长度中位 | token 加权均值 | 逐条均值 | A>0 的 token 占比 | teacher 不看反馈时 | 反馈带来的提升 |")
    for g in ["修复成功", "修复失败", "写不完"]:
        rows = [r for r in R if r["grp"] == g]
        if rows:
            print(f"  | {g} | {len(rows)} | {statistics.median(r['n'] for r in rows):.0f} | {tm(rows):+.3f} | {sm(rows):+.3f} | "
                  f"{100*pos(rows):.1f}% | {tmp(rows):+.3f} | {tm(rows)-tmp(rows):+.3f} |")
    for g in ["原解答·答对", "原解答·答错(全部)", "  其中被保留的前缀", "  其中被丢弃的尾部", "原解答·写不完"]:
        rows = [r for r in O if r["grp"] == g and r["n"]]
        if rows:
            print(f"  | {g} | {len(rows)} | {statistics.median(r['n'] for r in rows):.0f} | {tm(rows):+.3f} | {sm(rows):+.3f} | "
                  f"{100*pos(rows):.1f}% | — | — |")

    ok = [r for r in R if r["grp"] == "修复成功"]; bad = [r for r in R if r["grp"] == "修复失败"]
    groups = collections.defaultdict(list)
    for r in ok + bad: groups[r["roll"]].append(r)
    def f_tok(rows):
        o = [r for r in rows if r["grp"] == "修复成功"]; b = [r for r in rows if r["grp"] == "修复失败"]
        return tm(o) - tm(b) if o and b else 0.0
    d_tok = tm(ok) - tm(bad); ci = boot(f_tok, groups)
    print("\n二、修复成功 − 修复失败")
    print(f"  token 加权均值之差 {d_tok:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}](按解答整簇重抽样)")
    pairs = [(tm([r for r in g if r["grp"] == "修复成功"]), tm([r for r in g if r["grp"] == "修复失败"]))
             for g in groups.values() if {r["grp"] for r in g} == {"修复成功", "修复失败"}]
    if pairs:
        dd = [u - v for u, v in pairs]
        print(f"  同一份解答的两次重写一成一败({len(pairs)} 对):成功减失败的均值 {statistics.mean(dd):+.4f},"
              f"成功那次更高的比例 {100*statistics.mean(d > 0 for d in dd):.0f}%")
    so = [r["mean"] for r in ok]; sb = [r["mean"] for r in bad]
    sd = statistics.pstdev(so + sb)
    print(f"  逐条均值:成功 {statistics.mean(so):+.4f},失败 {statistics.mean(sb):+.4f},差 / 总标准差 = "
          f"{(statistics.mean(so)-statistics.mean(sb))/sd:+.2f}(Cohen d);用逐条均值预测修复成功的 AUC = {auc(so, sb):.3f}")
    print(f"  每条重写的 advantage 总和(梯度上的分量):成功 {statistics.mean(r['sum'] for r in ok):+.1f},"
          f"失败 {statistics.mean(r['sum'] for r in bad):+.1f};长度中位 成功 {statistics.median(r['n'] for r in ok):.0f},"
          f"失败 {statistics.median(r['n'] for r in bad):.0f}")
    print("\n三、修复 token 按位置分段的 token 加权均值(成功 / 失败)")
    for bi, (lo, hi) in enumerate(BUCKETS):
        sl = lambda rows: sum(r["b"][bi][0] for r in rows) / max(sum(r["b"][bi][1] for r in rows), 1)
        fe = lambda rows: (sum(r["b"][bi][0] - r["bp"][bi] for r in rows)) / max(sum(r["b"][bi][1] for r in rows), 1)
        nt = sum(r["b"][bi][1] for r in ok + bad)
        print(f"  第 {lo}-{hi if hi < 10**9 else '末尾'} 个 token(占修复 token {100*nt/sum(r['n'] for r in ok + bad):.1f}%):"
              f" advantage {sl(ok):+.3f} / {sl(bad):+.3f};反馈带来的提升 {fe(ok):+.3f} / {fe(bad):+.3f}")
    n_rep = sum(r["n"] for r in R); n_on = sum(r["n"] for r in O if r["grp"] in ("原解答·答对", "原解答·写不完"))
    print(f"\n四、本批数据里修复 token 占训练 token 的比例 ≈ {100*n_rep/max(n_rep+n_on,1):.1f}%"
          f"(修复 {n_rep},on-policy {n_on};每份答错的解答这里重写了 2 次,训练里是 1 次)")
    print("\n五、反馈泄露答案的重写(批改把标准答案写进了反馈)")
    for lk in (True, False):
        for g in ("修复成功", "修复失败"):
            rows = [r for r in R if r["grp"] == g and r["leak"] == lk]
            if rows:
                print(f"  {'泄露' if lk else '不泄露'}·{g}: {len(rows)} 条,token 加权均值 {tm(rows):+.3f},"
                      f"反馈带来的提升 {tm(rows)-tmp(rows):+.3f},长度中位 {statistics.median(r['n'] for r in rows):.0f}")
    json.dump({"d_tok": d_tok, "ci": ci, "auc": auc(so, sb)}, open(f"{OUT}/summary.json", "w"))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "report":
        stage_report()
    else:
        stage_score(st, int(sys.argv[2]), int(sys.argv[3]))
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
