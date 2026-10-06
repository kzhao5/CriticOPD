#!/usr/bin/env python3
"""文档阶段 2A:当前多错误版本的转折 token。
在 Section 2 诊断的带反馈重写上(方法断点 + 全部反馈,每条轨迹前 2 次重写),逐 token 计算
  Delta_t = A^fb_t - A^clean_t = log pi_T(z_t | x, y_<=, f, z_<t) - log pi_T(z_t | x, y_<=, z_<t)
(学生项相同,相减抵消)。teacher 用学生的对话模板,与训练时打分完全相同。
输出 turn_tokens.json / turn_tokens.md:Delta 沿重写位置的分布、超过阈值的比例、大 Delta 在后段的位置(是否多峰)。"""
import collections, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/kzhao2/Relay-OPD/diag2")
import diag as D   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

OUTD = "/home/kzhao2/Relay-OPD/results_final/analysis"
SRC = ["/home/kzhao2/Relay-OPD/diag2/out", "/home/kzhao2/Relay-OPD/diag2/out_b"]
BUCKETS = [(0, 8), (8, 32), (32, 128), (128, 256), (256, 1024), (1024, 4096), (4096, 10 ** 9)]
TAUS = (0.1, 0.3, 0.5)


def main():
    ts = AutoTokenizer.from_pretrained(D.STUDENT)
    model = AutoModelForCausalLM.from_pretrained(D.TEACHER, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    def lp(seq, n):                                   # 序列末尾 n 个 token 的 teacher logprob
        ids = torch.tensor([seq], device="cuda")
        with torch.no_grad():
            h = model.model(input_ids=ids).last_hidden_state[0, len(seq) - n - 1:len(seq) - 1]
            tgt = ids[0, len(seq) - n:]
            out = []
            for i in range(0, n, 2048):
                lg = model.lm_head(h[i:i + 2048]).float()
                out.append(torch.log_softmax(lg, -1).gather(1, tgt[i:i + 2048, None])[:, 0])
        return torch.cat(out).cpu().numpy()

    deltas = []
    for src in SRC:
        R = {r["rollout_id"]: r for r in D.jl(f"{src}/rollouts.jsonl")}
        MC = {int(k): v for k, v in json.load(open(f"{src}/method_cut.json")).items() if v.get("fb")}
        for x in D.jlg(f"{src}/cont_student_e6_[0-9]*.jsonl"):
            if x["cond"] != "critique" or x["rid"] not in MC: continue
            r, m = R[x["rid"]], MC[x["rid"]]
            keep = r["resp_ids"][:m["keep_tok"]]
            ctx = ts.encode(ts.decode(keep, skip_special_tokens=True) + D.M.FEEDBACK_TMPL.format(fb=m["fb"]), add_special_tokens=False)
            for s in x["samples"][:2]:
                z = s.get("ids")
                if not z: continue
                z = z[:8192]
                a = lp(r["s_prompt_ids"] + ctx + z, len(z)); b = lp(r["s_prompt_ids"] + keep + z, len(z))
                deltas.append((a - b).astype(np.float32))
    print("rewrites:", len(deltas), flush=True)
    allpos = np.concatenate([np.arange(len(d)) for d in deltas]); alld = np.concatenate(deltas)
    res = {"n_rewrites": len(deltas), "n_tokens": int(len(alld)), "by_bucket": {}, "turning_fraction": {}, "late_peaks": {}}
    for lo, hi in BUCKETS:
        sel = (allpos >= lo) & (allpos < hi)
        if sel.sum() == 0: continue
        res["by_bucket"][f"{lo}-{hi}"] = {"n": int(sel.sum()), "mean": float(alld[sel].mean()), "p90": float(np.percentile(alld[sel], 90)),
                                          **{f"frac>{t}": float((alld[sel] > t).mean()) for t in TAUS}}
    for t in TAUS:
        res["turning_fraction"][str(t)] = float((alld > t).mean())
        # 每条重写里,超过阈值的 token 有多少落在前 64 个之后(后段有无第二个峰)
        late = [int(((d > t) & (np.arange(len(d)) >= 64)).sum()) for d in deltas]
        res["late_peaks"][str(t)] = {"rewrites_with_late_tokens": float(np.mean([v > 0 for v in late])), "late_tokens_per_rewrite": float(np.mean(late))}
    # 绝对贡献:Delta 的总和里,前 32 个 token 占多少
    tot = sum(float(d.sum()) for d in deltas); head = sum(float(d[:32].sum()) for d in deltas)
    res["share_of_total_delta_in_first_32"] = head / tot if tot else None
    json.dump(res, open(f"{OUTD}/turn_tokens.json", "w"), indent=1)
    L = ["# 转折 token(当前多错误版本,Section 2 的带反馈重写)\n", f"重写 {res['n_rewrites']} 条,共 {res['n_tokens']} 个 token。Delta_t = 带反馈与不带反馈的 teacher logprob 之差。\n",
         "| 位置 | token 数 | Delta 均值 | p90 | >0.1 | >0.3 | >0.5 |", "|---" * 7 + "|"]
    for k, d in res["by_bucket"].items():
        L.append(f"| {k} | {d['n']} | {d['mean']:.3f} | {d['p90']:.3f} | {d['frac>0.1']:.1%} | {d['frac>0.3']:.1%} | {d['frac>0.5']:.1%} |")
    L.append("\n| 阈值 | 转折 token 占重写 token | 有后段(>=64)大 Delta 的重写比例 | 每条重写后段大 Delta 个数 |\n|---|---|---|---|")
    for t in TAUS:
        lp_ = res["late_peaks"][str(t)]
        L.append(f"| {t} | {res['turning_fraction'][str(t)]:.1%} | {lp_['rewrites_with_late_tokens']:.1%} | {lp_['late_tokens_per_rewrite']:.1f} |")
    L.append(f"\nDelta 总和中前 32 个 token 的占比:{res['share_of_total_delta_in_first_32']:.1%}")
    open(f"{OUTD}/turn_tokens.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
