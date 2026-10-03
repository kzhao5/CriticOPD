#!/usr/bin/env python3
"""各方法训练算力核算(Qwen3-1.7B 学生 <- Qwen3-4B-Instruct-2507 教师,主表所选 checkpoint)。

一、实测:训练日志里每一步 timing_s/step 求和 x 分配的 GPU 数(不含启动、存 checkpoint、离线数据生成)。
    各方法跑在不同硬件上(A100 / H100 / B200),所以实测 GPU 小时只能在同一种硬件内比较。
二、估算 FLOPs(与硬件无关):每个 token 前向 2N、训练 6N(前向+反向),verl 每步另有一次
    旧策略 logprob 前向 2N;注意力项未计(平均上下文 2-4k 时约占 7%,各方法相近)。
    离线数据(SFT / SeqKD / TRD)按「训练到所选步数实际用到的样本数」计入生成成本。
    CriticOPD 的批改、重写、带反馈重打分没有逐步记录 token 数:用训练日志里每一步的
    「被修复样本占比」乘以离线数据(R4GT@20、同一个批改提示词)量到的每次修复的长度。
"""
import collections, glob, json, os, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")

R = "/home/kzhao2/Relay-OPD"
CK = f"{R}/outputs/checkpoints"
NS, NT = 1.72e9, 4.02e9            # Qwen3-1.7B / Qwen3-4B(含与词嵌入共享的输出层)
B = 128

RUNS = [  # 名称, 目录, 所选步数, GPU 数, 硬件
    ("SFT", "sft_pt17b", 40, 4, "H100"),
    ("SeqKD", "seqkd_pt17b", 139, 4, "H100"),
    ("TRD", "trd_pt17b", 40, 4, "H100"),
    ("GRPO", "grpo_pt17b", 120, 4, "A100"),
    ("SKD", "skd_pt17b", 120, 4, "A100"),
    ("OPD", "opd_1p7b", 80, 8, "A100"),
    ("FastOPD", "fastopd8192_1p7b", 60, 8, "A100"),
    ("RelayOPD", "relay_1p7b", 40, 8, "A100"),
    ("CriticOPD", "criticopd_R4GT_pt17b", 40, 4, "A100"),
]
# 离线数据每个样本的平均长度(outputs/offline_data/*_pt17b/*.parquet)
TEACHER_TRAJ = {"prompt": 157.0, "teacher_gen": 4219.1}
TRD_DATA = {"prompt": 157.0, "student_gen": 1415.0, "teacher_prompt": 1657.6, "teacher_gen": 3173.1}
SFT_ELAPSED_H, SFT_STEPS = 3.5, 139          # SFT 训练器不记录每步时间:用整次任务时长按步数摊


def parse(path):
    """解析 verl 日志里的 'step:N - k:v - k:v ...';同一步出现多次(续跑)时取最后一次。"""
    txt = open(path, errors="ignore").read()
    out = {}
    # "step:" 前面不能是字母/下划线/斜杠:perf/time_per_step:141 - ... 不是新的一步
    for m in re.finditer(r"(?<![\w/])step:(\d+) - (.*?)(?=\n|(?<![\w/])step:\d+ - |\Z)", txt):
        d = {}
        for kv in m.group(2).split(" - "):
            if ":" in kv:
                k, v = kv.rsplit(":", 1)
                try: d[k.strip()] = float(re.match(r"[-+0-9.eE]+", v.strip()).group(0))
                except Exception: pass
        if d: out[int(m.group(1))] = d
    return out


def critic_ratios():
    """离线数据(out_ref_sh,R4GT@20,批改提示词 A)上量到的每次批改/修复的长度。"""
    import verl.experimental.agent_loop.critic_opd_agent_loop as M
    from transformers import AutoTokenizer
    tok_t = AutoTokenizer.from_pretrained("/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507")
    tok_s = AutoTokenizer.from_pretrained("/home/kzhao2/OPD/model/Qwen3-1.7B")
    jl = lambda p: [json.loads(l) for l in open(p)]
    S = f"{R}/criticopd/out_ref_sh"
    rolls = {(r["prob"], r["rep"]): r for f in sorted(glob.glob(f"{S}/shard_*/rollouts.jsonl")) for r in jl(f)}
    crit = {}
    for f in sorted(glob.glob(f"{S}/shard_*/critic.jsonl")):
        for c in jl(f):
            if c["variant"] == "A": crit[(c["prob"], c["rep"])] = c
    reps = [x for f in sorted(glob.glob(f"{S}/shard_*/repair.jsonl")) for x in jl(f) if x["variant"] == "A"]
    cin, cout, lo_all = [], [], []
    for k, c in crit.items():
        r = rolls[k]; segs = M.segment(r["text"])
        body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
        sysp = M.CRITIC_SYS_GT_CAPPED if r["capped"] else M.CRITIC_SYS_GT
        msg = [{"role": "system", "content": sysp},
               {"role": "user", "content": f"Problem:\n{r['question']}\n\nCorrect answer (reference only -- never reveal it): "
                                           f"{r['gt']}\n\nStudent's solution, split into numbered segments:\n\n{body}"}]
        cin.append(len(tok_t.encode(tok_t.apply_chat_template(msg, add_generation_prompt=True, tokenize=False),
                                    add_special_tokens=False)))
        cout.append(int(c["critic_tokens"])); lo_all.append(len(r["resp_ids"]))
    kept, rep, fbx, lo_rep = [], [], [], []
    for x in reps:
        k = (x["prob"], x["rep"]); c = crit[k]; r = rolls[k]
        kept.append(x["n_kept"]); rep.append(x["new_tokens"]); lo_rep.append(len(r["resp_ids"]))
        fbx.append(len(tok_s.encode(M.FEEDBACK_TMPL.format(fb=c["feedback"]), add_special_tokens=False)))
    m = statistics.mean
    return {"n_attempt": len(cin), "C_in": m(cin), "C_out": m(cout), "L_o_attempt": m(lo_all),
            "K": m(kept), "Rr": m(rep), "F": m(fbx), "L_o_rep": m(lo_rep)}


def main():
    cr = critic_ratios()
    print("CriticOPD 每次批改/修复的长度(离线 R4GT@20,批改 %d 次):批改输入 %.0f,批改输出 %.0f;"
          "被修复解答原长 %.0f,保留前缀 %.0f,反馈文本 %.0f,重写 %.0f" %
          (cr["n_attempt"], cr["C_in"], cr["C_out"], cr["L_o_rep"], cr["K"], cr["F"], cr["Rr"]))
    tally_q = 2586 / 2994          # R4GT 训练结束时的计数:修复 / 送去批改(其余是 NONE 或解析失败)

    rows = []
    for name, d, S, G, hw in RUNS:
        L = parse(f"{CK}/{d}/training.log")
        steps = [s for s in range(1, S + 1) if s in L]
        miss = [s for s in range(1, S + 1) if s not in L]
        tail = [L[s] for s in steps[-10:]]
        get = lambda s, k, default=0.0: (L[s] if s in L else {}).get(k, statistics.mean(t.get(k, default) for t in tail))
        sec = flop_s = flop_t = flop_tr = 0.0
        tok = collections.Counter(); per = []
        for s in range(1, S + 1):
            P = get(s, "prompt_length/mean"); Rl = get(s, "response_length/mean")
            T = get(s, "perf/total_num_tokens") or get(s, "train/global_tokens")
            dsec = get(s, "timing_s/step") if name != "SFT" else SFT_ELAPSED_H * 3600 / SFT_STEPS
            sec += dsec
            sg = sp = tg = tp = 0.0               # 学生生成 / 学生预填 / 教师生成 / 教师读入
            tr = 8 * NS * T                       # 训练 6N + 旧策略 logprob 2N
            if name == "SFT":
                tr = 6 * NS * T
                tg = B * TEACHER_TRAJ["teacher_gen"]; tp = B * TEACHER_TRAJ["prompt"]
            elif name == "SeqKD":
                tg = B * TEACHER_TRAJ["teacher_gen"]; tp = B * TEACHER_TRAJ["prompt"] + T
            elif name == "TRD":
                sg = B * TRD_DATA["student_gen"]; sp = B * TRD_DATA["prompt"]
                tg = B * TRD_DATA["teacher_gen"]; tp = B * TRD_DATA["teacher_prompt"]       # 离线改写
                tp += B * (TRD_DATA["teacher_prompt"] + Rl)                                 # 在线打分(教师提示)
            elif name == "GRPO":
                sg = 8 * B * Rl; sp = 8 * B * P
            elif name == "SKD":
                sg = B * Rl; sp = B * P
                tp = B * Rl + T                    # 教师逐 token 验证学生草稿 + 训练用的打分
            elif name in ("OPD", "FastOPD"):
                sg = B * Rl; sp = B * P; tp = T
            elif name == "RelayOPD":
                rho = get(s, "actor/distillation/relay_teacher_mask_ratio")
                sg = B * Rl * (1 - rho); tg = B * Rl * rho; sp = B * P; tp = T
            elif name == "CriticOPD":
                # 离线长度是第 20 步测的;训练中回答长度从 1.3k 涨到 6.8k,按本步与第 20 步的长度比缩放
                sc = Rl / L[20]["response_length/mean"]
                K, Rr, Lo = cr["K"] * sc, cr["Rr"] * sc, cr["L_o_rep"] * sc
                Cin = (cr["C_in"] - cr["L_o_attempt"]) + cr["L_o_attempt"] * sc
                f = get(s, "critic_opd/repair_seq_frac"); nrep = f * B; natt = nrep / tally_q
                sg = B * Rl + nrep * (Lo - K)                          # 原始解答(含被丢弃的尾部)+ 重写
                sp = B * P + nrep * (P + K + cr["F"])                  # 重写时读入「题目+保留前缀+反馈」
                tp = natt * Cin + T + nrep * (P + K + cr["F"] + Rr)    # 批改 + 打分 + 带反馈重打分
                tg = natt * cr["C_out"]
                tok["批改次数"] += natt; tok["修复次数"] += nrep
            tok["学生生成"] += sg; tok["学生读入"] += sp; tok["教师生成"] += tg; tok["教师读入"] += tp; tok["训练"] += T
            flop_s += 2 * NS * (sg + sp); flop_t += 2 * NT * (tg + tp); flop_tr += tr
            per.append((dsec, 2 * NS * (sg + sp) + 2 * NT * (tg + tp) + tr))
        rows.append({"name": name, "step": S, "gpus": G, "hw": hw, "gpu_h": sec * G / 3600, "wall_h": sec / 3600,
                     "flops": flop_s + flop_t + flop_tr, "f_s": flop_s, "f_t": flop_t, "f_tr": flop_tr,
                     "tok": dict(tok), "miss": miss, "sec_step": sec / S,
                     "gpu_h40": sum(x[0] for x in per[:40]) * G / 3600, "flops40": sum(x[1] for x in per[:40])})

    opd = next(r for r in rows if r["name"] == "OPD")
    opd_step = opd["flops"] / opd["step"]
    print("\n一、到所选 checkpoint 的总成本")
    print("| 方法 | 步数 | 硬件 | 实测 GPU 小时 | 每步秒数 | 总 FLOPs (1e18) | 相对 OPD@80 | 每步 FLOPs 相对 OPD | 学生生成 / 教师读入 / 教师生成 / 训练 (百万 token) |")
    for r in rows:
        t = r["tok"]
        print(f"| {r['name']} | {r['step']} | {r['gpus']}x{r['hw']} | {r['gpu_h']:.1f} | {r['sec_step']:.0f} | "
              f"{r['flops']/1e18:.2f} | {r['flops']/opd['flops']:.2f}x | {r['flops']/r['step']/opd_step:.2f}x | "
              f"{t.get('学生生成',0)/1e6:.0f} / {t.get('教师读入',0)/1e6:.0f} / {t.get('教师生成',0)/1e6:.0f} / {t['训练']/1e6:.0f} |"
              + (f" 缺 {len(r['miss'])} 步按最后 10 步均值补" if r["miss"] else ""))
    print("\n一(b)、前 40 步(所有方法同样步数)")
    print("| 方法 | 硬件 | 实测 GPU 小时 | FLOPs (1e18) | 相对 OPD 前 40 步 |")
    for r in rows:
        print(f"| {r['name']} | {r['gpus']}x{r['hw']} | {r['gpu_h40']:.1f} | {r['flops40']/1e18:.2f} | {r['flops40']/opd['flops40']:.2f}x |")
    print("\n二、FLOPs 构成(占该方法总量)")
    for r in rows:
        print(f"  {r['name']:9s} 学生推理 {100*r['f_s']/r['flops']:4.1f}%  教师推理 {100*r['f_t']/r['flops']:4.1f}%  训练 {100*r['f_tr']/r['flops']:4.1f}%")
    c = next(r for r in rows if r["name"] == "CriticOPD")
    print(f"\n三、CriticOPD 前 40 步:批改 {c['tok']['批改次数']:.0f} 次,修复 {c['tok']['修复次数']:.0f} 次"
          f"(共 {40*B} 个样本)")
    json.dump(rows, open(f"{R}/criticopd/cost_table.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
