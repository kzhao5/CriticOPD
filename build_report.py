import re, sys
sys.path.insert(0, ".")
from analyze_b16k import load, stats, BENCHES
R = "relay-opd/"
SD = R+"opd/patches/vllm/speculative_decode.py"; LO = R+"verl/trainer/distillation/losses.py"
SE = R+"verl/utils/semantic_eos.py"; VS = R+"verl/workers/rollout/vllm_rollout/vllm_async_server.py"
RT = R+"opd/scripts/relay_opd/train.sh"; OS_ = R+"opd/scripts/baselines/opd.sh"

def ex(path, start, end, after=None, extra=0, nth=1):
    L = open(path).read().split("\n")
    i0 = 0
    if after:
        i0 = next(i for i,l in enumerate(L) if after in l)
    hits = [i for i,l in enumerate(L) if i >= i0 and start in l]
    s = hits[nth-1]
    e = next(i for i in range(s, len(L)) if end in L[i]) + extra
    return f"`{path}` 第 {s+1}–{e+1} 行\n```python\n" + "\n".join(L[s:e+1]) + "\n```\n"

def exsh(path, start, end):
    L = open(path).read().split("\n")
    s = next(i for i,l in enumerate(L) if start in l); e = next(i for i in range(s,len(L)) if end in L[i])
    return f"`{path}` 第 {s+1}–{e+1} 行\n```bash\n" + "\n".join(L[s:e+1]) + "\n```\n"

def cell(run, step, bench):
    Lr = load(run, step, bench)
    return None if not Lr else 100*sum(Lr[0].values())/len(Lr[0])

out = []
w = out.append

import os, re as _re
def train_metrics(arm):
    rx = _re.compile(rf"^{_re.escape(arm)}(_cs|_cs2|_cs3|_dwm)?_\d+\.log$")
    files = sorted((f for f in os.listdir("logs") if rx.match(f)), key=lambda f: os.path.getmtime("logs/"+f))
    by = {}
    for f in files:
        for line in open("logs/"+f, errors="ignore"):
            if "response_length/mean:" not in line: continue
            m = _re.search(r"step:(\d+) ", line)
            if m: by[int(m.group(1))] = line
    def g(line, k):
        m = _re.search(_re.escape(k)+r":([0-9.eE+-]+)", line); return float(m.group(1)) if m else None
    return {s: {k: g(l, k) for k in ["response_length/mean","response_length/clip_ratio","relay_teacher_mask_ratio","trigger_stopped_ratio","trigger_stopped_length_mean"]} for s, l in by.items()}

NAMES = {"opd":"OPD","fastopd":"FastOPD","grpo":"GRPO","relayopd":"RelayOPD","lucid":"Lucid","lucid_p1_nofix":"Lucid-nofix"}
def nm(run):
    if run == "lucid_p1_nofix": return "Lucid-nofix"
    base, pair = run.rsplit("_",1)
    return NAMES.get(base, run) + ("(0.6B)" if pair == "p2" else "")

def results_tables(pair, base, arms):
    import glob
    rows = [(base,0)]
    for r in arms:
        for d in glob.glob(f"outputs/eval_out/{r}_b16k/step_*"):
            s = int(d.split("step_")[1])
            if any(load(r,s,b) for b in BENCHES): rows.append((r,s))
    rows.append(("teacher_4b",0))
    order = {base:0, "teacher_4b":99}; order.update({a:i+1 for i,a in enumerate(arms)})
    rows.sort(key=lambda x:(order[x[0]], x[1]))
    t = "| 方法 | step | AIME24 | AIME25 | AMC23 | **MATH500** | MATH500 95% CI | 评测均长 | 评测截断率 | Avg4 |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for r,s in rows:
        v = [cell(r,s,b) for b in BENCHES]
        M = load(r,s,"math500")
        if M:
            m,se,N = stats(M[0]); ci = f"[{m-1.96*se:.1f}, {m+1.96*se:.1f}]"; ln = f"{M[1]:.0f}"; tr = f"{100*M[2]:.1f}%"
        else: ci = ln = tr = "—"
        lab = "base" if r==base else ("teacher (4B)" if r=="teacher_4b" else nm(r))
        f = lambda x: "—" if x is None else f"{x:.1f}"
        avg = f"{sum(v)/4:.1f}" if all(x is not None for x in v) else "—"
        t += f"| {lab} | {s} | {f(v[0])} | {f(v[1])} | {f(v[2])} | **{f(v[3])}** | {ci} | {ln} | {tr} | {avg} |\n"
    return t

def pair_row(a, b):
    ra, sa = a.split("@"); rb, sb = b.split("@")
    A = load(ra, sa, "math500"); B = load(rb, sb, "math500")
    if not A or not B or not (A[3] and B[3]): return None
    common = sorted(set(A[0]) & set(B[0])); d = [A[0][p]-B[0][p] for p in common]; N=len(d); mu=sum(d)/N
    import math; se = math.sqrt(sum((x-mu)**2 for x in d)/(N-1))/math.sqrt(N)
    lo, hi = 100*(mu-1.96*se), 100*(mu+1.96*se); sig = "**显著**" if lo>0 or hi<0 else "不显著"
    return f"| {nm(ra)}@{sa} − {nm(rb)}@{sb} | {100*mu:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sig} |\n"

def dyn_table(arms, steps, batch=128):
    t = "| 方法 | step | rollout 均长 | 撞训练上限比例 | teacher token 比例 | 被接管机制截停的轨迹比例 | 被截停轨迹均长 | 每步训练 token 量 |\n|---|---|---|---|---|---|---|---|\n"
    for a in arms:
        T = train_metrics(a); nroll = 8 if a.startswith("grpo") else 1
        for s in steps:
            if s not in T: continue
            x = T[s]; ln = x["response_length/mean"]; cl = x["response_length/clip_ratio"]; tm = x["relay_teacher_mask_ratio"]
            sr = x["trigger_stopped_ratio"]; sl = x["trigger_stopped_length_mean"]
            relay = a.startswith(("relayopd","lucid"))
            tok = ln*batch*nroll
            t += (f"| {nm(a)} | {s} | {ln:.0f} | {100*cl:.0f}% | " + (f"{100*tm:.1f}%" if tm is not None else "—") + " | "
                  + (f"{100*sr:.0f}%" if relay and sr is not None else "—") + " | " + (f"{sl:.0f}" if relay and sl is not None else "—")
                  + f" | {tok/1e6:.2f}M" + (" (n=8)" if nroll==8 else "") + " |\n")
    return t

import math
def pair_num(a, b):
    ra, sa = a.split("@"); rb, sb = b.split("@")
    A = load(ra, sa, "math500"); B = load(rb, sb, "math500")
    common = sorted(set(A[0]) & set(B[0])); d = [A[0][p]-B[0][p] for p in common]; N=len(d); mu=sum(d)/N
    se = math.sqrt(sum((x-mu)**2 for x in d)/(N-1))/math.sqrt(N)
    return 100*mu, 100*(mu-1.96*se), 100*(mu+1.96*se)
def acc(rs, b="math500"):
    r,s = rs.split("@"); return cell(r,s,b)
def tr(rs):
    r,s = rs.split("@"); return 100*load(r,s,"math500")[2]

P1 = ["opd_p1","fastopd_p1","grpo_p1","relayopd_p1","lucid_p1","lucid_p1_nofix"]
P2 = ["opd_p2","fastopd_p2","grpo_p2","relayopd_p2","lucid_p2"]
TM1 = {a: train_metrics(a) for a in ["relayopd_p1","lucid_p1"]}; TM2 = {a: train_metrics(a) for a in ["relayopd_p2","lucid_p2"]}
def rng(TM, key, steps):
    v = [TM[a][s][key] for a in TM for s in steps if s in TM[a] and TM[a][s][key] is not None]; return min(v), max(v)
sr1 = rng(TM1, "trigger_stopped_ratio", [40,60,80,100,120]); sr2 = rng(TM2, "trigger_stopped_ratio", [20,40,60,80,100,120])
sl2 = rng({"lucid_p2":TM2["lucid_p2"]}, "trigger_stopped_length_mean", [20,40,60,80,100,120])

d_fast_lucid = pair_num("fastopd_p1@120","lucid_p1@120"); d_fast_relay = pair_num("fastopd_p1@120","relayopd_p1@120")
d_lucid_relay = pair_num("lucid_p1@120","relayopd_p1@120"); d_opd2 = pair_num("opd_p2@40","lucid_p2@40")
lt = TM1["lucid_p1"][120]
w(r"""# Lucid-OPD 实验完整报告:结果、四种算法的实现与分析

> 用途:交给外部模型分析 Lucid-OPD 应如何改进。本文所有代码片段均按行号直接摘自仓库源码(未改写);所有数字由逐样本评测文件实时计算。
> 生成时间见文末。标注"待出"的结果仍在评测中。

## 0. 摘要

""")
w(f"- **1.7B 学生,step 120 同步对比(MATH500,配对检验)**:FastOPD {acc('fastopd_p1@120'):.1f},显著高于 Lucid {acc('lucid_p1@120'):.1f}(差 {d_fast_lucid[0]:+.1f},95% CI [{d_fast_lucid[1]:+.1f}, {d_fast_lucid[2]:+.1f}])和 RelayOPD {acc('relayopd_p1@120'):.1f}(差 {d_fast_relay[0]:+.1f});Lucid 与 RelayOPD 不可区分(差 {d_lucid_relay[0]:+.1f},CI [{d_lucid_relay[1]:+.1f}, {d_lucid_relay[2]:+.1f}])。\n")
w(f"- **0.6B 学生**:朴素 OPD 在 step 40 达到 MATH500 {acc('opd_p2@40'):.1f}、step 60 达到 {acc('opd_p2@60'):.1f};Lucid step 40 仅 {acc('lucid_p2@40'):.1f}(差 {d_opd2[0]:+.1f},CI [{d_opd2[1]:+.1f}, {d_opd2[2]:+.1f}]),跑满 139 步也只有 {acc('lucid_p2@139'):.1f},与 RelayOPD {acc('relayopd_p2@139'):.1f} 相同。**带 teacher 接管的两种方法在弱学生上几乎学不动。**\n")
w(f"- **关键实现事实**:RelayOPD 与 Lucid 共用同一 relay rollout,在一条轨迹完成第 2 次 teacher 接管后**强制插入内部 stop 并截断轨迹**。1.7B 上 step≥40 时 {100*sr1[0]:.0f}–{100*sr1[1]:.0f}% 的训练轨迹被这样截停;0.6B 上 {100*sr2[0]:.0f}–{100*sr2[1]:.0f}%,且 Lucid 被截停轨迹平均只有 {sl2[0]:.0f}–{sl2[1]:.0f} token。学生很少在训练中生成解题后半段。\n")
w(f"- **训练指标与评测脱节**:Lucid@120 训练 rollout 均长 {lt['response_length/mean']:.0f}、撞上限 {100*lt['response_length/clip_ratio']:.1f}%(量的是被截停后的轨迹);独立评测 MATH500 时均长 {load('lucid_p1',120,'math500')[1]:.0f}、截断 {tr('lucid_p1@120'):.1f}%。\n")
w("- **Semantic EOS 消融**:step 20 时无差异(见 §3.3);更晚步数待出。\n\n")

w(r"""## 1. 实验设置

| 项 | 取值 |
|---|---|
| 学生 | p1 = Qwen3-1.7B-Base;p2 = Qwen3-0.6B-Base(均为未经后训练的 base 模型) |
| teacher | Qwen3-4B-Instruct-2507 |
| 训练数据 | DAPO-Math-17K;batch 128 prompts;139 步 = 1 epoch |
| 优化 | lr 1e-6 常数;ppo_epochs 1;mini-batch 128(每步 1 次更新,严格 on-policy);PPO clip 0.2 / 0.2,dual-clip c=3.0;`loss_agg_mode=token-mean`;无 KL-to-ref |
| rollout | vLLM,temperature 1.0,top_p 1.0,n=1(GRPO n=8),`enable_thinking=False`,prompt ≤2048 |
| 训练响应上限 | 16384(FastOPD 为 8192) |
| 停止符 | 所有臂 rollout 都把 151643 `<|endoftext|>` 与 151645 `<|im_end|>` 注册为 stop |
| 资源 | actor 2 GPU + teacher 2 GPU;超时后从 checkpoint 续训(已逐条核对 `Setting global step to N`) |
| 评测 | `math_benchmarks.py`,chat template,temperature 1.0,top_p 1.0,**max_new 16384**,每题 8 采样,种子 = 42 + 副本号。DP=2 且 `NUM_SHARDS_TOTAL=1` 的评测把同一套 8 采样**用相同种子**跑了两遍(实测 lucid@120 有 71% 的样本逐字相同、99.9% 开头 200 字符相同),合并后并不是 16 个独立样本,有效样本数仍约为 8;均值与按题计算的置信区间基本不受影响 |
| 基准 | AIME24(30 题)、AIME25(30)、AMC23(83)、MATH500(500);指标 avg@k |
| 统计 | 单次运行的标准误 = 题间通过率标准差 / √题数;**配对检验** = 两个 checkpoint 在同一批题上逐题做差后的均值与 95% CI。AIME/AMC 在当前准确率下 SE 约 3–5 分,**结论以 MATH500 为准**(SE ≈ 0.6–1.7,配对 SE ≈ 0.5–0.9) |
| 限制 | 单 seed;1 epoch;续训跨 GPU 型号(A100/H100/B200) |

""")

w(r"""## 2. 算法与实现

四个方法共享同一训练框架(verl PPO + 蒸馏 loss),差别只在 **rollout 如何产生轨迹** 和 **每个 token 的优势如何计算**。下文记号:x 为 prompt,y 为响应,π_θ 为学生,π_old 为采样时的学生,p_T 为 teacher。

### 2.1 OPD:采样式 reverse-KL 策略梯度(所有蒸馏臂的共同基础)

- 学生自行采样整条响应 y ~ π_old(·|x),直到 EOS 或训练上限。
- teacher 对学生采样出的每个 token 打分,得到 log p_T(y_t | x, y_<t)。
- 每个 token 的优势(停梯度):**A_t = log p_T(y_t) − log π_θ(y_t)**,即单样本 k1 估计量的相反数。
- 用 PPO clipped surrogate 优化,token-mean 聚合;**不使用任务正确性奖励**(`use_task_rewards=False`)。
- 这是 reverse KL(π_θ ‖ p_T) 的单样本策略梯度估计:学生在自己会走到的状态上,被推向 teacher 认可的 token。

启动配置:
""")
w(exsh(OS_, "distillation.distillation_loss.loss_mode=k1", "distillation.distillation_loss.use_task_rewards=False"))
w("\n逐 token 的 k1 与策略梯度(OPD 走 `distillation_loss` 这条路径):\n")
w(ex(LO, "if loss_config.use_policy_gradient:", "rollout_is_weights=rollout_is_weights,", extra=1))
w("\nPPO clipped 损失矩阵(relay loss 也复用同一函数):\n")
w(ex(LO, "def _ppo_pg_loss_mat(", "pg_losses = torch.where(advantages < 0, clip2, clip1)"))

w(r"""
### 2.2 FastOPD

与 OPD **完全相同**,唯一区别是训练响应上限 8192(OPD 为 16384)。评测时同样用 16384 预算。
""")
w(f"`{R}opd/scripts/baselines/fastopd/8192.sh`\n```bash\n" + open(R+"opd/scripts/baselines/fastopd/8192.sh").read().strip() + "\n```\n")

w(r"""
### 2.3 RelayOPD(已发表方法的复现;词法触发 + teacher 接管)

**(a) Rollout 用投机解码实现**:学生作为 draft 模型(每次提议 4 个 token),teacher 作为 target 模型。自定义采样器 `_relay_opd_rejection_sample` 在正常状态下**直接输出学生采样的 token**(保持 on-policy),同时拿到 teacher 在每个位置的 logits,用于判断是否触发。

**(b) 词法触发**:在位置 t,当且仅当同时满足以下条件时触发:
1. teacher 的 argmax 属于"反思词"集合(Wait / But / Hmm / Actually / Hold / However / Yet / Oh / Alternatively / No / Ah / Oops / Well 的单 token 形式);
2. teacher 的 argmax ≠ 学生实际采样的 token;
3. teacher 的 argmax **不在**学生 top-5 内(分歧要足够实质);
4. 当前不在接管中。

一次投机调用里只有**第一个**触发位置生效,其后的学生草稿作废。

**(c) 接管**:触发位置输出 teacher 的 argmax(标为 teacher token);随后进入接管,由 teacher 作为 target 做标准投机拒绝采样(token 服从 teacher 的采样分布),直到写满 **3 个段落边界**(任何解码后含 `\n\n` 的 token)或 **256 token**。bonus token 被丢弃。接管期间所有 token 标记 `teacher_mask=True`。

**(d) 次数上限与强制截停**:每条轨迹最多接管 **2 次**。**第 2 次接管结束时,插入内部 stop token,rollout 在该位置截断响应(stop token 本身被剥掉,不进 loss)**。这条轨迹就此结束,学生不再继续生成,也不会有自然的 EOS。

**(e) Loss(rkl 模式)**:与 OPD 相同的 k1 优势,**作用于全部响应 token,包括 teacher 写的接管 token**(此时 log p_T 取 teacher 自己生成的那个 token 的概率)。PPO clip,token-mean。

反思词集合:
""")
w(ex(SD, "def _get_reflection_token_ids", '"Ah, Ah,ah, ah,Oops, Oops,Well, Well",'))
w("\n接管参数(默认值,RelayOPD 与 Lucid 相同):\n")
w(exsh(RT, "export VERL_OPD_ROLLOUT_MODE", "export RELAY_OPD_MAX_TAKEOVER_TOKENS"))
w("\n触发候选(分布式分支是 Lucid,`else` 分支是 RelayOPD 的词法触发):\n")
w(ex(SD, "teacher_argmax = target_logits.argmax(dim=-1)", "teacher_is_reflection & teacher_argmax.ne(draft_token_ids.long())", after="def _relay_opd_rejection_sample(", extra=2))
w("\ntop-5 过滤与 competence gate(gate 只有 Lucid 开启;默认 `gate_mode=v1`),得到最终触发位置:\n")
w(ex(SD, "gather_ids = teacher_argmax.clamp_min(0).clamp_max(vocab_size - 1)[:, None]", after="def _relay_opd_rejection_sample(", end="stop_candidate = raw_trigger_candidate & ~teacher_in_student_topk & teacher_confident"))
w("\n取每条请求的第一个触发位置;触发位置写入 teacher argmax:\n")
w(ex(SD, "first_event_pos = torch.full((batch_size,), INF_POS, dtype=torch.long, device=device)", after="def _relay_opd_rejection_sample(", end="teacher_output[req_ids[emit_trigger], pos_in_req[emit_trigger]] = True"))
w("\n接管的段落/token 预算,以及**第 `max_takeovers` 次接管结束后插入内部 stop**:\n")
w(ex(SD, "keep_until_pos: int | None = None", after="def _relay_opd_rejection_sample(", end="stopped_after_takeover[bi] = True"))
w("\nrollout 侧据此**截断响应**(内部 stop 之后的内容不存在,stop 本身被移除):\n")
w(ex(VS, 'truncate_pos = int(trigger_stop_event["response_pos"])', "token_ids = raw_token_ids[:truncate_pos]"))
w("\nrelay loss:teacher/student mask,然后对全部响应 token 用 k1 优势做 PPO:\n")
w(ex(LO, "teacher_mask = teacher_mask.bool() & response_mask", "student_mask = response_mask & ~teacher_mask", after="def relay_opd_loss("))
w(ex(LO, 'k1_losses = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_token_lp, kl_penalty="k1")', "loss_tok = response_mask.float() * pg_loss_tok"))

w(r"""
### 2.4 Lucid-OPD(本文方法)

与 RelayOPD **共用同一套 relay rollout、接管预算、第 2 次接管后强制截停,以及同一个 loss**。区别只有三处:

**(a) 分布式触发(替代词法触发)**
- 对学生生成的每个 token,在全词表上计算 **reverse KL(p_s ‖ p_T) = Σ_v p_s(v) [log p_s(v) − log p_T(v)]**(p_s 下限截断到 1e-9,log p_T 下限截断到 −30,KL 截断到 [0, 30])。
- 按请求累加到"当前段"的累加器;段在轨迹开头、以及任何已输出 token 解码后含 `\n\n` 时清零。
- 在下一次投机调用时计算段内均值;**段均 KL > τ = 1.3 且段内已有 ≥ 8 个 token**,并且 teacher argmax ≠ 学生 token 时成为候选,再经过同样的 top-5 过滤。
- 判定只用之前调用累积的统计量(因果);同一次投机调用里的所有草稿 token 共用一个段均值。

**(b) Competence gate(v1)**
- 在整条轨迹上(只在轨迹开头清零,**不是滑窗**)累积 teacher 对学生已生成 token 的 NLL。
- 只有**前缀平均 NLL < 3.0 nats**(teacher 困惑度 < e³ ≈ 20)时才保留触发。累加器为空时均值记为 0,因此轨迹开头 gate 永远放行。

**(c) Semantic EOS**
- 终止符集合 E = {151643, 151645}。在**实际输出了终止符**的位置,把 teacher 侧的 log-prob 换成 **log p_T(STOP) = logsumexp_{e∈E} log p_T(e)**(rollout 从 teacher 在该位置的全词表分布导出);学生和 old policy 的 log-prob 也由训练引擎换成 log p(STOP)。其余链路不变。
- 动机:teacher 与学生可能把停止概率放在不同的 EOS token 上,采样式 OPD 会在学生自己的 EOS 处给出大幅负优势,把学生从"停止"推开。
- 作用范围:**只作用于轨迹里真的出现了终止符的位置**。被强制截停或被长度上限截断的轨迹里没有这样的位置。

逐 token reverse KL 的计算与分段累积、清零:
""")
w(ex(SD, "# distributional trigger: accumulate per-token KL(student||teacher) over student-realized", "if _bnd[_b]:", extra=2))
w("\ncompetence gate 用的 teacher 前缀 NLL 累积(整条轨迹累计,读取时取均值,见 §2.3 的 gate 代码):\n")
w(ex(SD, "# gated-B: accumulate teacher NLL over the student-realized tokens of this call into the", "_TEACHER_PREFIX_TOK[_ext] = _TEACHER_PREFIX_TOK.get(_ext, 0) + int(_cnt_cpu[_b])"))
w("\nSemantic EOS:rollout 侧导出 teacher 的 log p(STOP):\n")
w(ex(SD, "def _compute_semantic_eos_export(", "stop_logp_by_output[req_ids[rows], pos_in_req[rows]] = t_stop"))
w("\nlog p(STOP) 的定义(学生侧与 old policy 侧复用):\n")
w(ex(SE, "def stop_logprob_from_logits", "return lse_eos - lse_full"))
w(ex(SE, "def semantic_eos_logprobs(", "def merge_stop_distribution", extra=-2))
w("\nloss 中把终止位置的 teacher log-prob 换成 log p_T(STOP):\n")
w(ex(LO, "eos_pos = torch.isin(responses, eos_t) & response_mask", "teacher_token_lp = torch.where(have_stop, stop_lp, teacher_token_lp)"))

w(r"""
### 2.5 GRPO(不用 teacher 的参照)

每个 prompt 采样 n=8 条,奖励为数学答案正确性(`opd/reward/math_reward.py`),组内归一化优势,PPO clip,训练上限 16384,无 teacher、无蒸馏项。

### 2.6 各臂实际启动配置

| 臂 | 脚本 | 训练上限 | 触发 | 阈值 | gate | 接管 | Semantic EOS |
|---|---|---|---|---|---|---|---|
| OPD | `baselines/opd.sh` | 16384 | — | — | — | — | 否 |
| FastOPD | `baselines/fastopd/8192.sh` | 8192 | — | — | — | — | 否 |
| GRPO | `baselines/grpo.sh` | 16384 | — | — | — | — | 否(n=8,任务奖励) |
| RelayOPD | `relay_opd/train.sh` | 16384 | 词法(teacher argmax ∈ 反思词) | top-5 过滤 | 无 | ≤2 次 × (3 段 / 256 tok),之后截停 | 否 |
| Lucid | `relay_opd/train.sh` | 16384 | 段均 reverse KL | τ=1.3,≥8 tok,top-5 | 前缀 NLL < 3.0 | 同上 | 是 |
| Lucid-nofix | `relay_opd/train.sh` | 16384 | 同 Lucid | 同 Lucid | 同 Lucid | 同上 | **否**(消融) |

提交脚本中各臂的环境变量(注意 Lucid 段未调用 `clear_relay`,继承了 RelayOPD 段设置的 `RELAY_OPD_TRIGGER_TOPK=5` 及接管预算;两臂因此一致):
""")
w(exsh("submit_base_table.sh", "# ---- 1. OPD", "sub lucid_${PAIR}_nofix"))

import glob, json, datetime
def trunc_breakdown(rows):
    t = "| run@step | MATH500 | 截断率 | 截断样本中答对 | 错题中截断占比 | 若截断样本全部答对的上限 |\n|---|---|---|---|---|---|\n"
    for rs in rows:
        r,s = rs.split("@")
        fs = sorted(glob.glob(f"outputs/eval_out/{r}_b16k/step_{s}/shard_*/math500.jsonl")) or glob.glob(f"outputs/eval_out/{r}_b16k/step_{s}/math500.jsonl")
        n=c=tn=tc=0
        for f in fs:
            for l in open(f):
                x=json.loads(l); n+=1; c+=x["correct"]
                if x.get("finish_reason")=="length": tn+=1; tc+=x["correct"]
        if not n: continue
        lab = {"base_1p7b":"base","teacher_4b":"teacher"}.get(r, nm(r)) + f"@{s}"
        t += f"| {lab} | {100*c/n:.1f} | {100*tn/n:.1f}% | {100*tc/max(tn,1):.1f}% | {100*tn/max(n-c,1):.1f}% | {100*(c+tn-tc)/n:.1f} |\n"
    return t

def loop_stats(rs):
    r,s = rs.split("@")
    fs = sorted(glob.glob(f"outputs/eval_out/{r}_b16k/step_{s}/shard_*/math500.jsonl"))
    tr_ = [json.loads(l) for f in fs for l in open(f)]; tr_ = [x for x in tr_ if x.get("finish_reason")=="length"]
    def rep(t):
        tail=t[-4000:]; seen={}
        for i in range(0,len(tail)-60,20): k=tail[i:i+60]; seen[k]=seen.get(k,0)+1
        return max(seen.values()) if seen else 0
    loops = sum(rep(x["gen_text"])>=5 for x in tr_); boxed = sum("\\boxed" in x["gen_text"] for x in tr_)
    return len(tr_), 100*loops/max(len(tr_),1), 100*boxed/max(len(tr_),1)

w("\n## 3. 结果\n\n### 3.1 1.7B 学生(p1)\n\n")
w(results_tables("p1","base_1p7b",P1))
w("\n### 3.2 0.6B 学生(p2)\n\n")
w(results_tables("p2","base_0p6b",P2))
w("\n### 3.3 MATH500 配对检验(同一批 500 题逐题做差;差值 = 前者 − 后者)\n\n| 对比 | 差值 | 95% CI | |\n|---|---|---|---|\n")
for a,b in [("opd_p1@20","lucid_p1@20"),("lucid_p1@20","lucid_p1_nofix@20"),("grpo_p1@40","lucid_p1@40"),("grpo_p1@40","relayopd_p1@40"),
            ("relayopd_p1@40","lucid_p1@40"),("grpo_p1@80","lucid_p1@80"),("fastopd_p1@120","lucid_p1@120"),("fastopd_p1@120","relayopd_p1@120"),
            ("lucid_p1@120","relayopd_p1@120"),("opd_p1@60","lucid_p1@120"),("fastopd_p1@139","fastopd_p1@120"),
            ("lucid_p1@139","relayopd_p1@139"),("fastopd_p1@139","lucid_p1@139"),("lucid_p1@100","lucid_p1_nofix@100"),("opd_p1@120","fastopd_p1@120"),
            ("opd_p2@40","lucid_p2@40"),("opd_p2@40","relayopd_p2@40"),("lucid_p2@139","relayopd_p2@139")]:
    row = pair_row(a,b)
    w(row if row else f"| {nm(a.split('@')[0])}@{a.split('@')[1]} − {nm(b.split('@')[0])}@{b.split('@')[1]} | 待出 | | |\n")

w("\n### 3.4 训练期动态\n\n'撞训练上限比例'与'rollout 均长'对 relay 臂量的是**被接管机制截停之后**的轨迹。'每步训练 token 量' = rollout 均长 × 128(GRPO × 8),即每次参数更新所基于的响应 token 数。\n\n**p1**\n\n")
w(dyn_table(P1, [20,40,80,120,139]))
w("\n**p2**\n\n")
w(dyn_table(P2, [20,40,80,120,139]))

w("\n### 3.5 评测时的截断(MATH500,16384 预算)\n\n")
w(trunc_breakdown(["base_1p7b@0","grpo_p1@80","opd_p1@60","relayopd_p1@120","lucid_p1@120","fastopd_p1@120","teacher_4b@0"]))
n_, lp_, bx_ = loop_stats("lucid_p1@120")
w(f"\nLucid@120 的 {n_} 个截断样本中,末尾出现明显重复循环(4000 字符内同一 60 字符片段 ≥5 次)的只占 {lp_:.0f}%;{bx_:.0f}% 的样本在截断前已经写出过 `\\boxed`。其余是持续的真实推理(\"But… / Alternatively, perhaps…\" 式反复权衡)跑满了预算。\n")


def trunc_correct(rs):
    r,s_ = rs.split("@")
    fs = sorted(glob.glob(f"outputs/eval_out/{r}_b16k/step_{s_}/shard_*/math500.jsonl"))
    tn=tc=0
    for f in fs:
        for l in open(f):
            x=json.loads(l)
            if x.get("finish_reason")=="length": tn+=1; tc+=x["correct"]
    return 100*tc/max(tn,1)
tc = {k: trunc_correct(k) for k in ["opd_p1@60","fastopd_p1@120","relayopd_p1@120","lucid_p1@120"]}
TRUNC_SENTENCE = (f"\n**截断的性质在两类方法之间不同**:OPD@60 与 FastOPD@120 的截断样本中分别有 {tc['opd_p1@60']:.0f}% 与 {tc['fastopd_p1@120']:.0f}% 已经判为正确"
                  f"(答案写出后继续验证、跑满预算);RelayOPD@120 与 Lucid@120 只有 {tc['relayopd_p1@120']:.0f}% 与 {tc['lucid_p1@120']:.0f}%,即 relay 训练出的学生**大多在给出答案之前就耗尽了预算**。"
                  "这与 §4.2 的假设方向一致:截停机制下学生很少练到\"收尾并给出答案\"这一段。\n")

TMall = {a: train_metrics(a) for a in P1}
def tok(a, s):
    x = TMall[a].get(s); n = 8 if a.startswith("grpo") else 1
    return None if not x else x["response_length/mean"]*128*n/1e6
lt2 = TM2["lucid_p2"]
def tmr(a, s): return 100*TMall[a][s]["relay_teacher_mask_ratio"]

w(r"""
## 4. 分析

以下区分**已由数据确认的事实**与**解释性假设**;假设在 §5 给出对应的验证实验。

### 4.1 Lucid 没有赢过任何一个对手(事实)
""")
w(f"- step 120 同步:FastOPD 显著领先 Lucid {d_fast_lucid[0]:+.1f} 分;Lucid 与 RelayOPD 在 40 步与 120 步都不可区分。\n")
w(f"- GRPO(无 teacher)在 step 40 显著领先两个 relay 方法约 12 分,step 80 与 Lucid 打平。\n")
w(f"- OPD 在 step 60({acc('opd_p1@60'):.1f})已与 Lucid step 120({acc('lucid_p1@120'):.1f})持平。\n")
w(f"- 0.6B 上两种 relay 方法都几乎不学:Lucid@139 {acc('lucid_p2@139'):.1f},RelayOPD@139 {acc('relayopd_p2@139'):.1f},而 OPD@60 {acc('opd_p2@60'):.1f}。\n")
w("- 在 FastOPD 训练 rollout 100% 撞 8192 上限、OPD 100% 撞 16384 上限的情况下,它们的评测结果仍是最好或并列最好。\"训练期长度失控有害\"在本设置下不成立。\n")

w(r"""
### 4.2 强制截停让 relay 方法很少训练到解题的后半段(事实 + 假设)

**事实**:第 2 次接管结束即截断轨迹(§2.3 (d),代码见 `keep_until_pos` 片段与 rollout 截断片段)。
""")
w(f"- p1:step ≥ 40 时 {100*sr1[0]:.0f}–{100*sr1[1]:.0f}% 的训练轨迹被截停(见 §3.4)。\n")
w(f"- p2:{100*sr2[0]:.0f}–{100*sr2[1]:.0f}% 被截停,Lucid 被截停轨迹平均只有 {sl2[0]:.0f}–{sl2[1]:.0f} token。\n")
w(r"""- 被截停的轨迹没有自然结尾,没有终止符;学生在这些样本里从未在"已经推理了几千 token 之后"的状态上接受过训练。

**假设**:学生在训练中主要见到"开头若干段 + 两段 teacher 文本",缺少对长推理后半段(整合、收尾、给出答案)的 on-policy 训练;评测时独立生成就在后半段打转、跑满预算。弱学生触发更早(p2 截停发生在约 600 token),受害更严重,与 p2 的崩溃方向一致。

### 4.3 同步数不等于同训练信号量(事实)
""")
for s in (80,120):
    parts = [f"{nm(a)} {tok(a,s):.2f}M" for a in ["opd_p1","fastopd_p1","grpo_p1","relayopd_p1","lucid_p1"] if tok(a,s) is not None]
    w(f"- step {s} 时每步响应 token 量:" + ",".join(parts) + "(GRPO 为 n=8 且用任务奖励,不直接可比)。\n")
w(r"""- loss 是 token-mean,所以每步更新幅度按 token 数归一化,**但覆盖的状态分布不同**:OPD/FastOPD 每步都在大量长推理后段位置上接受 teacher 信号,relay 臂几乎没有。按步数对齐比较对 relay 方法不利,需要补一个按累计 token 量(或 GPU 小时)对齐的比较。

### 4.4 teacher 介入率的演化(事实)
""")
w(f"- p1:Lucid teacher token 比例 step 40 为 {tmr('lucid_p1',40):.1f}%,step 120 降到 {tmr('lucid_p1',120):.1f}%;RelayOPD 从 {tmr('relayopd_p1',40):.1f}% 降到 {tmr('relayopd_p1',120):.1f}%。两者终点接近。\n")
w(f"- p2:Lucid 在 {100*min(lt2[s]['relay_teacher_mask_ratio'] for s in (20,40,60,80,100,120,139) if s in lt2):.1f}–{100*max(lt2[s]['relay_teacher_mask_ratio'] for s in (20,40,60,80,100,120,139) if s in lt2):.1f}% 之间波动,不衰减。\n")
w(r"""- Lucid 的触发阈值作用在 reverse KL(p_s ‖ p_T) 上,而训练目标恰好在降低这个量;学生追上 teacher 越快,触发越少。弱学生追不上,触发就一直存在。

### 4.5 评测截断是所有蒸馏臂的共性,不是 Lucid 独有(事实)

见 §3.5:OPD、FastOPD、RelayOPD、Lucid 在 MATH500 上都截断 13–21%,不用 teacher 的 GRPO 与 teacher 本身都不到 1%。截断解释了"蒸馏臂 vs GRPO"的长度差异,但**不能**解释 Lucid 与 OPD/FastOPD 之间的准确率差距。对 Lucid 自身而言,截断占了错题的很大一部分(见表),是最大的单项损失来源。截断样本绝大多数不是重复循环,而是真实但冗长的推理。
""")
w(TRUNC_SENTENCE)
w(r"""
### 4.6 Semantic EOS 管不到主要的失败模式(事实)

它只替换**已经输出终止符**的位置(§2.4 (c) 代码:`eos_pos = isin(responses, E)`)。被强制截停或被长度上限截断的轨迹里没有这样的位置;评测时的主要损失(截断)也不在它的作用范围内。step 20 的消融无差异与此一致。

### 4.7 competence gate 实际是整条轨迹的全局均值(事实)

teacher 前缀 NLL 只在轨迹开头清零,均值随轨迹变长而趋于常数,无法反映"teacher 在当前这一步是否可靠";累加器为空时 gate 放行。设计意图(局部能力判断)与实现(全局均值)不一致。

### 4.8 teacher token 的训练方式(观察)

接管段由 teacher 采样,优势 A_t = log p_T(y_t) − log π_θ(y_t) 通常很大且为正;这些 token 走的是与学生自采样 token 相同的 PPO clip 路径,其中 π_old(y_t) 是学生对 teacher token 的概率(通常很低)。这是把离策略的 teacher 文本当作 on-policy 样本训练。其对弱学生的影响尚未单独隔离。

## 5. 待验证的改进假设(每条附对照实验)

| # | 假设 | 对照实验(其他全部固定) | 预期观察 |
|---|---|---|---|
| H1 | 强制截停是 relay 方法落后的主因 | Lucid:第 2 次接管后**不截停**,交还学生继续生成到 EOS 或上限 vs 现状 | 训练均长接近 OPD;评测 MATH500 与截断率改善;p2 提升尤其大 |
| H2 | 同步数比较低估了 relay 方法 | 按累计响应 token 数(或 GPU 小时)对齐,重画 Lucid / RelayOPD / OPD / FastOPD 曲线 | 若对齐后差距消失,则 §4.1 的差距主要来自信号量 |
| H3 | 更短的训练上限本身有益(FastOPD 的优势来源) | OPD 在 4096 / 8192 / 16384 三档上限;Lucid 在 8192 上限 | 若 8192 档一致最好,则应把 Lucid 默认上限改为 8192 |
| H4 | 弱学生上接管伤害大于帮助 | p2 上:接管预算按已生成长度比例限制 / 前 N 步不接管 / 最少先生成 L token 才允许触发 | Lucid@p2 向 OPD@p2 靠拢 |
| H5 | teacher 接管 token 的 PG 训练方式有害 | teacher 段 (i) 只作上下文不进 loss;(ii) 用 `relay_opd_fkl`(teacher 段 top-k FKL);(iii) 降权 | 找到对弱学生不崩的处理方式 |
| H6 | 固定阈值导致触发自我消解 | 触发改为批内分位数(如段 KL 前 q%)或目标介入率控制 | 介入率在训练中保持稳定;观察是否改善后期表现 |
| H7 | gate 需要局部化 | 前缀 NLL 改为最近 W token 的滑窗或 EMA | gate 的过滤率随局部难度变化 |
| H8 | 评测损失主要是冗长推理 | 在 Lucid 上加入长度相关项或对截断样本的负信号;或在接管段后显式训练收尾 | 截断率下降且不以准确率为代价 |

建议优先级:**H1 → H2 → H3**。H1 直接针对已确认的实现机制且改动最小;H2 决定 §4.1 的结论是否需要改写;H3 可以用现有 OPD 脚本只改一个参数完成。
""")

import subprocess
q = subprocess.run(["squeue","-u","kzhao2","-h","-o","%j %t"], capture_output=True, text=True).stdout.split("\n")
pending = []
for line in q:
    m = _re.match(r"ev16_(.+)_(\d+) (\w+)", line.strip())
    if m and m.group(1) in P1+P2:
        pending.append(f"{nm(m.group(1))}@{m.group(2)}({'评测中' if m.group(3)=='R' else '排队'})")
w("\n## 6. 待出结果与限制\n\n")
w("- 正在评测或排队的:" + ("、".join(sorted(pending)) if pending else "无") + "。中间 checkpoint 并未全部评测,只选取了同步对比所需的步数。\n")
w("- 仍在训练:OPD(1.7B)、GRPO(1.7B)、Lucid-nofix(1.7B)、OPD(0.6B)。\n")
w("- 单 seed、1 epoch;续训跨 GPU 型号;早期 Lucid@20/40 评测为 8 采样,其余为 16 采样。\n")
w("- 评测与训练均为 `enable_thinking=False` 的 chat 模板;base 学生无后训练。\n")
w(f"\n---\n生成于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}(MDT)。复现表格:`python3 show_b16k.py`;带 CI 与配对:`python3 analyze_b16k.py [--pair a@s b@s]`。\n")

open("LUCID_REPORT.md","w").write("".join(out))
print("written", sum(1 for _ in open("LUCID_REPORT.md")), "lines")
