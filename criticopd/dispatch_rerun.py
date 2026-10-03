#!/usr/bin/env python3
"""把主表全部基线 + OOD 表按统一协议重测,分发到能抢占 gstandby 的分区组。

数学:8 samples x 32768(aime24/aime25/amc23 用 32 samples)—— 与 CriticOPD 各 arm 同口径。
OOD :4 samples x 32768 —— 采样数沿用原 OOD 协议,只把预算从 8192/4096 放到 32768。
      原协议下 mmlu_pro_2k 截断 6.9%(SFT 13.2%)、mbpp 2.8%(SKD 7.5%),按实测的
      -0.25 分/1% 截断率,这已经是系统偏差。
分区组(一个 job 只能挂一个 QOS):
  A cs,cs2,cs3 / qos=cs   —— 能抢 gstandby,且没有单用户 CPU 上限
  B dw / qos=dw87          —— 同上
  C m13l,m13h / qos=gpu    —— 能抢,但分区 QOS 有 CPU 上限(m13l 128、m13h 192),少分一点
结果一律写进带 _u32 后缀的新目录,和旧协议的数据物理隔离。
用法: python3 dispatch_rerun.py [--dry]
"""
import os, subprocess, sys
DRY = "--dry" in sys.argv
ROOT = "/home/kzhao2/Relay-OPD"
M = "/home/kzhao2/OPD/model"
GROUPS = {
    "A": ["--partition=cs,cs2,cs3", "--qos=cs", "--exclude=cs-1-2"],   # cs-1-2 疑有残留进程占卡
    "B": ["--partition=dw", "--qos=dw87", "--exclude=dw-2-4,dw-1-3"],
    "C": ["--partition=m13l,m13h", "--qos=gpu", "--exclude=m13h-1-1"],
}
CYCLE = list("AAAAAAABBBC")          # 7:3:1,和各组可抢占卡数大致成比例

MATH = ["olympiad", "math500", "minerva", "amc23", "aime24", "aime25"]   # 慢的先投
NS = lambda b: 32 if b in ("aime24", "aime25", "amc23") else 8
# (run 标签, step, 模型路径或 None=用 checkpoint)
MATH_MODELS = [
    ("student_1p7b", 0, f"{M}/Qwen3-1.7B"), ("teacher_4b", 0, f"{M}/Qwen3-4B-Instruct-2507"),
    ("sft_pt17b", 40, None), ("seqkd_pt17b", 139, None), ("grpo_pt17b", 120, None),
    ("trd_pt17b", 40, None), ("skd_pt17b", 120, None), ("relay_1p7b", 60, None),
    ("gatedB_1p7b", 60, None), ("lucid_T1_from0_pt17b", 60, None),
    ("opd_1p7b", 80, None), ("fastopd8192_1p7b", 60, None), ("relay_1p7b", 40, None),
    ("lucid_T1donly_pt17b", 60, None),
    ("student_0p6b", 0, f"{M}/Qwen3-0.6B"),
    ("sft_pt06b", 139, None), ("seqkd_pt06b", 139, None), ("grpo_pt06b", 80, None),
    ("opd_pt06b", 40, None), ("trd_pt06b", 139, None), ("fastopd8192_pt06b", 40, None),
    ("skd_pt06b", 40, None), ("relay_pt06b", 60, None), ("lucid_T1donly_pt06b", 60, None),
]
OOD = ["mmlu_pro_2k", "mmlu_mini", "mbpp", "humanevalplus", "arc_c", "obqa"]
OOD_MODELS = [
    ("student_1p7b", 0, f"{M}/Qwen3-1.7B"), ("teacher_4b", 0, f"{M}/Qwen3-4B-Instruct-2507"),
    ("sft_pt17b", 40, None), ("seqkd_pt17b", 139, None), ("grpo_pt17b", 120, None),
    ("opd_1p7b", 80, None), ("trd_pt17b", 40, None), ("fastopd8192_1p7b", 60, None),
    ("skd_pt17b", 120, None), ("relay_1p7b", 40, None), ("lucid_T1donly_pt17b", 60, None),
    ("criticopd_R4_pt17b", 40, None),       # 我们的方法(当前最好的 R4@40)也放进 OOD 表
]
ck = lambda r, s: f"{ROOT}/outputs/checkpoints/{r}/global_step_{s}/actor/huggingface"

# 正在跑、但 summary 要跑完才落盘的,文件层面查不出,只能显式排除,否则会投一个
# 写同一目录的副本互相覆盖。13944155 正在跑 relay_1p7b@40 aime24。
INFLIGHT = set()   # 已改为按队列去重
# 队列里已有同名 job(在跑或在排)的一律跳过:summary 要跑完才落盘,不查队列就会投出
# 写同一目录的副本。job 名由 (run, step, bench) 唯一决定。
QUEUED = set(subprocess.run(["squeue", "-u", "kzhao2", "-h", "-o", "%j"],
                            capture_output=True, text=True).stdout.split())
jobs = []
for r, s, mp in MATH_MODELS:
    out = f"{r}_u32"
    for b in MATH:
        if os.path.exists(f"{ROOT}/outputs/eval_out/{out}/step_{s}/{b}.summary.json"): continue
        if (out, s, b) in INFLIGHT or f"rm_{r}_{s}_{b}" in QUEUED: continue
        if os.path.exists(f"{ROOT}/outputs/eval_out/{out}/step_{s}/{b}.nshards"): continue   # 已改为多分片
        env = dict(RUN=r, OUT_RUN=out, STEP=s, BENCHES=b, N_SAMPLES=NS(b), MAX_NEW=32768,
                   MAX_MODEL_LEN=34817, SHARD_BASE=0, NUM_SHARDS_TOTAL=1)
        if mp: env["MODEL_PATH"] = mp
        jobs.append(("eval_shard.sl", f"rm_{r}_{s}_{b}", env, mp or ck(r, s)))
for r, s, mp in OOD_MODELS:
    out = f"{r}_ood_u32"
    for b in OOD:
        if os.path.exists(f"{ROOT}/outputs/eval_out/{out}/step_{s}/{b}.summary.json"): continue
        if f"ro_{r}_{s}_{b}" in QUEUED: continue
        env = dict(RUN_NAME=out, STEP=s, MODEL=mp or ck(r, s), BENCHES=b, N_SAMPLES=4,
                   MAX_NEW=32768, MAX_MODEL_LEN=34817, SHARD_BASE=0, NUM_SHARDS_TOTAL=1)
        jobs.append(("eval_ood.sl", f"ro_{r}_{s}_{b}", env, mp or ck(r, s)))

miss = [j for j in jobs if not os.path.exists(j[3] + "/config.json")]
if miss:
    print("!!! 模型缺失,跳过:", sorted({j[3] for j in miss})); jobs = [j for j in jobs if j not in miss]
cnt = {"A": 0, "B": 0, "C": 0}
print(f"待提交 {len(jobs)} 个(数学 {sum(j[0]=='eval_shard.sl' for j in jobs)},OOD {sum(j[0]=='eval_ood.sl' for j in jobs)})")
# OOD 先投:它是真正会改数字的那部分
jobs.sort(key=lambda j: 0 if j[0] == "eval_ood.sl" else 1)
for i, (script, name, env, _) in enumerate(jobs):
    g = CYCLE[i % len(CYCLE)]; cnt[g] += 1
    exp = "ALL," + ",".join(f"{k}={v}" for k, v in env.items())
    nice = [] if "criticopd" in name else ["--nice=5000"]
    cmd = ["sbatch", "--parsable", *GROUPS[g], *nice, "--gres=gpu:1", "--cpus-per-task=4", "--mem=64G",
           "--time=12:00:00", f"--job-name={name}", f"--output={ROOT}/logs/%x_%j.log",
           f"--export={exp}", script]
    if DRY: continue
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode: print("!!", name, r.stderr.strip()[:200])
print("分组:", cnt, "(dry run,未提交)" if DRY else "(已提交)")
