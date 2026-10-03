#!/bin/bash
# R1 = 算力对齐对照:plain OPD,但每题多采样若干条 rollout,使总推理量与 R3(带 critic + 修复)相同。
# 没有它,R3/R4 就算赢了也分不清收益来自反馈还是来自多跑了 rollout。
#
# 从 opd_1p7b@40 fork,训到 step 80 —— 这样 R0 就是现成的 opd_1p7b@80(同数据同顺序),不用重跑。
# rollout.n 由离线报告算出:n = 1 + P(写完但答错) x (critic 成本 + 修复次数),四舍五入。
set -euo pipefail
cd /home/kzhao2/Relay-OPD
N=${ROLLOUT_N:-}
if [ -z "$N" ]; then
  N=$(/home/kzhao2/.conda/envs/relay-opd/bin/python3 - <<'PY'
import json, math, os
try:
    ro = json.load(open("/home/kzhao2/Relay-OPD/criticopd/out/rollout_stat.json"))
    p_wrong = ro["wrong_finished"] / max(ro["total"], 1)
    cs = json.load(open("/home/kzhao2/Relay-OPD/criticopd/out/critic_stat.json"))
    # critic 的成本按 token 数折算成 rollout 的等价份额
    crit = cs["mean_critic_tokens"] / max(ro["mean_tok"], 1)
    k_repair = int(os.environ.get("REPAIR_K", "3"))
    n = 1 + p_wrong * (crit + k_repair)
    print(max(1, round(n)))
except Exception:
    print(2)          # 离线还没结果时的保守默认
PY
)
fi
echo "[R1] rollout.n = $N  (算力对齐 R3)"
export ROLLOUT_N=$N
# opd.sh 默认 4 actor + 4 teacher = 8 张,而 8 卡 job 在本账户的 billing cap 下排不进队。
# 沿用 Lucid 的 2+2 配置(submit.sh 里记录过这个偏差):优化目标不变,只是 wall-clock 慢些。
# Ray 默认把运行时文件放 /tmp/ray。m13h-2-2 这类多人共用节点上会和别的 job 撞,
# 表现是 "GCS cannot find the node" + startup timeout(13915528 就是这么挂的)。
# 给每个 job 一个独立的 temp 目录。
export RAY_TMPDIR_BASE=/tmp/ray_${USER}
export TMPDIR=${TMPDIR:-/tmp}
export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-2} TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-2}
export STUDENT_MODEL=${STUDENT_MODEL:-/home/kzhao2/OPD/model/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507}
export METHOD_SCRIPT=opd/scripts/baselines/opd.sh
# 不 fork:checkpoint 只存了 actor/,没有 data.pt(dataloader 状态),从 @40 续训会让数据
# 顺序从头开始,R1 就会重看已见过的题,和 opd_1p7b@80(训的是 5120-10240)不再可比。
# 所有 arm 一律从 step 0 训到 80 —— 这样 opd_1p7b@80 本身就是 R0,完全等价。
NAME=${NAME_OVERRIDE:-criticopd_R1_n${N}_pt17b}
export OUTPUT_DIR=$PWD/outputs/checkpoints/$NAME EXP_ID=$NAME
export EXTRA_ARGS="trainer.total_training_steps=${CHUNK_END:-80}"
export SAVE_FREQ=${SAVE_FREQ:-40}
mkdir -p "$OUTPUT_DIR"
# QOS 必须显式给:指定单分区而不给 QOS 时默认变成 normal,而 cs/dw/m13h 都不允许 normal。
# cs QOS 能抢占 gstandby/standby,而且计费系数只有 0.10(gpu 是 1.0)—— fairshare 见底时优先用它。
case "${PART:-cs,dw,m13h}" in
  cs)   QOSARG="--qos=cs" ;;
  dw)   QOSARG="--qos=dw87" ;;
  m13h) QOSARG="--qos=gpu" ;;
  *)    QOSARG="" ;;                 # 多分区时让 Slurm 自己按分区选默认 QOS
esac
SB="--partition=${PART:-cs,dw,m13h} $QOSARG --exclude=dw-2-4,dw-1-3 --time=${TIME:-14:00:00} --gres=gpu:4"
SB="$SB --cpus-per-task=32 --mem=350G --job-name=$NAME --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log"
DEP=""; [ -n "${DEPEND:-}" ] && DEP="--dependency=afterok:${DEPEND}"
echo "[R1] $NAME  step 0 -> ${CHUNK_END:-80}(不 fork)"
sbatch --parsable $DEP $SB --export=ALL lucid_v2/run_locked.sl
