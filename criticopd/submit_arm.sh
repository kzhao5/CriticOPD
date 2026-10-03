#!/bin/bash
# CriticOPD 五个 arm 的统一提交入口。
#   ARM=R0  plain OPD —— 不必重跑,就是主表里的 opd_1p7b(同起点、同 16384 上限)
#   ARM=R1  plain OPD + 多采样(算力对齐 R3)
# spec 的主线 R2/R3/R4 全用 L_center/L_fb;L_clean 只是"预计信号弱"的对照,后缀 c 区分。
#   ARM=R2   修复但不给反馈,L_center(与 R3 成对,回答"反馈本身值不值")
#   ARM=R3   修复 + 反馈,L_center      ← 主方法 1
#   ARM=R4   修复 + 反馈,L_fb          ← 未实现(需 teacher 带反馈再前向),闸门会挡住
#   ARM=R2c / R3c  同上但用 L_clean    ← 弱损失对照 / 管线验证
#
# 所有 arm:从后训练学生(step 0)起训,rollout 上限 16384,2 actor + 2 teacher —— 与主表
# baseline 逐项对齐,所以 R1..R4 既互相可比,也能直接和主表对比。
# NODELIST=<node> 可以把 job 钉在指定节点上(抢占刚释放出来的空卡)。
set -euo pipefail
cd /home/kzhao2/Relay-OPD
ARM=${ARM:?ARM=R0|R1|R2|R3|R4}
export RAY_TMPDIR_BASE=/tmp/ray_${USER}
export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-2} TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-2}
# 起点和主表所有 baseline 完全一致:后训练的 Qwen3-1.7B(step 0),rollout 上限 16384。
# 这样 R0(plain OPD)就等于现成的 opd_1p7b,不必重跑,而且 R1..R4 与整张主表直接可比。
# 之前用 opd@40 当起点是错的:既破坏了与主表的可比性,又逼得 R0 要重跑。
export STUDENT_MODEL=${STUDENT_MODEL:-/home/kzhao2/OPD/model/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}   # 与主表 baseline 一致
unset FORK_FROM FORK_STEP 2>/dev/null || true       # 从 step 0 起训,没有可 fork 的东西
export METHOD_SCRIPT=opd/scripts/baselines/opd.sh
case "$ARM" in
  R0) export ROLLOUT_N=1 ;;
  R1) export ROLLOUT_N=${ROLLOUT_N:-2} ;;
  R2)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=0 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=center CRITIC_OPD_KERR= ;;
  R3)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=center CRITIC_OPD_KERR= ;;
  R4)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=0 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KERR= ;;
  # 以 R4(当前最好)为基底,各只改一处,互不影响;两者都有提升再合并成 R4GTTR。
  #   R4GT:critic 拿到正确答案(并用重写过的 prompt),写不完的轨迹也交给 teacher 判断
  #   R4TR:学生改完出错的那一步就停 —— 重写截在第一个空行,预算 512
  R4GT)   export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR= ;;
  R4TR)   export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=0 CRITIC_OPD_TRUNC=1 CRITIC_OPD_TRUNC_MAX=512 CRITIC_OPD_KERR= ;;
  # R4GTV:R4GT + 只保留修对的重写(重写后用 gt 再判,没修对退回原始 rollout 按 plain OPD 训练)
  R4GTV)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=1 CRITIC_OPD_KERR= ;;
  # R4GTA:R4GT + DAgger 式退火,训 60 步:1-20 全用修复,20-40 线性降到 0,40-60 纯 plain OPD
  R4GTA)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_ANNEAL=20,40,0 CRITIC_OPD_KERR= ;;
  # R4GTR:R4GT + 参考解答版批改(critic 额外拿到 teacher 离线做对、用标准答案核对过的一份解答)
  R4GTR)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE=/home/kzhao2/Relay-OPD/criticopd/refs_train40.json CRITIC_OPD_KERR= ;;
  # R4GTR2:同 R4GTR,但参考解答覆盖训练真正用到的题(训练用 8 个进程加载数据,顺序与单进程不同)
  R4GTR2) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE=/home/kzhao2/Relay-OPD/criticopd/refs_train40_v2.json CRITIC_OPD_KERR= ;;
  # R4GTB:消融"断在哪里"——与 R4GT 只差 SPLICE:不保留出错段,学生从出错段之前开始重写
  R4GTB)  export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=before CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR= ;;
  # k 消融:与 R4GT 只差批改方式——批改老师按顺序列出所有错误,在第 k 个错误处断开(保留出错段)
  R4GTK1) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR=1 ;;
  R4GTK2) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR=2 ;;
  R4GTK3) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR=3 ;;
  R4GTKL) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE= CRITIC_OPD_KERR=last ;;
  R4GTTR) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1 CRITIC_OPD_TRUNC=1 CRITIC_OPD_TRUNC_MAX=512 CRITIC_OPD_KERR= ;;
  R2c) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=0 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=clean CRITIC_OPD_KERR= ;;
  R3c) export ROLLOUT_N=1 CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=clean CRITIC_OPD_KERR= ;;
  *) echo "未知 ARM=$ARM"; exit 1 ;;
esac
# L_center / L_fb 还没写进 loss;误投会静默退化成 L_clean 而标签写着 center/fb,所以挡住。
case "${CRITIC_OPD_LOSS:-}" in
  center) grep -rqs "critic_repair_mask" relay-opd/verl/trainer/distillation/losses.py 2>/dev/null || {
      echo "!!! L_center 未实现(losses.py 里没有 critic_repair_mask)"; exit 1; } ;;
  # L_fb 不走 losses.py:实现方式是在 agent loop 里用"带反馈上下文"重算的 teacher 分布
  # 覆盖修复 token 那几行,损失侧无需改动。所以闸门要检查真正的实现标记。
  fb)     grep -rqs "critic_fb_prompt_ids" relay-opd/verl/experimental/agent_loop/agent_loop.py 2>/dev/null || {
      echo "!!! L_fb 未实现:agent_loop.py 里没有 critic_fb_prompt_ids 的重打分逻辑。"
      echo "    先跑 R3(L_center)。"; exit 1; } ;;
esac
# R2/R3* 走自定义 agent loop;R0/R1 走默认
if [ "${CRITIC_OPD_ENABLE:-0}" = 1 ]; then
  export EXTRA_ARGS_AGENT="actor_rollout_ref.rollout.agent.default_agent_loop=critic_opd_agent"
fi
NAME=${NAME_OVERRIDE:-criticopd_${ARM}_pt17b}
export OUTPUT_DIR=$PWD/outputs/checkpoints/$NAME EXP_ID=$NAME
END=${CHUNK_END:-80}                        # 训到 80,对上 opd_1p7b@80 这个峰值
# Ray 在 SLURM 上必须显式给 num_cpus:默认 null 时它按 /proc 看到的整机核数(m13h 是 96)
# 去铺 worker,而 cgroup 只给我 ${CPUS} 核,raylet 30 秒内注册不上 GCS,直接抛
# "node timed out during startup"。verl 自己的配置注释也写着 "use a fixed number when using SLURM"。
# 13936795/13936796/13936805 三次全挂在这上面;唯一成功的一次跑在空闲的 cs-1-5。
# agent.num_workers 默认 8:每个 AgentLoopWorker 都各自加载一套 tokenizer/processor。
# m13h 这些节点 Slurm 账面还有余量但物理内存只剩 ~140G(别人申请了没用满),8 个 worker
# 一起起来会被内核 OOM 掉(job 13937168 的 agent_loop_worker_4 就是这么死的)。
export EXTRA_ARGS="trainer.total_training_steps=$END ray_kwargs.ray_init.num_cpus=${CPUS:-32} ${EXTRA_ARGS_AGENT:-}"
# num_workers 由 opd.sh 统一设(它还会从 criticopd/.agent_workers 读,以便影响已排队的 job)。
# 这里只导出环境变量 —— 两处都往 hydra 传同名覆盖会直接报重复。
export AGENT_WORKERS=${AGENT_WORKERS:-2}
# 训练中的 validation 只影响墙钟、不影响权重(主表数字来自独立的 eval job),关掉省时间。
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} TEST_FREQ=${TEST_FREQ:-100000}
export RAY_RAYLET_START_WAIT_S=${RAY_RAYLET_START_WAIT_S:-600}   # 补丁被拒,留着以后能用
export SAVE_FREQ=${SAVE_FREQ:-20}
mkdir -p "$OUTPUT_DIR"
# QOS 必须显式给:指定单分区而不给 QOS 时默认变成 normal,而 cs/dw/m13h 都不允许 normal。
# cs QOS 能抢占 gstandby/standby,而且计费系数只有 0.10(gpu 是 1.0)—— fairshare 见底时优先用它。
case "${PART:-cs,dw,m13h}" in
  cs|cs2|cs3|cs,cs2|cs,cs2,cs3|cs2,cs3) QOSARG="--qos=cs" ;;
  dw)   QOSARG="--qos=dw87" ;;
  m13h) QOSARG="--qos=gpu" ;;
  *)    QOSARG="" ;;                 # 多分区时让 Slurm 自己按分区选默认 QOS
esac
NGPU=$(( ACTOR_GPUS_PER_NODE + TEACHER_GPUS_PER_NODE )); export NEED_GPUS=$NGPU
SB="--partition=${PART:-cs,dw,m13h} $QOSARG --exclude=${EXCLUDE:-dw-2-4,dw-1-3} --time=${TIME:-20:00:00} --gres=gpu:$NGPU --nodes=1"
[ -n "${NODELIST:-}" ] && SB="$SB --nodelist=${NODELIST}"
SB="$SB --cpus-per-task=${CPUS:-32} --mem=${MEM:-350G} --job-name=$NAME --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log"
# 链式接续要用 afterany:前一段是 TIMEOUT 退出的,afterok 会让后续 job 永远不启动。
DEP=""; [ -n "${DEPEND:-}" ] && DEP="--dependency=${DEPEND_TYPE:-afterany}:${DEPEND}"
echo "[$ARM] $NAME  起点=后训练学生(step 0) -> $END 步  rollout_n=${ROLLOUT_N}  cap=${MAX_RESPONSE_LENGTH}  gpu=$NGPU@${NODELIST:-any}"
sbatch --parsable $DEP $SB --export=ALL lucid_v2/run_locked.sh 2>/dev/null || \
sbatch --parsable $DEP $SB --export=ALL lucid_v2/run_locked.sl
