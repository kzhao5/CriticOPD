#!/bin/bash
# 1.7B 学生的基线多 seed 实验(方案 A:新跑 seed 43、44;seed 42 是已有结果)。
#  - 全部 4xA100(actor 2 + teacher 2),与原配置只差 GPU 数和随机种子;只训到主表所选步数
#    (所有基线都是恒定学习率、无 warmup,提前停与原来训到同一步完全等价)
#  - 每段同时向 cs 和 dw 两个 A100 分区排队,先开始的取消另一个(train_segment.sl 的锁);
#    cs 单个任务上限 24 小时,长的分几段,自动从最近的 checkpoint 续跑;多交的段在训练完成后会直接退出
#  - 最后一段之后接评测守护(eval_guard.sh):评测、汇总、核对、补交
# 用法: [DRY=1] [SIZE=1.7B|0.6B] [ONLY="opd grpo"] [SEEDS="43 44"] [FIRST_DEP=afterok:<id>] bash multiseed/submit.sh
#  0.6B 不报算力成本,不要求统一硬件:用 H100(cs2)/ B200(cs3)/ H200(m13h);A100 留给 1.7B
set -euo pipefail
cd /home/kzhao2/Relay-OPD
DRY=${DRY:-0}; SEEDS=${SEEDS:-"43 44"}; ONLY=${ONLY:-}; SIZE=${SIZE:-1.7B}; FIRST_DEP=${FIRST_DEP:-}
TPQ=$PWD/outputs/offline_data/teacher_traj_pt17b/teacher_trajectories.parquet
DPQ=$PWD/outputs/offline_data/trd_pt17b/trd_trajectories.parquet
DAPO=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
REG=multiseed/jobs.tsv
PARTS=("--partition=cs --qos=cs --exclude=cs-1-2,cs-1-3" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-4,dw-1-2")   # cs-1-3、dw-1-3 在 2026-10-03 下午曾把任务在 2 秒内 root 杀掉(健康检查),傍晚已恢复
COMMON="--parsable --gres=gpu:4 --cpus-per-task=32 --mem=350G --time=${SEG_TIME:-23:30:00}"   # 运行中计费额度按「计费 x 申请时长」算,短任务用 SEG_TIME 缩短
STUDENT=/home/kzhao2/OPD/model/Qwen3-1.7B; TP=""          # TP:0.6B 的任务名前缀,避免与 1.7B 重名
if [ "$SIZE" = 0.6B ]; then
  STUDENT=/home/kzhao2/OPD/model/Qwen3-0.6B; TP=6
  PARTS=("--partition=cs2,cs --qos=cs --exclude=cs-1-2,cs-1-3" "--partition=cs3 --qos=cs" "--partition=m13h --qos=gpu --exclude=m13h-1-1" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-4,dw-1-2")   # 0.6B 不报算力,哪里空就在哪里跑
  DPQ=$PWD/outputs/offline_data/trd_pt06b/trd_trajectories.parquet      # 0.6B 学生自己的改写数据
fi

#  方法  脚本  所选步数  训练数据  基础目录名  段数  总步数怎么传(env|arg)
SPEC17=(
  "grpo     opd/scripts/baselines/grpo.sh          120 $DAPO grpo_pt17b        3 env"
  "skd      opd/scripts/baselines/skd.sh           120 $DAPO skd_pt17b         3 env"
  "seqkd    opd/scripts/baselines/seqkd.sh         139 $TPQ  seqkd_pt17b       3 env"
  "opd      opd/scripts/baselines/opd.sh            80 $DAPO opd_1p7b          2 arg"
  "trd      opd/scripts/baselines/trd.sh            40 $DPQ  trd_pt17b         2 env"
  "fastopd  opd/scripts/baselines/fastopd/8192.sh   60 $DAPO fastopd8192_1p7b  2 arg"
  "relay    opd/scripts/relay_opd/train.sh          40 $DAPO relay_1p7b        2 arg"
  "sft      opd/scripts/baselines/sft.sh            40 $TPQ  sft_pt17b         2 env"
  "critic   opd/scripts/baselines/opd.sh            40 $DAPO criticopd_final_pt17b 2 arg"
)
# CriticOPD 最终版:列出全部错误、在最后一个错误处断开(R4GTKL),批改输出上限 2048,
# 截断时丢掉最后那条没写完的错误。其余开关与 criticopd/submit_arm.sh 的 R4GTKL 逐项一致。
CRIT_ENV=",ROLLOUT_N=1,CRITIC_OPD_ENABLE=1,CRITIC_OPD_USE_FEEDBACK=1,CRITIC_OPD_SPLICE=after,CRITIC_OPD_LOSS=fb,CRITIC_OPD_GIVE_GT=1"
CRIT_ENV="$CRIT_ENV,CRITIC_OPD_TRUNC=0,CRITIC_OPD_KEEP_CORRECT=0,CRITIC_OPD_ANNEAL=,CRITIC_OPD_REF_FILE=,CRITIC_OPD_KERR=last"
CRIT_ENV="$CRIT_ENV,CRITIC_OPD_CRITIC_TOKENS=2048,AGENT_WORKERS=2,MAX_RESPONSE_LENGTH=16384"
CRIT_ARGS="actor_rollout_ref.rollout.agent.default_agent_loop=critic_opd_agent ray_kwargs.ray_init.num_cpus=32"
#  0.6B:所选步数按主表;SFT/KD 用与 1.7B 同一份 teacher 数据(两者分词器和对话模板完全相同)
SPEC06=(
  "seqkd    opd/scripts/baselines/seqkd.sh         139 $TPQ  seqkd_pt06b       3 env"
  "trd      opd/scripts/baselines/trd.sh           139 $DPQ  trd_pt06b         3 env"
  "sft      opd/scripts/baselines/sft.sh           139 $TPQ  sft_pt06b         2 env"
  "grpo     opd/scripts/baselines/grpo.sh           80 $DAPO grpo_pt06b        3 env"
  "relay    opd/scripts/relay_opd/train.sh          60 $DAPO relay_pt06b       2 arg"
  "opd      opd/scripts/baselines/opd.sh            40 $DAPO opd_pt06b         2 arg"
  "fastopd  opd/scripts/baselines/fastopd/8192.sh   40 $DAPO fastopd8192_pt06b 2 arg"
  "skd      opd/scripts/baselines/skd.sh            40 $DAPO skd_pt06b         2 env"
  "critic   opd/scripts/baselines/opd.sh            60 $DAPO criticopd_final_pt06b 2 arg"
)
#  0.6B 的 CriticOPD 没有先验峰值:训到 60,在 40 和 60 都评测(第 40 步由附加守护评测,见 eval_guard.sh 的 GUARD_NO_RESUBMIT)
if [ "$SIZE" = 0.6B ]; then SPEC=("${SPEC06[@]}"); else SPEC=("${SPEC17[@]}"); fi
sb() { if [ "$DRY" = 1 ]; then echo "DRY sbatch $*" >&2; echo "9999$RANDOM"; else sbatch "$@"; fi; }

for spec in "${SPEC[@]}"; do
  read -r m script S data base nseg how <<< "$spec"
  nseg=$((nseg + ${NSEG_ADD:-0}))      # 节点会在作业启动 2 秒内把它杀掉,每被杀一次就耗掉一段,多备几段
  [ -n "$ONLY" ] && ! grep -qw "$m" <<< "$ONLY" && continue
  for seed in $SEEDS; do
    run=${base}_seed$seed; tag=$TP$m$seed; out=$PWD/outputs/checkpoints/$run
    extra="trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=2"
    stepv=""
    if [ "$how" = arg ]; then extra="$extra trainer.total_training_steps=$S"; else stepv=",TOTAL_TRAINING_STEPS=$S"; fi
    [ "$m" = sft ] && extra=""                       # SFT 训练器不接受额外参数
    envs="ALL,METHOD_SCRIPT=$script,TRAIN_DATA=$data,STUDENT_MODEL=$STUDENT"
    envs="$envs,OUTPUT_DIR=$out,EXP_ID=$run,SAVE_FREQ=20,TEST_FREQ=-1,VAL_BEFORE_TRAIN=False"
    envs="$envs,ACTOR_GPUS_PER_NODE=2,TEACHER_GPUS_PER_NODE=2,NEED_GPUS=4,SEED=$seed,SFT_DATA_SEED=$seed,TARGET_STEP=$S$stepv"
    [ "$m" = sft ] && envs="$envs,NUM_GPUS=4"
    [ "$m" = grpo ] && envs="$envs,N_GPUS=4"          # grpo.sh 读 N_GPUS(默认 8)
    [ "$m" = critic ] && { envs="$envs$CRIT_ENV"; extra="$extra $CRIT_ARGS"; }
    prev=""
    for seg in $(seq 1 $nseg); do
      dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
      [ -z "$prev" ] && [ -n "$FIRST_DEP" ] && dep="--dependency=$FIRST_DEP"
      ids=""
      for p in "${PARTS[@]}"; do
        pe=""; [[ $p == *cs3* ]] && pe=",EXTRA_MODULES=cuda/12.8.1"        # B200 节点要另载 CUDA 12.8
        j=$(sb $p $COMMON $dep --job-name=ms_${tag}_s$seg --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log \
              --export="$envs$pe,SEG=$seg,EXTRA_ARGS=$extra" paper_baselines/train_segment.sl)
        ids="$ids:$j"
      done
      prev=${ids#:}
      [ "$DRY" = 1 ] || printf "%s\t%s\t%s\ttrain_seg%s\t%s\n" "$(date '+%F %T')" "$run" "$S" "$seg" "$prev" >> $REG
    done
    g=$(sb -p cs --qos=cs -c 2 --mem=8G --time=0:30:00 --parsable --job-name=g_$tag --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log \
          --dependency=afterany:$prev --wrap="bash /home/kzhao2/Relay-OPD/multiseed/eval_guard.sh $run $S $tag 0 0")
    [ "$DRY" = 1 ] || printf "%s\t%s\t%s\tguard\t%s\n" "$(date '+%F %T')" "$run" "$S" "$g" >> $REG
    echo "$run: 训练 $nseg 段(最后一段 $prev)-> 评测守护 $g"
  done
done
