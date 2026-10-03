#!/bin/bash
# Throwaway smoke tests for the four new code paths (SFT, SeqKD, TRD, SKD) on tiny data, so a broken path shows up
# in about an hour instead of after the multi-hour data synthesis. standby is acceptable here (smoke only).
# 16 prompts, 1024-token responses, batch 8, 2 steps, same launcher and env as paper_baselines/submit.sh.
#   bash paper_baselines/smoke.sh
set -euo pipefail
cd /home/kzhao2/Relay-OPD
TS=$(date +%m%d%H%M)
REG=.jobids/paper_baselines_smoke.txt
SM=$PWD/outputs/offline_data/smoke_$TS
P1=/home/kzhao2/OPD-yyx/model/Qwen3-1.7B-Base
Q="--partition=dw,cs,cs2 --qos=standby --exclude=dw-1-5,dw-2-4 --parsable"
note() { echo "$1 $2" | tee -a "$REG"; }
echo "# $(date '+%F %T') smoke $TS" >> "$REG"

export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
export BENCH=/home/kzhao2/Relay-OPD/outputs/data/bench
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=1024 VAL_MAX_RESPONSE_LENGTH=1024
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 TOTAL_TRAINING_STEPS=2
export SAVE_FREQ=2 TOTAL_EPOCHS=1 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False
export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
export VERL_OPD_STOP_TOKEN_IDS='151643;151645' STOP_TOKEN_IDS='151643;151645' SEMANTIC_EOS=0
export STUDENT_MODEL=$P1
unset EXTRA_ARGS NUM_GPUS 2>/dev/null || true

dt=$(NEED_GPUS=1 sbatch $Q --job-name=smk_pb_data_teacher --gres=gpu:1 --cpus-per-task=8 --mem=100G --time=2:00:00 \
     --export=ALL,MODE=teacher,SMOKE_ROWS=16,MAX_NEW=1024,NEED_GPUS=1,STUDENT_MODEL=$P1,OUT_PARQUET=$SM/teacher/teacher_trajectories.parquet paper_baselines/gen_data.sl)
note smk_pb_data_teacher "$dt"
dr=$(sbatch $Q --job-name=smk_pb_data_trd --gres=gpu:1 --cpus-per-task=8 --mem=100G --time=2:00:00 \
     --export=ALL,MODE=trd,SMOKE_ROWS=16,MAX_NEW=1024,NEED_GPUS=1,STUDENT_MODEL=$P1,OUT_PARQUET=$SM/trd/trd_trajectories.parquet paper_baselines/gen_data.sl)
note smk_pb_data_trd "$dr"

train() {  # train <method> <script> <train parquet> [dependency]
  local m=$1 script=$2 data=$3 dep=${4:-} extra=""
  [ -n "$dep" ] && extra="--dependency=afterok:$dep"
  local nm=smk_pb_${m}
  local j
  if [ "$m" = sft ]; then export NUM_GPUS=4; else unset NUM_GPUS; fi
  j=$(METHOD_SCRIPT=$script TRAIN_DATA=$data OUTPUT_DIR=$PWD/outputs/checkpoints/smoke_pb_${m}_$TS EXP_ID=smoke_pb_${m}_$TS \
      sbatch $Q $extra --job-name=$nm --gres=gpu:4 --cpus-per-task=32 --mem=350G --time=1:30:00 --export=ALL run_method_freegpu.sl)
  note "$nm" "$j"
}
train skd opd/scripts/baselines/skd.sh /home/kzhao2/OPD/datasets/dapo-math-17k.parquet
train sft opd/scripts/baselines/sft.sh $SM/teacher/teacher_trajectories.parquet "$dt"
train seqkd opd/scripts/baselines/seqkd.sh $SM/teacher/teacher_trajectories.parquet "$dt"
train trd opd/scripts/baselines/trd.sh $SM/trd/trd_trajectories.parquet "$dr"
