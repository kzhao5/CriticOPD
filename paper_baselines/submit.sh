#!/bin/bash
# Relay-OPD paper baselines that the base-student table does not have yet, for both students.
#   paper baselines (relay-opd/README.md): SFT, SeqKD, GRPO, OPD, FastOPD@{1024,2048,4096,8192}, TRD, SKD, Relay-OPD
#   already in the table:                  OPD, FastOPD@8192, GRPO, RelayOPD      -> not touched, not rerun
#   submitted here:                        SFT, SeqKD, TRD, SKD, FastOPD@1024/2048/4096
# Protocol = submit_base_table.sh (teacher Qwen3-4B-Instruct-2507, DAPO-Math-17K, batch 128, 1 epoch = 139 steps,
# max response 16384 unless the method fixes its own, actor 2 + teacher 2 GPUs, save every 20 steps, rollout
# and eval stop tokens 151643/151645). Method settings are the paper scripts' defaults, unchanged.
#
#   bash paper_baselines/submit.sh            # submit everything
#   DRY=1 bash paper_baselines/submit.sh      # print the sbatch lines only
#   ONLY="skd_p1 fastopd1024_p2" bash ...     # restrict to some runs (data jobs are submitted when needed)
#   ONLY=skd_p2 SEG_FROM=3 NSEG_TO=5 FIRST_DEP=afterany:<s2 ids> bash ...   # add segments 3-5 (cancel the old chaser first)
set -euo pipefail
cd /home/kzhao2/Relay-OPD
DRY=${DRY:-0}
REG=.jobids/paper_baselines.txt
DATA=$PWD/outputs/offline_data
P1=/home/kzhao2/OPD-yyx/model/Qwen3-1.7B-Base
P2=/home/kzhao2/OPD/model/Qwen3-0.6B-Base

# ---- shared environment, copied from submit_base_table.sh ----
export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
export BENCH=/home/kzhao2/Relay-OPD/outputs/data/bench
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=16384
export VAL_MAX_RESPONSE_LENGTH=16384
export SAVE_FREQ=${SAVE_FREQ_OVERRIDE:-20} TOTAL_EPOCHS=1 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False
export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
export VERL_OPD_STOP_TOKEN_IDS='151643;151645'
export STOP_TOKEN_IDS='151643;151645'
export SEMANTIC_EOS=0
unset EXTRA_ARGS RELAY_OPD_TRIGGER_MODE RELAY_OPD_TRIGGER_KL_TAU RELAY_OPD_TRIGGER_MIN_TOK RELAY_OPD_KL_DIR \
      RELAY_OPD_TEACHER_PPL_GATE RELAY_OPD_MAX_TAKEOVERS RELAY_OPD_MAX_TAKEOVER_TOKENS \
      RELAY_OPD_PARAGRAPHS_PER_TAKEOVER RELAY_OPD_TRIGGER_TOPK 2>/dev/null || true

# dw-2-4 dies in CUDA init, dw-1-5 is the 7-GPU node (see memory / submit_base_table.sh)
DW="--partition=dw --qos=dw87 --exclude=dw-1-5,dw-2-4"
# cs3 (B200) is left out: training there has not been validated with this launcher (flashinfer JIT needs cuda 12.8)
CS="--partition=cs,cs2 --qos=cs"
COMMON="--parsable --gres=gpu:4 --cpus-per-task=32 --mem=350G"

note() { if [ "$DRY" = 1 ]; then echo "$1 $2"; else echo "$1 $2" | tee -a "$REG"; fi; }
sb() {  # sb <name> <sbatch args...>  -> job id (or DRY placeholder)
  local nm=$1; shift
  if [ "$DRY" = 1 ]; then echo "DRY sbatch --job-name=$nm $*" >&2; echo "DRY_$nm"; return; fi
  sbatch --job-name="$nm" "$@"
}
want() { [ -z "${ONLY:-}" ] || grep -qw "$1" <<< "$ONLY"; }
mkdir -p .jobids "$DATA"

# ---- offline data ----
TEACHER_PQ=$DATA/teacher_traj_base/teacher_trajectories.parquet   # shared by p1/p2: identical tokenizer + template
declare -A TRD_PQ=([p1]=$DATA/trd_p1/trd_trajectories.parquet [p2]=$DATA/trd_p2/trd_trajectories.parquet)
declare -A STUDENT=([p1]=$P1 [p2]=$P2)
declare -A DATA_JOB=()

need_teacher=0; need_trd_p1=0; need_trd_p2=0
for p in p1 p2; do
  { want sft_$p || want seqkd_$p; } && need_teacher=1
  want trd_$p && eval "need_trd_$p=1"
done
if [ "$need_teacher" = 1 ] && [ ! -s "$TEACHER_PQ" ]; then
  j=$(sb pb_data_teacher $DW $COMMON --time=16:00:00 --export=ALL,MODE=teacher,STUDENT_MODEL=$P1,OUT_PARQUET=$TEACHER_PQ paper_baselines/gen_data.sl)
  note pb_data_teacher "$j"; DATA_JOB[teacher]=$j
fi
for p in p1 p2; do
  v=need_trd_$p
  if [ "${!v}" = 1 ] && [ ! -s "${TRD_PQ[$p]}" ]; then
    j=$(sb pb_data_trd_$p $DW $COMMON --time=20:00:00 --export=ALL,MODE=trd,STUDENT_MODEL=${STUDENT[$p]},OUT_PARQUET=${TRD_PQ[$p]} paper_baselines/gen_data.sl)
    note pb_data_trd_$p "$j"; DATA_JOB[trd_$p]=$j
  fi
done

# ---- training: run <name> <pair> <method script> <hours> <segments> <train parquet> [data job key] ----
run() {
  local name=$1 pair=$2 script=$3 hours=$4 nseg=$5 tdata=$6 dkey=${7:-}
  want "$name" || return 0
  export STUDENT_MODEL=${STUDENT[$pair]} METHOD_SCRIPT=$script TRAIN_DATA=$tdata
  if [[ $script == *sft.sh ]]; then export NUM_GPUS=4; else unset NUM_GPUS; fi
  export OUTPUT_DIR=$PWD/outputs/checkpoints/$name EXP_ID=$name
  local dep="" prev="" seg ids j
  if [ -n "$dkey" ] && [ -n "${DATA_JOB[$dkey]:-}" ]; then dep="afterok:${DATA_JOB[$dkey]}"; fi
  # extending a run: SEG_FROM=<first new segment> NSEG_TO=<last segment> FIRST_DEP=afterany:<ids of the current last segment>
  [ -n "${NSEG_TO:-}" ] && nseg=$NSEG_TO
  [ -n "${FIRST_DEP:-}" ] && dep=$FIRST_DEP
  for seg in $(seq "${SEG_FROM:-1}" "$nseg"); do
    local d=$dep; [ -n "$prev" ] && d="afterany:$prev"
    local dflag=""; [ -n "$d" ] && dflag="--dependency=$d"
    ids=""
    for where in "$DW" "$CS"; do
      j=$(sb pb_${name}_s$seg $where $COMMON --time=${hours}:00:00 $dflag \
            --export=ALL,SEG=$seg,TARGET_STEP=139 paper_baselines/train_segment.sl)
      note "pb_${name}_s$seg" "$j"; ids="$ids:$j"
    done
    prev=${ids#:}
  done
  j=$(sb chase_pb_$name --parsable --dependency=afterany:$prev --export=ALL,RUN=$name paper_baselines/chase_eval.sl)
  note "chase_pb_$name" "$j"
}

DAPO=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
for p in p1 p2; do
  run fastopd1024_$p $p opd/scripts/baselines/fastopd/1024.sh 8 2 $DAPO
  run fastopd2048_$p $p opd/scripts/baselines/fastopd/2048.sh 10 2 $DAPO
  run fastopd4096_$p $p opd/scripts/baselines/fastopd/4096.sh 16 2 $DAPO
  run skd_$p $p opd/scripts/baselines/skd.sh 20 2 $DAPO
  run seqkd_$p $p opd/scripts/baselines/seqkd.sh 20 2 $TEACHER_PQ teacher
  run trd_$p $p opd/scripts/baselines/trd.sh 20 2 ${TRD_PQ[$p]} trd_$p
  # sft_trainer runs with resume_mode=disable: a second segment would restart from step 0 and overwrite, so one 24 h segment
  run sft_$p $p opd/scripts/baselines/sft.sh 24 1 $TEACHER_PQ teacher
done
