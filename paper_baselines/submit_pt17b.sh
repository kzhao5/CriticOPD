#!/bin/bash
# Paper baselines for the POST-TRAINED 1.7B student (the original Relay-OPD setting).
#
# The post-trained column of the table has only OPD, FastOPD@8192, RelayOPD and gatedB; GRPO's checkpoints
# were lost in the 2026-09-17 storage cleanup and the other seven baselines were never trained here.
#
# Protocol = the existing runs of THIS setting (relay_1p7b / opd_1p7b / fastopd8192_1p7b / gatedB_1p7b,
# see lucid_pt_setting/submit.sh): student Qwen3-1.7B post-trained, teacher Qwen3-4B-Instruct-2507,
# train response cap 16384, engine context 32768, batch 128, 1 epoch = 139 steps, save every 20 steps,
# validation off. NOT the base-table b16k protocol -- these runs sit next to the PT runs above.
#
#   DRY=1 bash paper_baselines/submit_pt17b.sh                 # print the sbatch lines
#   ONLY="fastopd1024 sft" bash paper_baselines/submit_pt17b.sh
#   PARTS="dw cs cs3" ONLY=fastopd1024 bash ...                # also queue a B200 copy
#   ONLY=skd SEG_FROM=3 NSEG_TO=5 FIRST_DEP=afterany:<ids> bash ...    # extend a run
set -euo pipefail
cd /home/kzhao2/Relay-OPD
DRY=${DRY:-0}
REG=.jobids/paper_baselines.txt
DATA=$PWD/outputs/offline_data
STUDENT=/home/kzhao2/OPD/model/Qwen3-1.7B          # post-trained, non-thinking template

export STUDENT_MODEL=$STUDENT
export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
export BENCH=/home/kzhao2/Relay-OPD/outputs/data/bench
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=16384 VAL_MAX_RESPONSE_LENGTH=32768
export SAVE_FREQ=${SAVE_FREQ_OVERRIDE:-20} TOTAL_EPOCHS=1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False
export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
export VERL_OPD_STOP_TOKEN_IDS='151643;151645' STOP_TOKEN_IDS='151643;151645' SEMANTIC_EOS=0
unset EXTRA_ARGS RELAY_OPD_TRIGGER_MODE RELAY_OPD_TRIGGER_KL_TAU RELAY_OPD_KL_DIR \
      RELAY_OPD_TEACHER_PPL_GATE RELAY_OPD_MAX_TAKEOVERS 2>/dev/null || true

DW="--partition=dw --qos=dw87 --exclude=dw-1-5,dw-2-4"
CS="--partition=cs,cs2 --qos=cs"
CS3="--partition=cs3 --qos=cs"     # B200: flashinfer JIT needs nvcc, so EXTRA_MODULES is set below
COMMON="--parsable --gres=gpu:4 --cpus-per-task=32 --mem=350G"
PARTS=${PARTS:-"dw cs"}

TEACHER_PQ=$DATA/teacher_traj_pt17b/teacher_trajectories.parquet
TRD_PQ=$DATA/trd_pt17b/trd_trajectories.parquet
# Data jobs already queued on the B200 node (pb_data_teacher_pt17b / pb_data_trd_pt17b); pass their ids
# as TEACHER_DATA_JOB / TRD_DATA_JOB so the runs that need them wait instead of failing on a missing file.
TEACHER_DEP=${TEACHER_DATA_JOB:+afterok:$TEACHER_DATA_JOB}
TRD_DEP=${TRD_DATA_JOB:+afterok:$TRD_DATA_JOB}

note() { if [ "$DRY" = 1 ]; then echo "$1 $2"; else echo "$1 $2" | tee -a "$REG"; fi; }
sb() { local nm=$1; shift; if [ "$DRY" = 1 ]; then echo "DRY sbatch --job-name=$nm $*" >&2; echo "DRY_$nm"; return; fi; sbatch --job-name="$nm" "$@"; }
want() { [ -z "${ONLY:-}" ] || grep -qw "$1" <<< "$ONLY"; }
mkdir -p .jobids

run() {  # run <method> <script> <hours> <segments> <train parquet> [data dependency]
  local m=$1 script=$2 hours=$3 nseg=$4 tdata=$5 dep0=${6:-}
  want "$m" || return 0
  local name=${m}_pt17b
  export STUDENT_MODEL=$STUDENT METHOD_SCRIPT=$script TRAIN_DATA=$tdata
  if [[ $script == *sft.sh ]]; then export NUM_GPUS=4; else unset NUM_GPUS; fi
  export OUTPUT_DIR=$PWD/outputs/checkpoints/$name EXP_ID=$name
  local dep=$dep0 prev="" seg ids j where flags
  [ -n "${NSEG_TO:-}" ] && nseg=$NSEG_TO
  [ -n "${FIRST_DEP:-}" ] && dep=$FIRST_DEP
  for seg in $(seq "${SEG_FROM:-1}" "$nseg"); do
    local d=$dep; [ -n "$prev" ] && d="afterany:$prev"
    local dflag=""; [ -n "$d" ] && dflag="--dependency=$d"
    ids=""
    for where in $PARTS; do
      case $where in
        dw)  flags="$DW";  extra="" ;;
        cs)  flags="$CS";  extra="" ;;
        cs3) flags="$CS3"; extra=",EXTRA_MODULES=cuda/12.8.1" ;;
        *) echo "unknown partition group '$where'"; exit 1 ;;
      esac
      j=$(sb pb_${name}_s$seg $flags $COMMON --time=${hours}:00:00 $dflag \
            --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log \
            --export=ALL,SEG=$seg,TARGET_STEP=139${extra} paper_baselines/train_segment.sl)
      note "pb_${name}_s$seg" "$j"; ids="$ids:$j"
    done
    prev=${ids#:}
  done
  j=$(sb chase_pb_$name --parsable --partition=m8 --qos=normal --ntasks=1 --cpus-per-task=2 --mem=4G \
        --time=2:00:00 --dependency=afterany:$prev \
        --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log \
        --export=ALL,RUN=$name paper_baselines/chase_eval.sl)
  note "chase_pb_$name" "$j"
}

DAPO=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
# cheapest first, so the column fills in even if the budget only allows a few runs at a time
run fastopd1024 opd/scripts/baselines/fastopd/1024.sh  8 2 "$DAPO"
run fastopd2048 opd/scripts/baselines/fastopd/2048.sh 10 2 "$DAPO"
run fastopd4096 opd/scripts/baselines/fastopd/4096.sh 16 2 "$DAPO"
run sft         opd/scripts/baselines/sft.sh          24 1 "$TEACHER_PQ" "$TEACHER_DEP"
run trd         opd/scripts/baselines/trd.sh          20 2 "$TRD_PQ"     "$TRD_DEP"
run seqkd       opd/scripts/baselines/seqkd.sh        20 2 "$TEACHER_PQ" "$TEACHER_DEP"
run skd         opd/scripts/baselines/skd.sh          20 4 "$DAPO"
# GRPO: the post-trained run's checkpoints were deleted, so it has to be retrained from scratch.
run grpo        opd/scripts/baselines/grpo.sh         20 2 "$DAPO"
