#!/bin/bash
# Base-student main table (2026-09-09, user: "student 全部用 base model").
# Protocol copied EXACTLY from the existing main-table runs (opd_1p7b / relay_1p7b / gatedB_1p7b):
#   teacher Qwen3-4B-Instruct-2507, max_response_length 16384 (FastOPD 8192), batch 128,
#   rollout n=1 (GRPO n=8), 139 steps = 1 epoch, actor 2 + teacher 2 GPUs.
# Only the student changes: Qwen3-1.7B-Base / Qwen3-0.6B-Base instead of the Non-Thinking models.
# Baselines are UNFIXED; the Semantic EOS class is ours (SEMANTIC_EOS=1 only on the Lucid arm).
# usage: bash submit_base_table.sh <p1|p2> [cs]
set -euo pipefail
cd /home/kzhao2/Relay-OPD
PAIR=${1:?pair: p1 (1.7B-Base) or p2 (0.6B-Base)}; PART=${2:-dw}
case "$PAIR" in
  p1) export STUDENT_MODEL=/home/kzhao2/OPD-yyx/model/Qwen3-1.7B-Base ;;
  p2) export STUDENT_MODEL=/home/kzhao2/OPD/model/Qwen3-0.6B-Base ;;
  *) echo "pair must be p1 or p2"; exit 1 ;;
esac
export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
export TRAIN_DATA=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
export BENCH=/home/kzhao2/Relay-OPD/outputs/data/bench
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=16384
# Base models cap out at max_position_embeddings=32768 (the post-trained Qwen3-1.7B has 40960),
# so the main table's VAL_MAX_RESPONSE_LENGTH=32768 -> max_model_len 34817 is rejected at startup.
# 16384 keeps max_model_len at 2048+16384+1 = 18433, enough for the 16384 training horizon.
export VAL_MAX_RESPONSE_LENGTH=16384
export SAVE_FREQ=20 TOTAL_EPOCHS=1 TEST_FREQ=1000 VAL_BEFORE_TRAIN=False
export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
export VERL_OPD_STOP_TOKEN_IDS='151643;151645'
export STOP_TOKEN_IDS='151643;151645'
unset EXTRA_ARGS 2>/dev/null || true
case "$PART" in
  # AssocGrpBillingRunMinutes budgets sum(GPUs x REQUESTED time) over RUNNING jobs, so an
  # oversized --time reserves budget we never use and blocks every other job. 139 steps at
  # 16384 takes ~13 h; 20 h leaves headroom and the checkpoints are resumable if it is not enough.
  cs)  SB="--parsable --partition=cs  --qos=cs --time=20:00:00"; SUF=_cs ;;
  cs3) SB="--parsable --partition=cs3 --qos=cs --time=20:00:00"; SUF=_cs3 ;;   # B200; env has sm_100 in arch_list
  cs2) SB="--parsable --partition=cs2 --qos=cs --time=20:00:00"; SUF=_cs2 ;;   # H100; blocked by dell_amber_light healthchecks
  # dwmatrix = dw-2-5 only, QOS standby (may be preempted; checkpoints make that a resume, not a loss)
  dwmatrix) SB="--parsable --partition=dwmatrix --qos=standby --time=20:00:00"; SUF=_dwm ;;
  # dw-2-4 looks idle but every job dies in CUDA init at ~72 s; dw-1-5 is the 7-GPU node
  *)   SB="--parsable --partition=dw --qos=dw87 --exclude=dw-1-5,dw-2-4 --time=20:00:00"; SUF="" ;;
esac
SB="$SB --gres=gpu:4 --cpus-per-task=32 --mem=350G --export=ALL"
# DEPEND=<jobid> 提交续训:verl trainer.resume_mode=auto 会读 OUTPUT_DIR/latest_checkpointed_iteration.txt
# 从最后一个 global_step 继续(checkpoint 分片是 world_size_2,所以 ACTOR_GPUS_PER_NODE 必须仍为 2)。
[ -n "${DEPEND:-}" ] && SB="$SB --dependency=afterany:${DEPEND}"
# ONLY="a b c" restricts submission to those arms (used to add duplicates for the ones still queued)
sub() { local nm=$1; shift
  if [ -n "${ONLY:-}" ] && ! grep -qw "$nm" <<< "$ONLY"; then return 0; fi
  local J=$(sbatch $SB --job-name=${nm}${SUF} "$@" run_method_freegpu.sl); echo "${nm}${SUF} $J" | tee -a .jobids/current.txt
  # chase the evals off a Slurm dependency rather than a polling watcher: it fires however the
  # training job ends and does not depend on any client session still being alive.
  local C=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=$nm,WHAT=math after_train_evals.sl 2>/dev/null)
  [ -n "$C" ] && echo "chase_${nm}${SUF} $C" >> .jobids/current.txt; }
clear_relay() { unset RELAY_OPD_TRIGGER_MODE RELAY_OPD_TRIGGER_KL_TAU RELAY_OPD_TRIGGER_MIN_TOK RELAY_OPD_KL_DIR RELAY_OPD_TEACHER_PPL_GATE RELAY_OPD_MAX_TAKEOVERS RELAY_OPD_MAX_TAKEOVER_TOKENS RELAY_OPD_PARAGRAPHS_PER_TAKEOVER RELAY_OPD_TRIGGER_TOPK 2>/dev/null || true; }

# ---- 1. OPD (sampled reverse-KL PG, h=16384) ----
export SEMANTIC_EOS=0; clear_relay
export METHOD_SCRIPT=opd/scripts/baselines/opd.sh
export OUTPUT_DIR=$PWD/outputs/checkpoints/opd_${PAIR} EXP_ID=opd_${PAIR}
sub opd_${PAIR}

# ---- 2. FastOPD (= OPD with the 8192 cap, matching fastopd8192_1p7b) ----
export METHOD_SCRIPT=opd/scripts/baselines/fastopd/8192.sh
export OUTPUT_DIR=$PWD/outputs/checkpoints/fastopd_${PAIR} EXP_ID=fastopd_${PAIR}
sub fastopd_${PAIR}

# ---- 3. GRPO (task reward only, no teacher; rollout n=8) ----
export METHOD_SCRIPT=opd/scripts/baselines/grpo.sh
export OUTPUT_DIR=$PWD/outputs/checkpoints/grpo_${PAIR} EXP_ID=grpo_${PAIR}
N_GPUS=4 sub grpo_${PAIR}

# ---- 4. RelayOPD (published: lexical "Wait," trigger, no competence gate) ----
export METHOD_SCRIPT=opd/scripts/relay_opd/train.sh
export RELAY_OPD_TRIGGER_MODE=lexical RELAY_OPD_MAX_TAKEOVERS=2 RELAY_OPD_MAX_TAKEOVER_TOKENS=256 RELAY_OPD_PARAGRAPHS_PER_TAKEOVER=3 RELAY_OPD_TRIGGER_TOPK=5
export OUTPUT_DIR=$PWD/outputs/checkpoints/relayopd_${PAIR} EXP_ID=relayopd_${PAIR}
sub relayopd_${PAIR}

# ---- 5. Lucid-OPD + Semantic EOS (ours) ----
export RELAY_OPD_TRIGGER_MODE=distributional RELAY_OPD_TRIGGER_KL_TAU=1.3 RELAY_OPD_TRIGGER_MIN_TOK=8 RELAY_OPD_KL_DIR=reverse RELAY_OPD_TEACHER_PPL_GATE=3.0
export SEMANTIC_EOS=1 SEMANTIC_EOS_IDS=151643,151645
export OUTPUT_DIR=$PWD/outputs/checkpoints/lucid_${PAIR} EXP_ID=lucid_${PAIR}
sub lucid_${PAIR}

# ---- 6. (p1 only) Lucid-OPD WITHOUT the fix -- the Semantic-EOS ablation ----
if [ "$PAIR" = p1 ]; then
  export SEMANTIC_EOS=0
  export OUTPUT_DIR=$PWD/outputs/checkpoints/lucid_${PAIR}_nofix EXP_ID=lucid_${PAIR}_nofix
  sub lucid_${PAIR}_nofix
fi
