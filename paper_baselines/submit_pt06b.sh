#!/bin/bash
# Baselines for the POST-TRAINED 0.6B student (Qwen/Qwen3-0.6B, non-thinking template).
# Same regime as the post-trained 1.7B runs: train response 16384, engine context 32768, batch 128,
# 1 epoch = 139 steps, save every 20, validation off, 2 actor + 2 teacher GPUs.
# Qwen3-0.6B has max_position_embeddings 40960, so the 32x32768 evaluation protocol applies.
#   DRY=1 bash paper_baselines/submit_pt06b.sh
#   ONLY="relay opd" bash paper_baselines/submit_pt06b.sh
set -euo pipefail
cd /home/kzhao2/Relay-OPD
DRY=${DRY:-0}; REG=.jobids/pt06b.txt; mkdir -p .jobids
export STUDENT_MODEL=/home/kzhao2/OPD/model/Qwen3-0.6B
export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
export TRAIN_DATA=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
export BENCH=/home/kzhao2/Relay-OPD/outputs/data/bench
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=16384 VAL_MAX_RESPONSE_LENGTH=32768
export SAVE_FREQ=${SAVE_FREQ_OVERRIDE:-20} TOTAL_EPOCHS=1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False
export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
export VERL_OPD_STOP_TOKEN_IDS='151643;151645' STOP_TOKEN_IDS='151643;151645'
export VERL_OPD_DIR=/home/kzhao2/Relay-OPD/relay-opd
export PYTHONPATH=$VERL_OPD_DIR:${PYTHONPATH:-}
TIME=${TIME:-14:00:00}
DW="--partition=dw --qos=dw87 --exclude=dw-1-5,dw-2-4"
CS="--partition=cs,cs2 --qos=cs"
clear_relay(){ unset RELAY_OPD_TRIGGER_MODE RELAY_OPD_TRIGGER_KL_TAU RELAY_OPD_TRIGGER_MIN_TOK \
  RELAY_OPD_KL_DIR RELAY_OPD_TEACHER_PPL_GATE RELAY_OPD_MAX_TAKEOVERS RELAY_OPD_MAX_TAKEOVER_TOKENS \
  RELAY_OPD_PARAGRAPHS_PER_TAKEOVER RELAY_OPD_TRIGGER_TOPK 2>/dev/null || true; }
want(){ [ -z "${ONLY:-}" ] || grep -qw "$1" <<< "$ONLY"; }
sub(){ local nm=$1 where=$2
  export OUTPUT_DIR=$PWD/outputs/checkpoints/$nm EXP_ID=$nm
  mkdir -p "$OUTPUT_DIR"
  if [ "$DRY" = 1 ]; then echo "DRY $nm ($where) METHOD_SCRIPT=$METHOD_SCRIPT SEMANTIC_EOS=${SEMANTIC_EOS:-unset}"; return; fi
  local J=$(sbatch --parsable $where --time=$TIME --gres=gpu:4 --cpus-per-task=32 --mem=350G \
        --export=ALL --job-name=$nm --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log run_method_freegpu.sl)
  echo "$nm $J ($where)" | tee -a "$REG"
  # chase the math evaluation off a Slurm dependency so it fires however training ends
  # NOTE: after_train_evals.sl uses the b16k protocol (8x16384). The post-trained table uses
  # 32x32768, so evaluation is chased separately by lucid_pt_setting/chase_eval_pt.sl.
  local C=""
  [ -n "$C" ] && echo "chase_$nm $C" >> "$REG"; }

# ---- OPD (sampled reverse-KL PG, 16384) -- the Delta-vs-OPD reference row ----
if want opd; then export SEMANTIC_EOS=0; clear_relay
  export METHOD_SCRIPT=opd/scripts/baselines/opd.sh; sub opd_pt06b "$DW"; fi
# ---- FastOPD (= OPD with the 8192 cap) ----
if want fastopd; then export SEMANTIC_EOS=0; clear_relay
  export METHOD_SCRIPT=opd/scripts/baselines/fastopd/8192.sh; sub fastopd8192_pt06b "$CS"; fi
# ---- GRPO (task reward only, no teacher; rollout n=8, all 4 GPUs are actor) ----
if want grpo; then export SEMANTIC_EOS=0; clear_relay
  export METHOD_SCRIPT=opd/scripts/baselines/grpo.sh N_GPUS=4
  sub grpo_pt06b "$CS"; unset N_GPUS; fi
# ---- SKD (on-policy student-teacher mixture) ----
if want skd; then export SEMANTIC_EOS=0; clear_relay
  export METHOD_SCRIPT=opd/scripts/baselines/skd.sh; sub skd_pt06b "$DW"; fi
# ---- RelayOPD (published: lexical reflection-token trigger, no competence gate) ----
if want relay; then export SEMANTIC_EOS=0; clear_relay
  export METHOD_SCRIPT=opd/scripts/relay_opd/train.sh
  export RELAY_OPD_TRIGGER_MODE=lexical RELAY_OPD_MAX_TAKEOVERS=2 \
         RELAY_OPD_MAX_TAKEOVER_TOKENS=256 RELAY_OPD_PARAGRAPHS_PER_TAKEOVER=3 RELAY_OPD_TRIGGER_TOPK=5
  sub relay_pt06b "$DW"; fi
