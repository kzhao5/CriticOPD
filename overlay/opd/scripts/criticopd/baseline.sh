#!/usr/bin/env bash
# Baselines of the paper, with the settings of the main table.
#   SIZE=1.7B|0.6B SEED=42 OUTPUT_ROOT=<dir> bash opd/scripts/criticopd/baseline.sh <method> [Hydra overrides]
# Methods: opd fastopd grpo sft kd trd skd relayopd opsdt cgrpo
# Each run stops at the step selected in the main table (all methods use a constant learning rate).
# SFT and KD train on teacher solutions and TRD on teacher rewrites; create them first with offline_data.sh
# and pass the parquet in OFFLINE_DATA.
set -euo pipefail
method=${1:?usage: baseline.sh <method> [Hydra overrides]}; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${STUDENT_MODEL:?}" "${TEACHER_MODEL:?}" "${TRAIN_DATA:?}" "${BENCH:?}" "${OUTPUT_ROOT:?}"
export STUDENT_MODEL TEACHER_MODEL BENCH
SIZE=${SIZE:-1.7B}
export SEED=${SEED:-42} SFT_DATA_SEED=${SEED:-42}
export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-2} TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-2}
export SAVE_FREQ=${SAVE_FREQ:-20} TEST_FREQ=-1 VAL_BEFORE_TRAIN=False

declare -A STEPS_17=([opd]=80 [fastopd]=60 [grpo]=120 [sft]=40 [kd]=139 [trd]=40 [skd]=120 [relayopd]=40 [opsdt]=60 [cgrpo]=40)
declare -A STEPS_06=([opd]=40 [fastopd]=40 [grpo]=80 [sft]=139 [kd]=139 [trd]=139 [skd]=40 [relayopd]=60 [opsdt]=80 [cgrpo]=40)
if [[ "${SIZE}" == "0.6B" ]]; then steps=${STEPS_06[$method]:-}; else steps=${STEPS_17[$method]:-}; fi
[[ -n "${steps}" ]] || { echo "unknown method: ${method}"; exit 1; }
steps=${TOTAL_TRAINING_STEPS:-${steps}}
export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${method}_${SIZE}_seed${SEED}}
export EXP_ID=${EXP_ID:-$(basename "${OUTPUT_DIR}")}
export TRAIN_DATA

critic_loop=actor_rollout_ref.rollout.agent.default_agent_loop=critic_opd_agent
critic_env() {   # agent loop of CriticOPD, which also implements OPSD-T and Critique-GRPO
  export ROLLOUT_N=${ROLLOUT_N:-1} AGENT_WORKERS=2 MAX_RESPONSE_LENGTH=16384
  export CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_GIVE_GT=1
  export CRITIC_OPD_KERR=last CRITIC_OPD_CRITIC_TOKENS=2048 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL=
}

case "${method}" in
  opd)      exec bash "${SCRIPTS}/baselines/opd.sh" trainer.total_training_steps=${steps} "$@" ;;
  fastopd)  exec bash "${SCRIPTS}/baselines/fastopd/8192.sh" trainer.total_training_steps=${steps} "$@" ;;
  grpo)     export N_GPUS=4 TOTAL_TRAINING_STEPS=${steps}
            exec bash "${SCRIPTS}/baselines/grpo.sh" "$@" ;;
  sft)      export NUM_GPUS=4 TOTAL_TRAINING_STEPS=${steps} TRAIN_DATA=${OFFLINE_DATA:?teacher solutions from offline_data.sh}
            exec bash "${SCRIPTS}/baselines/sft.sh" ;;
  kd)       export TOTAL_TRAINING_STEPS=${steps} TRAIN_DATA=${OFFLINE_DATA:?teacher solutions from offline_data.sh}
            exec bash "${SCRIPTS}/baselines/seqkd.sh" "$@" ;;
  trd)      export TOTAL_TRAINING_STEPS=${steps} TRAIN_DATA=${OFFLINE_DATA:?teacher rewrites from offline_data.sh}
            exec bash "${SCRIPTS}/baselines/trd.sh" "$@" ;;
  skd)      export TOTAL_TRAINING_STEPS=${steps}
            exec bash "${SCRIPTS}/baselines/skd.sh" "$@" ;;
  relayopd) exec bash "${SCRIPTS}/relay_opd/train.sh" trainer.total_training_steps=${steps} "$@" ;;
  opsdt)    critic_env; export CRITIC_OPD_OPSD=1 CRITIC_OPD_LOSS=fb
            exec bash "${SCRIPTS}/baselines/opd.sh" ${critic_loop} trainer.total_training_steps=${steps} "$@" ;;
  cgrpo)    ROLLOUT_N=8 critic_env
            export CRITIC_OPD_LOSS=clean CRITIC_OPD_FB_ANSWER=1 CRITIC_OPD_CGRPO=1 CRITIC_OPD_CGRPO_N_INIT=7
            export USE_TASK_REWARDS=True ADV_ESTIMATOR=critique_grpo
            exec bash "${SCRIPTS}/baselines/opd.sh" ${critic_loop} trainer.total_training_steps=${steps} \
              actor_rollout_ref.actor.policy_loss.loss_mode=critique_grpo \
              actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
              actor_rollout_ref.actor.loss_scale_factor=16384 \
              actor_rollout_ref.actor.entropy_coeff=0.001 \
              distillation.distillation_loss.distillation_loss_coef=0.0 "$@" ;;
  *)        echo "unknown method: ${method}"; exit 1 ;;
esac
