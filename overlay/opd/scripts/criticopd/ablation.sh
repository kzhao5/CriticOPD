#!/usr/bin/env bash
# Ablations and sensitivity analysis of the paper (Qwen3-1.7B, seed 42, 40 steps).
#   OUTPUT_ROOT=<dir> bash opd/scripts/criticopd/ablation.sh <variant> [Hydra overrides]
# Every variant starts from the final configuration of train.sh and changes the switches below.
#   no_answer        the feedback contains only the critique (CRITIC_OPD_FB_ANSWER=0)
#   answer_only      the feedback states only the reference answer
#   generic          the feedback is a fixed generic hint
#   teacher_suffix   the teacher writes the suffix after the cut point            (on no_answer)
#   no_fb_scoring    the teacher scores the suffix without the feedback           (on no_answer)
#   k1 k2 k3         the cut follows the k-th listed error instead of the last    (on no_answer)
#   limit512 limit1024 limit4096   length limit of the critique                   (on no_answer)
set -euo pipefail
variant=${1:?usage: ablation.sh <variant>}; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUTPUT_ROOT:?}"
export SEED=${SEED:-42} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/ablation_${variant}_seed${SEED}}
case "${variant}" in
  no_answer)      export CRITIC_OPD_FB_ANSWER=0 ;;
  answer_only)    export CRITIC_OPD_FB_MODE=answer ;;
  generic)        export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_FB_MODE=generic ;;
  teacher_suffix) export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_REWRITER=teacher ;;
  no_fb_scoring)  export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_LOSS=clean ;;
  k1|k2|k3)       export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_KERR=${variant#k} ;;
  limit512|limit1024|limit4096)
                  export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_CRITIC_TOKENS=${variant#limit} CRITIC_OPD_CRITIC_TOKENS_EXACT=1 ;;
  *)              echo "unknown variant: ${variant}"; exit 1 ;;
esac
exec bash "${SCRIPT_DIR}/train.sh" "$@"
