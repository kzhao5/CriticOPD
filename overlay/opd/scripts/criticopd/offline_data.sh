#!/usr/bin/env bash
# Offline data of the baselines that train on fixed sequences (temperature 1.0, top-p 1.0, up to 16,384 new tokens).
#   MODE=teacher  one teacher solution per DAPO prompt            -> SFT and KD
#   MODE=trd      a student solution and its teacher rewrite      -> TRD (one file per student model)
#   MODE=teacher|trd OUTPUT_PARQUET=<file> GPU_GROUPS="0;1;2;3" bash opd/scripts/criticopd/offline_data.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE=${MODE:?MODE=teacher|trd}
: "${STUDENT_MODEL:?}" "${TEACHER_MODEL:?}" "${TRAIN_DATA:?}" "${OUTPUT_PARQUET:?}"
GROUPS_STR=${GPU_GROUPS:-0;1;2;3}
export GEN_STOP_TOKEN_IDS=${GEN_STOP_TOKEN_IDS:-151643;151645}
export MAX_NEW=${MAX_NEW:-16384}
export STUDENT_MODEL TEACHER_MODEL SOURCE_DATA=${TRAIN_DATA} OUTPUT_PARQUET
mkdir -p "$(dirname "${OUTPUT_PARQUET}")"
if [[ "${MODE}" == "teacher" ]]; then
  GPU_GROUPS="${GROUPS_STR}" bash "${SCRIPTS}/data/teacher_trajectories.sh"
else
  STUDENT_GPU_GROUPS="${GROUPS_STR}" REWRITE_GPU_GROUPS="${GROUPS_STR}" bash "${SCRIPTS}/data/trd_trajectories.sh"
fi
