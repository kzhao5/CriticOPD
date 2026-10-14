#!/usr/bin/env bash
# Evaluation protocol of the paper: temperature 1.0, top-p 1.0, up to 32,768 new tokens; avg@32 on AIME24,
# AIME25 and AMC23, avg@8 on MATH500, OlympiadBench and Minerva. Prints the six scores and their average.
#   MODEL=<run>/global_step_40/actor/huggingface DATA_DIR=<eval parquets> OUT_ROOT=<dir> RUN_NAME=<name> \
#   STEP=40 bash opd/scripts/criticopd/eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_SH="${SCRIPT_DIR}/../evaluation/math.sh"
: "${MODEL:?}" "${DATA_DIR:?}" "${OUT_ROOT:?}" "${RUN_NAME:?}" "${STEP:?}"
export MODEL DATA_DIR OUT_ROOT RUN_NAME STEP
export MAX_NEW=${MAX_NEW:-32768} MAX_MODEL_LEN=${MAX_MODEL_LEN:-34817} TEMPERATURE=1.0 TOP_P=1.0
export DP_SIZE=${DP_SIZE:-4} TP=${TP:-1} SEED=${SEED:-42}

BENCHES=aime24,aime25,amc23 N_SAMPLES=32 bash "${MATH_SH}"
BENCHES=math500,olympiad,minerva N_SAMPLES=8 bash "${MATH_SH}"

python3 - "${OUT_ROOT}/${RUN_NAME}/step_${STEP}" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
names = ["aime24", "aime25", "amc23", "math500", "olympiad", "minerva"]
acc = [100 * json.loads((d / f"{b}.summary.json").read_text())["avg@k"] for b in names]
for b, a in zip(names, acc):
    print(f"{b:9s} {a:6.2f}")
print(f"{'average':9s} {sum(acc) / len(acc):6.2f}")
PY
