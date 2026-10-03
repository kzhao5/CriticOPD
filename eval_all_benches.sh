#!/bin/bash
# Submit the full 7-benchmark suite for one checkpoint of one run.
#   math    aime25 / amc23 / math500 (+aime24)  -> eval_base.sl, sampled n=8, done by the math grader
#   science mmlu_pro / gpqa_diamond             -> eval_mc.sl, greedy 1-sample
#   code    humanevalplus / lcb                 -> eval_mc.sl generates, the codeeval env scores later
# usage: bash eval_all_benches.sh <run> <step> [math|sci|code|all]
# Keep --time small: AssocGrpBillingRunMinutes budgets sum(GPUs x REQUESTED time) over RUNNING jobs
# and an over-budget job is CANCELLED ~2 s after it starts rather than queued.
set -euo pipefail
cd /home/kzhao2/Relay-OPD
RUN=${1:?run}; STEP=${2:?step}; WHAT=${3:-all}
CKPT=$PWD/outputs/checkpoints/${RUN}/global_step_${STEP}/actor/huggingface
[ -f "$CKPT/config.json" ] || { echo "!!! missing $CKPT"; exit 1; }
note() { echo "$1 $2" >> .jobids/evbase.txt; echo "🧪 $1 -> $2"; }
# Done = the LAST bench's summary exists, at top level (split DP layout) or in any shard (the
# replicated NUM_SHARDS_TOTAL=1 layout the training arms use writes ONLY shard_*/). The old check
# looked at the top level only, so every chaser re-submitted already-finished training-arm evals.
have() { local d="outputs/eval_out/$1/step_${STEP}" last="${2:-*}"
  compgen -G "$d/${last}.summary.json" >/dev/null || compgen -G "$d/shard_*/${last}.summary.json" >/dev/null; }
# 一条臂可能挂着两个 chaser(原训练 + 续训各一个), 两边都会调本脚本。若同名评测已在队列里,
# 再提一份会让两个 job 并发写同一个 shard 目录 -> 结果互相覆盖。按 job 名去重。
queued() { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$1"; }

if [ "$WHAT" = all ] || [ "$WHAT" = math ]; then
  # H100(cs2)+DP=2:实测未训练 base/teacher 的四个 bench 全套 25-35 min(A100 单卡 4 h 还超时)。
  # 训练过的 checkpoint 生成更长会慢不少,8 h 仍留足余量。
  # DP=2 + 8 h: measured cost of one checkpoint at DP=1 is ~35+34+67 min for aime24/aime25/amc23 and
  # ~400 min for math500 (500 x 8 generations at 16384); DP=2 halves that to ~4.5 h, so 4 h times out.
  # These only run after training frees the GrpBillingRunMinutes budget (submitted by after_train_evals.sl).
  # math_benchmarks.py skips benches whose summary.json already exists, so a retry only redoes what is missing.
  have "${RUN}_b16k" math500 || queued "ev16_${RUN}_${STEP}" || note "ev16_${RUN}_${STEP}" "$(sbatch --parsable --partition=cs --qos=cs --gres=gpu:a100:2 --time=8:00:00 \
    --job-name=ev16_${RUN}_${STEP} --export=ALL,RUN=$RUN,STEPS=$STEP,DP=2,NUM_SHARDS_TOTAL=1,N_SAMPLES=8,MAX_NEW=16384,MAX_MODEL_LEN=18433,EVAL_TAG=_b16k,EVAL_STOP_TOKEN_IDS='151643;151645' eval_base.sl)"
fi
# science + code share eval_mc.sl (greedy). 16384 budget so truncation is not an artefact of the cap.
if [ "$WHAT" = all ] || [ "$WHAT" = sci ]; then
  for b in mmlu_pro gpqa_diamond; do
    have "${RUN}_${b}" || queued "evmc_${RUN}_${b}_${STEP}" || note "evmc_${RUN}_${b}_${STEP}" "$(sbatch --parsable --partition=cs --qos=cs --gres=gpu:a100:2 --time=5:00:00 \
      --job-name=evmc_${RUN}_${b}_${STEP} --export=ALL,RUN=$RUN,MODEL=$CKPT,BENCH=$b,STEP=$STEP,DP=2,MAXNEW=16384 eval_mc.sl)"
  done
fi
if [ "$WHAT" = all ] || [ "$WHAT" = code ]; then
  for b in humanevalplus lcb; do
    have "${RUN}_${b}" || queued "evmc_${RUN}_${b}_${STEP}" || note "evmc_${RUN}_${b}_${STEP}" "$(sbatch --parsable --partition=cs --qos=cs --gres=gpu:a100:2 --time=5:00:00 \
      --job-name=evmc_${RUN}_${b}_${STEP} --export=ALL,RUN=$RUN,MODEL=$CKPT,BENCH=$b,STEP=$STEP,DP=2,MAXNEW=16384 eval_mc.sl)"
  done
fi
