#!/bin/bash
# Score the code benchmarks for one checkpoint (CPU only, separate `codeeval` env).
# The math eval leaves `correct=False` for code benches on purpose (CODE_BENCHES in
# math_benchmarks.py) -- these two scorers are what actually produce the numbers.
# usage: bash score_code.sh <run> <step>
set -euo pipefail
cd /home/kzhao2/Relay-OPD
RUN=${1:?run}; STEP=${2:?step}
if [ -f "outputs/eval_out/${RUN}_humanevalplus/step_${STEP}/humanevalplus.jsonl" ] || \
   ls outputs/eval_out/${RUN}_humanevalplus/step_${STEP}/shard_*/humanevalplus.jsonl >/dev/null 2>&1; then
  J=$(sbatch --parsable --partition=cs --qos=cs --time=2:00:00 --job-name=hescore_${RUN}_${STEP} \
      --wrap="module load miniforge3; eval \"\$(conda shell.bash hook)\"; conda activate codeeval; export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; cd /home/kzhao2/Relay-OPD && python newbench/score_heplus.py $RUN $STEP")
  echo "hescore_${RUN}_${STEP} $J" | tee -a .jobids/evbase.txt
fi
if ls outputs/eval_out/${RUN}_lcb/step_${STEP}/*.jsonl outputs/eval_out/${RUN}_lcb/step_${STEP}/shard_*/*.jsonl >/dev/null 2>&1; then
  J=$(sbatch --parsable --export=ALL,RUN=$RUN,STEP=$STEP score_lcb.sl)
  echo "lcbscore_${RUN}_${STEP} $J" | tee -a .jobids/evbase.txt
fi
