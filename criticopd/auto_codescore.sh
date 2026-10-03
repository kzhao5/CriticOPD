#!/bin/bash
# OOD 两个代码题:生成完成(有 summary)但还没 evalplus 分数的,逐个打分。
# math.sh 给代码题的 avg@k 只是占位(数学判分器判不了代码),真实分数在 <bench>_score.json。
cd /home/kzhao2/Relay-OPD
PY=/home/kzhao2/.conda/envs/codeeval/bin/python
while :; do
  for S in outputs/eval_out/*_ood_u32/step_*/{humanevalplus,mbpp}.summary.json; do
    [ -f "$S" ] || continue
    D=$(dirname "$S"); B=$(basename "$S" .summary.json)
    [ -f "$D/${B}_score.json" ] && continue
    RUN=$(echo "$D" | cut -d/ -f3); STEP=$(basename "$D" | sed 's/step_//')
    echo "$(date +%H:%M) 打分 $RUN@$STEP $B"
    nice -n 10 $PY newbench/score_code_ood.py "$RUN" "$STEP" "$B" 2>&1 | tail -1 | sed 's/^/    /'
  done
  sleep 300
done
