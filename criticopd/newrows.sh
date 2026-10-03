#!/bin/bash
# 报告新增结果时只认 summary 文件,不从 results.tsv 的列里取数 ——
# 多行同时落地时 tail -n 的行数会和实际新增对不上,导致字段错位报出不存在的数字。
cd /home/kzhao2/Relay-OPD
S=/tmp/.seen_summaries; touch $S
find outputs/eval_out -path "*_u32/step_*" -name "*.summary.json" -newer $S 2>/dev/null | while read f; do
  /home/kzhao2/.conda/envs/relay-opd/bin/python3 -c "
import json,sys
f='$f'; d=json.load(open(f))
p=f.split('/'); run=p[2]; step=p[3].replace('step_','')
print(f\"  {run}@{step} {d.get('bench') or d.get('tag')} = {100*d['avg@k']:.2f} (n={d['n_problems']}, samples={d['n_samples']})\")"
done
touch $S
