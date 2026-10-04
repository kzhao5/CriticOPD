#!/bin/bash
# E9(独立标注第一个错误)+ E10(以错误为锚的 V^T / V^S)。用法: bash pipeline_e10.sh <out 目录> <名字前缀> [judge_first 的依赖] [anchors 的额外依赖]
set -euo pipefail
cd /home/kzhao2/Relay-OPD/diag2; B=$PWD/$1; P=$2; D1=${3:-}; D2=${4:-}
sbr() { local i o; for i in 1 2 3 4 5; do o=$(sbatch "$@" 2>/dev/null) && { echo $o; return 0; }; sleep 3; done; echo "sbatch 失败: $*" >&2; return 1; }
G=("--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-3" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-3" "--partition=m13h --qos=gpu --exclude=m13h-1-1")
gpu() { local name=$1 t=$2 dep=$3; shift 3; local ids=""
  for g in "${G[@]}"; do
    ids="$ids:$(sbr --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,DIAG_OUT=$B,CMD="$*" run_gpu.sl)"
  done; echo ${ids#:}; }
cpu() { local name=$1 dep=$2; shift 2
  sbr --parsable --account=als44 -p cs --qos=cs -c 40 --mem=120G --time=2:00:00 ${dep:+--dependency=$dep} --job-name=$name \
    --output=$PWD/logs/%x_%j.log --wrap="cd $PWD && DIAG_OUT=$B CRITIC_OPD_GRADER_PROCS=36 /home/kzhao2/.conda/envs/relay-opd/bin/python3 diag.py $*"; }
JF=""; for k in 0 1; do JF="$JF:$(gpu ${P}_jf$k 3:00:00 "${D1:+afterok:$D1}" judge_first $k 2)"; done
AN=$(cpu ${P}_anchors "afterany${JF}${D2:+,afterok:$D2}" anchors)
E=""; for a in teacher student; do for k in 0 1 2 3; do E="$E:$(gpu ${P}_e10_${a:0:1}$k 3:00:00 afterok:$AN cont $a e10 $k 4)"; done; done
G10=$(cpu ${P}_grade_e10 afterany${E} grade e10)
echo "judge_first ${JF#:} | anchors $AN | e10 ${E#:} | grade $G10" | tee -a $B/jobs.txt
