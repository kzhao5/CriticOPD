#!/bin/bash
# 扩充批次(out_b):主批次没用到的错误轨迹(全部写完的),只跑到 E1 + 关闭点;下游是否合并另定。
set -euo pipefail
cd /home/kzhao2/Relay-OPD/diag2; mkdir -p out_b logs
B=$PWD/out_b
echo '{"rollouts": 2, "e1": {"teacher": 12, "student": 12}}' > $B/expect.json
G=("--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-3" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-3" "--partition=m13h --qos=gpu --exclude=m13h-1-1")
gpu() { local name=$1 t=$2 dep=$3; shift 3; local ids=""
  for g in "${G[@]}"; do
    ids="$ids:$(sbatch --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,DIAG_OUT=$B,CMD="$*" run_gpu.sl)"
  done; echo ${ids#:}; }
cpu() { local name=$1 dep=$2; shift 2
  sbatch --parsable --account=als44 -p cs --qos=cs -c 40 --mem=120G --time=3:00:00 ${dep:+--dependency=$dep} --job-name=$name \
    --output=$PWD/logs/%x_%j.log --wrap="cd $PWD && DIAG_OUT=$B DIAG_EXTEND_FROM=$PWD/out DIAG_NC=0 DIAG_NW=1000 CRITIC_OPD_GRADER_PROCS=36 /home/kzhao2/.conda/envs/relay-opd/bin/python3 diag.py $*"; }
S=$(cpu d_b_select "" select)
E1=""; for a in teacher student; do for k in $(seq 0 11); do E1="$E1:$(gpu d_b_e1_${a:0:1}$k 5:00:00 afterok:$S cont $a e1 $k 12)"; done; done
G1=$(cpu d_b_grade_e1 afterany${E1} grade e1)
C=$(cpu d_b_closing afterok:$G1 closing)
echo "select $S | e1 ${E1#:} | grade $G1 | closing $C" | tee $B/jobs.txt
