#!/bin/bash
# 扩充批次(out_b)的下游:与主批次完全相同的 critic / E3 / E6 / E7 / E8 / 单错误对照。用法: bash pipeline_ext2.sh <closing 任务 id>
set -euo pipefail
cd /home/kzhao2/Relay-OPD/diag2; B=$PWD/out_b; C=$1
echo '{"rollouts": 2, "e1": {"teacher": 12, "student": 12}, "critic": 4, "judge": 2, "e6": {"student": 4, "teacher": 2}, "e7": {"teacher": 4, "student": 4}}' > $B/expect.json
sbr() { local i o; for i in 1 2 3 4 5; do o=$(sbatch "$@" 2>/dev/null) && { echo $o; return 0; }; sleep 3; done; echo "sbatch 失败: $*" >&2; return 1; }   # Slurm 偶发 Circular job dependency
G=("--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-3" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-3" "--partition=m13h --qos=gpu --exclude=m13h-1-1")
gpu() { local name=$1 t=$2 dep=$3; shift 3; local ids=""
  for g in "${G[@]}"; do
    ids="$ids:$(sbr --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,DIAG_OUT=$B,CMD="$*" run_gpu.sl)"
  done; echo ${ids#:}; }
cpu() { local name=$1 dep=$2; shift 2
  sbr --parsable --account=als44 -p cs --qos=cs -c 40 --mem=120G --time=3:00:00 ${dep:+--dependency=$dep} --job-name=$name \
    --output=$PWD/logs/%x_%j.log --wrap="cd $PWD && DIAG_OUT=$B CRITIC_OPD_GRADER_PROCS=36 /home/kzhao2/.conda/envs/relay-opd/bin/python3 diag.py $*"; }
SC=""; for a in teacher student; do for k in 0 1; do SC="$SC:$(gpu d_b_sc_${a:0:1}$k 2:00:00 afterok:$C score $a $k 2)"; done; done
CR=""; for k in 0 1 2 3; do CR="$CR:$(gpu d_b_cr$k 3:00:00 afterok:$C critic $k 4)"; done
CS=""; for k in 0 1; do CS="$CS:$(gpu d_b_crs$k 2:00:00 afterok:$C critic_single $k 2)"; done
MS=$(cpu d_b_merge_single afterany${CS} merge_single)
MC=$(cpu d_b_merge_cr afterany${CR} merge_critic)
J=""; for k in 0 1; do J="$J:$(gpu d_b_judge$k 5:00:00 afterok:$MC judge $k 2)"; done
MJ=$(cpu d_b_merge_judge afterany${J} merge_judge)
E6=""; for k in 0 1 2 3; do E6="$E6:$(gpu d_b_e6_s$k 4:00:00 afterok:$MC cont student e6 $k 4)"; done; for k in 0 1; do E6="$E6:$(gpu d_b_e6_t$k 4:00:00 afterok:$MC cont teacher e6 $k 2)"; done
G6=$(cpu d_b_grade_e6 afterany${E6} grade e6)
E7=""; for a in teacher student; do for k in 0 1 2 3; do E7="$E7:$(gpu d_b_e7_${a:0:1}$k 5:00:00 afterok:$G6 cont $a e7 $k 4)"; done; done
G7=$(cpu d_b_grade_e7 afterany${E7} grade e7)
echo "score ${SC#:} | critic ${CR#:} | single ${CS#:} | merge $MC | judge ${J#:} | e6 ${E6#:} | grade_e6 $G6 | e7 ${E7#:} | grade_e7 $G7" >> $B/jobs.txt
echo submitted
