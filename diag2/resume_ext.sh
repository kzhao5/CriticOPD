#!/bin/bash
# 扩充批次从 critic 第 0 片起的下游(2026-10-05 第 0 片在 cs-1-2 上因卡被占满失败,下游随之取消)
set -euo pipefail
cd /home/kzhao2/Relay-OPD/diag2; B=$PWD/out_b
sbr() { local i o; for i in 1 2 3 4 5; do o=$(sbatch "$@" 2>/dev/null) && { echo $o; return 0; }; sleep 3; done; echo "sbatch 失败: $*" >&2; return 1; }
G=("--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-1,cs-1-2,cs-1-3,cs-1-4" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-3,dw-2-3" "--partition=m13h --qos=gpu --exclude=m13h-1-1,m13h-2-2")
gpu() { local name=$1 t=$2 dep=$3; shift 3; local ids=""
  for g in "${G[@]}"; do ids="$ids:$(sbr --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,DIAG_OUT=$B,CMD="$*" run_gpu.sl)"; done; echo ${ids#:}; }
gpuj() { local name=$1 t=$2 dep=$3; shift 3; local ids=""   # 判定任务:另带 DIAG_JUDGE_* 环境
  for g in "${G[@]}"; do ids="$ids:$(sbr --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,DIAG_OUT=$B,DIAG_JUDGE_KINDS=partial,DIAG_JUDGE_TAG=_partial,CMD="$*" run_gpu.sl)"; done; echo ${ids#:}; }
cpu() { local name=$1 dep=$2; shift 2; sbr --parsable --account=als44 -p cs --qos=cs -c 40 --mem=120G --time=3:00:00 --exclude=cs-1-1,cs-1-3,cs-1-4 ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --wrap="cd $PWD && DIAG_OUT=$B CRITIC_OPD_GRADER_PROCS=36 /home/kzhao2/.conda/envs/relay-opd/bin/python3 diag.py $*"; }
CR=$(gpu d_b_cr0 3:00:00 "" critic 0 4)
MC=$(cpu d_b_merge_cr afterany:$CR merge_critic)
J=""; for k in 0 1; do J="$J:$(gpu d_b_judge$k 4:00:00 afterok:$MC judge $k 2)"; done
MJ=$(cpu d_b_merge_judge afterany${J} merge_judge)
JP=""; for k in 0 1 2; do JP="$JP:$(gpuj d_b_judgep$k 3:00:00 afterok:$MC judge $k 3)"; done
E6=""; for k in 0 1 2 3; do E6="$E6:$(gpu d_b_e6_s$k 4:00:00 afterok:$MC cont student e6 $k 4)"; done; for k in 0 1; do E6="$E6:$(gpu d_b_e6_t$k 4:00:00 afterok:$MC cont teacher e6 $k 2)"; done
G6=$(cpu d_b_grade_e6 afterany${E6} grade e6)
E7=""; for a in teacher student; do for k in 0 1 2 3; do E7="$E7:$(gpu d_b_e7_${a:0:1}$k 5:00:00 afterok:$G6 cont $a e7 $k 4)"; done; done
G7=$(cpu d_b_grade_e7 afterany${E7} grade e7)
AN=$(cpu d_b_anchors afterok:$MC anchors)
E10=""; for a in teacher student; do for k in 0 1 2 3; do E10="$E10:$(gpu d_b_e10_${a:0:1}$k 3:00:00 afterok:$AN cont $a e10 $k 4)"; done; done
G10=$(cpu d_b_grade_e10 afterany${E10} grade e10)
echo "resume: cr0 $CR | merge $MC | judge ${J#:} | judgep ${JP#:} | e6 ${E6#:} | e7 ${E7#:} | anchors $AN | e10 ${E10#:} | grades $G6 $G7 $G10 $MJ" | tee -a $B/jobs.txt | cut -c1-200
