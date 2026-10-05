#!/bin/bash
# 按依赖一次性提交 Section 2 诊断实验的全部阶段。GPU 任务多分区排队;CPU 阶段(判分等)在 cs 分区跑。
# 用法: bash pipeline.sh                    从学生做题开始
#       ROLL=<id:id> bash pipeline.sh       做题已经交过(给出那两个任务的 id),从 select 开始接
set -euo pipefail
cd /home/kzhao2/Relay-OPD/diag2; mkdir -p out logs
# 各阶段应有的分片数:判分/合并前逐片核对,缺片就失败,下游不会在残缺数据上运行
echo '{"rollouts": 2, "e1": {"teacher": 12, "student": 12}, "critic": 4, "judge": 2, "e6": {"student": 4, "teacher": 2}, "e7": {"teacher": 4, "student": 4}}' > out/expect.json
G=("--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-3" "--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-3" "--partition=m13h --qos=gpu --exclude=m13h-1-1")
gpu() {  # gpu <任务名> <时长> <依赖或空> <diag.py 参数>
  local name=$1 t=$2 dep=$3; shift 3; local ids=""
  for g in "${G[@]}"; do
    ids="$ids:$(sbatch --parsable $g --time=$t ${dep:+--dependency=$dep} --job-name=$name --output=$PWD/logs/%x_%j.log --export=ALL,CMD="$*" run_gpu.sl)"
  done; echo ${ids#:}
}
cpu() {  # cpu <任务名> <依赖> <diag.py 参数>
  local name=$1 dep=$2; shift 2
  sbatch --parsable --account=als44 -p cs --qos=cs -c 40 --mem=120G --time=3:00:00 ${dep:+--dependency=$dep} --job-name=$name \
    --output=$PWD/logs/%x_%j.log --wrap="cd $PWD && CRITIC_OPD_GRADER_PROCS=36 /home/kzhao2/.conda/envs/relay-opd/bin/python3 diag.py $*"
}
[ "${1:-}" = rollouts_only ] && { echo "$(gpu d_roll0 3:00:00 "" rollouts 0 2):$(gpu d_roll1 3:00:00 "" rollouts 1 2)"; exit 0; }
R=${ROLL:-$(gpu d_roll0 3:00:00 "" rollouts 0 2):$(gpu d_roll1 3:00:00 "" rollouts 1 2)}
S=$(cpu d_select afterany:$R select)
E1=""; for a in teacher student; do for k in $(seq 0 11); do E1="$E1:$(gpu d_e1_${a:0:1}$k 5:00:00 afterok:$S cont $a e1 $k 12)"; done; done
G1=$(cpu d_grade_e1 afterany${E1} grade e1)
C=$(cpu d_closing afterok:$G1 closing)
SC=""; for a in teacher student; do for k in 0 1; do SC="$SC:$(gpu d_sc_${a:0:1}$k 2:00:00 afterok:$C score $a $k 2)"; done; done
CR=""; for k in 0 1 2 3; do CR="$CR:$(gpu d_cr$k 3:00:00 afterok:$C critic $k 4)"; done
MC=$(cpu d_merge_cr afterany${CR} merge_critic)
J=""; for k in 0 1; do J="$J:$(gpu d_judge$k 5:00:00 afterok:$MC judge $k 2)"; done
MJ=$(cpu d_merge_judge afterany${J} merge_judge)
E6=""; for k in 0 1 2 3; do E6="$E6:$(gpu d_e6_s$k 4:00:00 afterok:$MC cont student e6 $k 4)"; done; for k in 0 1; do E6="$E6:$(gpu d_e6_t$k 4:00:00 afterok:$MC cont teacher e6 $k 2)"; done
G6=$(cpu d_grade_e6 afterany${E6} grade e6)
E7=""; for a in teacher student; do for k in 0 1 2 3; do E7="$E7:$(gpu d_e7_${a:0:1}$k 5:00:00 afterok:$G6 cont $a e7 $k 4)"; done; done
G7=$(cpu d_grade_e7 afterany${E7} grade e7)
echo "rollouts $R | select $S | e1 ${E1#:} | grade_e1 $G1 | closing $C | score ${SC#:} | critic ${CR#:} | merge $MC | judge ${J#:} | e6 ${E6#:} | grade_e6 $G6 | e7 ${E7#:} | grade_e7 $G7" | tee out/jobs.txt
