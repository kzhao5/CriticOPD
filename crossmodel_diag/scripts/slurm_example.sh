#!/bin/bash
# The same stages as run_all.sh, sharded over GPUs with Slurm dependencies. Set SBATCH_GPU / SBATCH_CPU for your cluster.
# Usage: bash scripts/slurm_example.sh configs/olmo.yaml
# GPU time of the Qwen run: about 64 GPU-hours, 46 of them in `cont * e1`; every shard below took < 6 hours.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1
SBATCH_GPU=${SBATCH_GPU:-"--gres=gpu:1 --cpus-per-task=8 --mem=96G --time=12:00:00"}
SBATCH_CPU=${SBATCH_CPU:-"--cpus-per-task=16 --mem=64G --time=4:00:00"}
PY="python -m criticopd_diag.diag --config $CFG"
gpu() { local name=$1 dep=$2; shift 2; sbatch --parsable $SBATCH_GPU ${dep:+--dependency=$dep} -J $name -o logs/%x_%j.log --wrap="$PY $*"; }
cpu() { local name=$1 dep=$2; shift 2; sbatch --parsable $SBATCH_CPU ${dep:+--dependency=$dep} -J $name -o logs/%x_%j.log --wrap="$PY $*"; }
mkdir -p logs
R=$(gpu d_roll0 "" rollouts 0 2):$(gpu d_roll1 "" rollouts 1 2)
S=$(cpu d_select afterok:$R select)
E1=""; for a in teacher student; do for k in $(seq 0 11); do E1="$E1:$(gpu d_e1_${a:0:1}$k afterok:$S cont $a e1 $k 12)"; done; done
G1=$(cpu d_grade_e1 afterok${E1} grade e1)
C=$(cpu d_closing afterok:$G1 closing)
CR=""; for k in 0 1; do CR="$CR:$(gpu d_cr$k afterok:$C critic $k 2)"; done
MC=$(cpu d_merge afterok${CR} merge_critic)
J=""; for k in 0 1; do J="$J:$(gpu d_judge$k afterok:$MC judge $k 2)"; done
E6=""; for a in student teacher; do for k in 0 1 2 3; do E6="$E6:$(gpu d_e6_${a:0:1}$k afterok:$MC cont $a e6 $k 4)"; done; done
G6=$(cpu d_grade_e6 afterok${E6} grade e6)
E7=""; for a in teacher student; do for k in 0 1 2 3; do E7="$E7:$(gpu d_e7_${a:0:1}$k afterok:$G6 cont $a e7 $k 4)"; done; done
G7=$(cpu d_grade_e7 afterok${E7} grade e7)
P=$(cpu d_pack afterok:$G7${J} pack)
echo "submitted; the last job (pack: figures + return.tar.gz) is $P"
