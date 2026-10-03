#!/bin/bash
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:20:00
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#
# Fires (afterany) when a paper-baseline run's last training segment ends and queues its math evals with the
# base-table protocol (_b16k, 8 samples, 16384, both stop tokens) at every saved step.
# SFT (verl sft_trainer) writes global_step_N/huggingface instead of global_step_N/actor/huggingface; link
# the latter so the shared eval scripts find it. Nothing is copied or overwritten.
set -uo pipefail
cd /home/kzhao2/Relay-OPD
RUN=${RUN:?}
for d in outputs/checkpoints/$RUN/global_step_*; do
  if [ -f "$d/huggingface/config.json" ] && [ ! -e "$d/actor/huggingface" ]; then
    mkdir -p "$d/actor" && ln -s ../huggingface "$d/actor/huggingface" && echo "[chase] linked $d/actor/huggingface"
  fi
done
RUN=$RUN STEPS="20 40 60 80 100 120 139" WHAT=math bash after_train_evals.sl
