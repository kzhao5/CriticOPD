#!/bin/bash
set -xeuo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda create -n relay-opd python=3.12 -y
conda activate relay-opd
python -m pip install --upgrade pip setuptools wheel
cd /home/kzhao2/Relay-OPD/relay-opd
python -m pip install -c environment/vllm-constraints.txt vllm==0.21.0
python -m pip install -e .
python -m pip install -r requirements-relay-opd.txt
python environment/verify_install.py && echo "=== ENV BUILD OK ==="
