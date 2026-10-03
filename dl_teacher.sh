#!/bin/bash
export HF_HUB_ENABLE_HXFER=1
/home/kzhao2/.local/bin/hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507 \
  && echo "=== TEACHER DL OK ===" || echo "=== TEACHER DL FAIL ==="
