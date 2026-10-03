#!/bin/bash
# 11 个模型 x 6 个 OOD benchmark,统一协议:4 samples,MC 上限 8192 / code 上限 4096。
# 全部重跑(不复用已有的 mmlu_pro / humanevalplus)—— 那些用的是 12032 全量且采样数 2/4 不一致,
# 混协议会重演 olympiad 16-vs-32 samples 那个问题。
set -u
P=0
sub() { # sub <run_name> <model_or_ckpt> <step> <bench> <max_new> <hours>
  case $((P % 3)) in
    0) PART="--partition=cs --qos=cs" ;;
    1) PART="--partition=m13h" ;;
    2) PART="--partition=dw --qos=dw87 --exclude=dw-2-4" ;;
  esac
  P=$((P+1))
  local EXTRA=""
  [[ "$2" == /* ]] && EXTRA="MODEL=$2," || EXTRA="RUN=$2,"
  sbatch --parsable $PART --gres=gpu:1 --time=${6}:00:00 -J "ood" \
    --export=ALL,RUN_NAME=$1,${EXTRA}STEP=$3,BENCHES=$4,N_SAMPLES=4,MAX_NEW=$5,MAX_MODEL_LEN=$(( $5 + 2049 )),EVAL_STOP_TOKEN_IDS='151643;151645' \
    eval_ood.sl
}
MODELS=(
  "student_1p7b_ood|/home/kzhao2/OPD/model/Qwen3-1.7B|0"
  "teacher_4b_ood|/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507|0"
  "sft_pt17b_ood|sft_pt17b|40"
  "seqkd_pt17b_ood|seqkd_pt17b|139"
  "grpo_pt17b_ood|grpo_pt17b|120"
  "opd_1p7b_ood|opd_1p7b|80"
  "trd_pt17b_ood|trd_pt17b|40"
  "fastopd8192_1p7b_ood|fastopd8192_1p7b|60"
  "skd_pt17b_ood|skd_pt17b|120"
  "relay_1p7b_ood|relay_1p7b|40"
  "lucid_T1donly_pt17b_ood|lucid_T1donly_pt17b|60"
)
# bench | max_new | hours
BENCHES=("mmlu_mini|8192|10" "mmlu_pro_2k|8192|10" "arc_c|8192|6" "obqa|8192|3"
         "humanevalplus|4096|2" "mbpp|4096|3")
n=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r nm src st <<< "$m"
  for b in "${BENCHES[@]}"; do
    IFS='|' read -r bench mx hr <<< "$b"
    printf "  %-26s %-14s " "$nm" "$bench"; sub "$nm" "$src" "$st" "$bench" "$mx" "$hr"
    n=$((n+1))
  done
done
echo "共提交 $n 个 job"
