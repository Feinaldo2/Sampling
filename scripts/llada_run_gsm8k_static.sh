#!/bin/bash

# 选择模型
MODEL="$HOME/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/ce71e3c2523f535e022bccedbda192eb8869fd44"

# 评测配置
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=4
BLOCK_LENGTH=256
GEN_LENGTH=256
CFG_SCALE=0.0
OUTPUT_PATH="./gsm8k_log_static"

# 运行（静态权重 attention 融合）
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py -m lm_eval \
  --model LLADA \
  --tasks ${TASK} \
  --batch_size 1 \
  --model_args "pretrained=${MODEL},use_static_weight=True" \
  --gen_kwargs "block_length=${BLOCK_LENGTH},gen_length=${GEN_LENGTH},cfg_scale=${CFG_SCALE}" \
  --num_fewshot ${NUM_FEWSHOT} \
  --output_path ${OUTPUT_PATH} \
  --log_samples 