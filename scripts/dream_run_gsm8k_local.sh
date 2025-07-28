#!/bin/bash

# Dream 本地模型路径
MODEL="$HOME/Model/Dream-v0-Base-7B"

# 评测参数
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"
OUTPUT_PATH="./gsm8k_dream_local_log"

# 运行 Dream 本地模型评测
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH} \
  --log_samples 