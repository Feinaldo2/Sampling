#!/bin/bash

# Dream 本地模型路径
MODEL="/home/zhaoyifei/Sampling/Models/Dream-base"

# 评测参数
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"
OUTPUT_PATH="./gsm8k_dream_custom_log"

# 算法结构自定义参数
USE_ATTENTION_FUSION="True"         # True/False
FUSION_TYPE="static"                # static/dynamic
STATIC_WEIGHT="0.5|0.4|0.1"         # 仅 static 时有效
DYNAMIC_WEIGHT_PATH=""              # 仅 dynamic 时填写权重路径
FEATURE_PATH=""                     # 如有自定义特征文件则填写

# 运行 Dream 本地模型评测（支持自定义融合参数）
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=${USE_ATTENTION_FUSION},fusion_type=${FUSION_TYPE},static_weight=${STATIC_WEIGHT},dynamic_weight_path=${DYNAMIC_WEIGHT_PATH},feature_path=${FEATURE_PATH}" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH} \
  --log_samples 