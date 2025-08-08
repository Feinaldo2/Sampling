#!/bin/bash
# Dream 本地模型路径（与原始Slow-Fast保持一致）
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
STATIC_WEIGHT="0.7|0.2|0.1"         # 仅 static 时有效
DYNAMIC_WEIGHT_PATH=""              # 仅 dynamic 时填写权重路径
FEATURE_PATH=""                     # 如有自定义特征文件则填写

# ✅ 与原始Slow-Fast保持完全一致的算法参数
K_EXPLORATION_STEPS=6               # 探索步数：与原始版本完全一致
CYCLE_STABILITY_WINDOW=2            # 稳定性窗口：与原始版本完全一致
USE_FAST_ATTENTION="True"           # 仅启用向量化优化（不改变算法逻辑）

# 运行 Dream 本地模型评测（添加原始Slow-Fast的关键参数）
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=${USE_ATTENTION_FUSION},fusion_type=${FUSION_TYPE},static_weight=${STATIC_WEIGHT},dynamic_weight_path=${DYNAMIC_WEIGHT_PATH},feature_path=${FEATURE_PATH},k_exploration_steps=${K_EXPLORATION_STEPS},cycle_length_stability_window=${CYCLE_STABILITY_WINDOW},use_fast_attention=${USE_FAST_ATTENTION}" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH} \
  --log_samples

echo "Completed evaluation for ${TASK}"