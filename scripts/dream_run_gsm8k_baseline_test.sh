#!/bin/bash

# 🔍 基线测试：完全复制原始Slow-Fast的配置，但使用优化的代码
# 用于确定质量下降是否由attention融合引起

# Dream 本地模型路径（与原始Slow-Fast完全一致）
MODEL="/home/zhaoyifei/Sampling/Models/Dream-base"

# 评测参数（与原始Slow-Fast完全一致）
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"
OUTPUT_PATH="./gsm8k_baseline_test_log"

# ❌ 关闭所有attention融合功能（测试基线性能）
USE_ATTENTION_FUSION="False"        # 关闭attention融合
FUSION_TYPE="static"                
STATIC_WEIGHT="0.5|0.4|0.1"         
DYNAMIC_WEIGHT_PATH=""              
FEATURE_PATH=""                     

# ✅ 保持与原始Slow-Fast一致的算法参数
K_EXPLORATION_STEPS=6               # 与原始版本完全一致
CYCLE_STABILITY_WINDOW=2            # 与原始版本完全一致
USE_FAST_ATTENTION="True"           # 仅保留向量化优化

echo "🔍 开始基线测试（关闭attention融合）"
echo "📊 配置信息："
echo "  - 模型路径: ${MODEL}"
echo "  - 输出路径: ${OUTPUT_PATH}"
echo "  - Attention融合: ${USE_ATTENTION_FUSION} (关闭)"
echo "  - 探索步数: ${K_EXPLORATION_STEPS}"
echo "  - 向量化优化: ${USE_FAST_ATTENTION}"
echo ""

# 运行基线测试（完全复制原始Slow-Fast参数）
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=${USE_ATTENTION_FUSION},fusion_type=${FUSION_TYPE},static_weight=${STATIC_WEIGHT},dynamic_weight_path=${DYNAMIC_WEIGHT_PATH},feature_path=${FEATURE_PATH},k_exploration_steps=${K_EXPLORATION_STEPS},cycle_length_stability_window=${CYCLE_STABILITY_WINDOW},use_fast_attention=${USE_FAST_ATTENTION}" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH} \
  --log_samples

echo ""
echo "✅ 基线测试完成！"
echo "📁 结果保存在: ${OUTPUT_PATH}"
echo ""
echo "🔍 预期结果："
echo "  - 如果分数接近0.74-0.75，说明问题在attention融合"
echo "  - 如果分数仍然是0.52，说明问题在其他地方"
