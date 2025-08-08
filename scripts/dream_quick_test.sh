#!/bin/bash
# 🚀 AdLLM快速验证脚本 - 只测试少量样本，快速验证权重效果

MODEL="/home/zhaoyifei/Sampling/Models/Dream-base"

# 评测参数（与完整测试保持一致）
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"

# ✅ 关键：限制样本数量进行快速测试
LIMIT=50  # 只测试50个样本，大约10-15分钟

# 测试不同的权重配置
declare -a WEIGHT_CONFIGS=(
    "1.0|0.0|0.0"      # 纯置信度基线
    "0.98|0.01|0.01"   # 极保守
    "0.95|0.03|0.02"   # 推荐配置
    "0.9|0.05|0.05"    # 平衡配置
    "0.8|0.1|0.1"      # 激进配置
)

declare -a CONFIG_NAMES=(
    "baseline"
    "conservative"
    "recommended"
    "balanced"
    "aggressive"
)

echo "🚀 开始AdLLM权重快速验证..."
echo "📊 每个配置测试${LIMIT}个样本，预计总时间：$(( ${#WEIGHT_CONFIGS[@]} * 15 ))分钟"
echo ""

# 创建结果目录
RESULTS_DIR="./quick_test_results"
mkdir -p ${RESULTS_DIR}

# 测试每个权重配置
for i in "${!WEIGHT_CONFIGS[@]}"; do
    STATIC_WEIGHT="${WEIGHT_CONFIGS[$i]}"
    CONFIG_NAME="${CONFIG_NAMES[$i]}"
    OUTPUT_PATH="${RESULTS_DIR}/${CONFIG_NAME}"
    
    echo "🔧 测试配置 ${CONFIG_NAME}: ${STATIC_WEIGHT}"
    echo "📁 输出路径: ${OUTPUT_PATH}"
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 运行测试
    accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
      --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight=${STATIC_WEIGHT},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \
      --tasks ${TASK} \
      --num_fewshot ${NUM_FEWSHOT} \
      --batch_size 1 \
      --limit ${LIMIT} \
      --output_path ${OUTPUT_PATH} \
      --log_samples > ${RESULTS_DIR}/${CONFIG_NAME}_log.txt 2>&1
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 提取分数
    SCORE=$(grep -r "exact_match,strict-match" ${OUTPUT_PATH}/ | head -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    
    echo "✅ 配置 ${CONFIG_NAME} 完成！"
    echo "   权重: ${STATIC_WEIGHT}"
    echo "   分数: ${SCORE}"
    echo "   用时: ${DURATION}秒"
    echo ""
done

echo "🎯 快速验证完成！结果汇总："
echo "=================================="
printf "%-12s %-15s %-8s %-8s\n" "配置" "权重" "分数" "用时"
echo "=================================="

for i in "${!WEIGHT_CONFIGS[@]}"; do
    CONFIG_NAME="${CONFIG_NAMES[$i]}"
    STATIC_WEIGHT="${WEIGHT_CONFIGS[$i]}"
    
    # 提取分数和时间
    if [ -f "${RESULTS_DIR}/${CONFIG_NAME}_log.txt" ]; then
        SCORE=$(grep -r "exact_match,strict-match" ${RESULTS_DIR}/${CONFIG_NAME}/ 2>/dev/null | head -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        DURATION=$(grep "用时:" ${RESULTS_DIR}/${CONFIG_NAME}_log.txt 2>/dev/null | tail -1 | grep -o '[0-9]\+秒' | head -1)
        
        printf "%-12s %-15s %-8s %-8s\n" "${CONFIG_NAME}" "${STATIC_WEIGHT}" "${SCORE:-N/A}" "${DURATION:-N/A}"
    fi
done

echo "=================================="
echo ""
echo "📋 详细结果查看："
echo "   结果目录: ${RESULTS_DIR}/"
echo "   日志文件: ${RESULTS_DIR}/*_log.txt"
echo ""
echo "🎯 选择最佳配置后，使用完整脚本进行最终验证！"
