#!/bin/bash
# 🚀 单GPU测试脚本 - 在指定GPU上运行测试

# 检查参数
if [ $# -ne 4 ]; then
    echo "用法: $0 <GPU_ID> <权重配置> <配置名称> <样本数量>"
    echo "示例: $0 0 '1.5|0.0|0.3' 'high_conf_attention' 150"
    exit 1
fi

GPU_ID="$1"
STATIC_WEIGHT="$2"
CONFIG_NAME="$3"
LIMIT="$4"

MODEL="/home/zhaoyifei/Sampling/Models/Dream-base"

# 评测参数
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"

OUTPUT_PATH="./gpu_test_results/gpu${GPU_ID}_${CONFIG_NAME}"

echo "🚀 GPU${GPU_ID} 测试配置: ${CONFIG_NAME}"
echo "   权重: ${STATIC_WEIGHT}"
echo "   样本数: ${LIMIT}"
echo "   GPU: ${GPU_ID}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"
mkdir -p "./gpu_test_results"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date)"

# 激活conda环境
source /home/zhaoyifei/miniconda3/etc/profile.d/conda.sh
conda activate slow_fast_sampling

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 运行测试 - 单GPU模式
python evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight=${STATIC_WEIGHT},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --limit ${LIMIT} \
  --output_path ${OUTPUT_PATH} \
  --log_samples

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ GPU${GPU_ID} 测试完成！"
echo "结束时间: $(date)"
echo "用时: ${DURATION}秒 ($(($DURATION / 60))分钟)"

# 提取分数
if [ -f "${OUTPUT_PATH}"/results_*.json ]; then
    SCORE=$(grep -r "exact_match,strict-match" "${OUTPUT_PATH}"/ | head -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    TPS=$(grep -r "TPS" "${OUTPUT_PATH}"/ | head -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    echo "分数: ${SCORE}"
    echo "TPS: ${TPS}"
    
    # 保存结果摘要
    echo "$(date '+%Y-%m-%d %H:%M:%S'),GPU${GPU_ID},${CONFIG_NAME},${STATIC_WEIGHT},${LIMIT},${SCORE},${TPS},${DURATION}" >> gpu_test_results/summary.csv
else
    echo "❌ 未找到结果文件"
    echo "$(date '+%Y-%m-%d %H:%M:%S'),GPU${GPU_ID},${CONFIG_NAME},${STATIC_WEIGHT},${LIMIT},ERROR,0,${DURATION}" >> gpu_test_results/summary.csv
fi
