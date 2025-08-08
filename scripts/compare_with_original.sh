#!/bin/bash

# 🔄 对比测试脚本：优化版本 vs 原始版本
# 确保参数配置与原始版本一致，进行公平对比

echo "🔄 开始对比测试：优化版本 vs 原始版本"
echo "=" * 60

# 通用配置
MODEL="/home/zhaoyifei/Sampling/Models/Dream-base"
ACCEL_CONFIG="eval_config.yaml"
TASK="gsm8k"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"

# Attention融合配置（两个版本相同）
USE_ATTENTION_FUSION="True"
FUSION_TYPE="static"
STATIC_WEIGHT="0.5|0.4|0.1"
DYNAMIC_WEIGHT_PATH=""
FEATURE_PATH=""

# ✅ 关键对比参数（与原始版本保持一致）
K_EXPLORATION_STEPS=4               # 两个版本都使用4
CYCLE_STABILITY_WINDOW=2            # 两个版本都使用2
USE_FAST_ATTENTION="True"           # 优化版本启用，原始版本不支持

echo "📊 测试配置："
echo "  - 模型: ${MODEL}"
echo "  - 任务: ${TASK}"
echo "  - Attention融合: ${USE_ATTENTION_FUSION}"
echo "  - 融合权重: ${STATIC_WEIGHT}"
echo "  - 探索步数: ${K_EXPLORATION_STEPS}"
echo "  - 稳定性窗口: ${CYCLE_STABILITY_WINDOW}"
echo ""

# 第一步：运行优化版本
echo "🚀 第1步：运行优化版本（您的改进版本）"
echo "输出路径: ./gsm8k_optimized_vs_original"
echo ""

OUTPUT_PATH_OPTIMIZED="./gsm8k_optimized_vs_original"

accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=${USE_ATTENTION_FUSION},fusion_type=${FUSION_TYPE},static_weight=${STATIC_WEIGHT},dynamic_weight_path=${DYNAMIC_WEIGHT_PATH},feature_path=${FEATURE_PATH},k_exploration_steps=${K_EXPLORATION_STEPS},cycle_length_stability_window=${CYCLE_STABILITY_WINDOW},use_fast_attention=${USE_FAST_ATTENTION}" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH_OPTIMIZED} \
  --log_samples

echo ""
echo "✅ 优化版本测试完成"
echo ""

# 第二步：运行原始版本（需要切换到原始目录）
echo "🔄 第2步：运行原始版本（/Slow-Fast-Sampling/）"
echo "输出路径: /home/zhaoyifei/Sampling/Slow-Fast-Sampling/slow-fast-sampling/gsm8k_original_comparison"
echo ""

# 切换到原始版本目录
cd /home/zhaoyifei/Sampling/Slow-Fast-Sampling/slow-fast-sampling

OUTPUT_PATH_ORIGINAL="./gsm8k_original_comparison"

# 运行原始版本（使用相同的参数）
accelerate launch --config_file ${ACCEL_CONFIG} evaluation_script.py --model dream \
  --model_args "pretrained=${MODEL},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=False,is_cfg_cache=False,k_exploration_steps=${K_EXPLORATION_STEPS},cycle_length_stability_window=${CYCLE_STABILITY_WINDOW}" \
  --tasks ${TASK} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size 1 \
  --output_path ${OUTPUT_PATH_ORIGINAL} \
  --log_samples

echo ""
echo "✅ 原始版本测试完成"
echo ""

# 切换回优化版本目录
cd /home/zhaoyifei/Sampling/slow-fast-sampling

# 第三步：对比结果
echo "📊 第3步：对比测试结果"
echo "=" * 60

echo ""
echo "📁 结果文件位置："
echo "  - 优化版本: /home/zhaoyifei/Sampling/slow-fast-sampling/${OUTPUT_PATH_OPTIMIZED}"
echo "  - 原始版本: /home/zhaoyifei/Sampling/Slow-Fast-Sampling/slow-fast-sampling/${OUTPUT_PATH_ORIGINAL}"
echo ""

echo "🔍 快速对比命令："
echo "# 查看优化版本结果"
echo "ls -la ${OUTPUT_PATH_OPTIMIZED}/"
echo "grep -r 'exact_match\\|strict-match\\|flexible-extract' ${OUTPUT_PATH_OPTIMIZED}/ || true"
echo ""
echo "# 查看原始版本结果"
echo "ls -la /home/zhaoyifei/Sampling/Slow-Fast-Sampling/slow-fast-sampling/${OUTPUT_PATH_ORIGINAL}/"
echo "grep -r 'exact_match\\|strict-match\\|flexible-extract' /home/zhaoyifei/Sampling/Slow-Fast-Sampling/slow-fast-sampling/${OUTPUT_PATH_ORIGINAL}/ || true"
echo ""

echo "📈 预期对比结果："
echo "  - 速度提升: 优化版本应该比原始版本快 5-7倍"
echo "  - 质量保持: 两个版本的exact_match分数应该相近或更高"
echo "  - 功能增强: 优化版本包含attention融合功能"
echo "  - 评测指标: GSM8K使用exact_match (strict-match和flexible-extract两种)"
echo ""

echo "✅ 对比测试脚本执行完成！"
echo "请查看上述路径的结果文件进行详细对比。"
