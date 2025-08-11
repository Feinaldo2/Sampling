#!/bin/bash

# 设置工作目录
cd /home/zhaoyifei/Sampling

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate slow_fast_sampling

# 显示环境信息
echo "🔬 三维特征优化实验环境信息:"
echo "   Python: $(which python3)"
echo "   工作目录: $(pwd)"
echo "   Conda环境: $CONDA_DEFAULT_ENV"
echo "   GPU数量: $(nvidia-smi -L | wc -l)"
echo ""

# 显示GPU状态
echo "🚀 GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# 启动三维优化实验
echo "🔬 启动三维特征优化实验..."
echo "🎯 实验价值:"
echo "   1. 首次真正的三维独立特征优化"
echo "   2. confidence + entropy + attention_entropy 完全不同数据源"
echo "   3. 寻找三个特征的最佳协同效应"
echo "   4. 确定最终的融合置信度算法参数"
echo ""

python3 three_dimensional_optimization.py --samples 150

echo ""
echo "🏁 三维特征优化实验完成！"
echo "结束时间: $(date)"

# 自动退出
sleep 2
