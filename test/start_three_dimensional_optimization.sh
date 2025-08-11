#!/bin/bash
# 🔬 启动三维特征优化实验

echo "🔬 三维特征优化实验启动脚本"
echo "=" * 80

# 创建screen会话名称
SESSION_NAME="three_dimensional_optimization"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "📊 实验配置:"
echo "   会话名称: ${SESSION_NAME}"
echo "   开始时间: $(date)"
echo "   样本数量: 150个/配置"
echo "   目标: 三维空间最优权重搜索"
echo ""

echo "🔬 三维特征空间:"
echo "   维度1 - Confidence: 固定1.00 (基准维度)"
echo "   维度2 - Entropy: 9个权重 [-0.10, -0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06]"
echo "   维度3 - Attention_entropy: 9个权重 [-0.15, -0.12, -0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09]"
echo "   搜索空间: 9×9 = 81个组合 + 3个基准 = 84个配置"
echo ""

echo "📈 基准对比:"
echo "   当前最佳: 0.770 (权重: 1.00|0.00|-0.04)"
echo "   pmass最佳: 0.755 (权重: 1.00|0.00|-0.02)"
echo "   baseline: 0.66 (权重: 1.00|0.00|0.00)"
echo ""

echo "🎯 优化目标:"
echo "   1. 在三维空间中找到全局最优权重组合"
echo "   2. 超越当前最佳0.770分"
echo "   3. 验证三个特征的最佳协同效应"
echo "   4. 确定最终的融合置信度算法"
echo ""

echo "⏰ 预计时间:"
echo "   每个配置: ~25分钟 (150样本)"
echo "   总配置: 84个"
echo "   总时间: ~2.5小时 (8个GPU并行，不限时间)"
echo ""

# 检查是否已有同名会话
if screen -list | grep -q "${SESSION_NAME}"; then
    echo "⚠️  发现已存在的会话: ${SESSION_NAME}"
    echo "终止现有会话并重新开始..."
    screen -S "${SESSION_NAME}" -X quit
    sleep 2
fi

# 创建启动脚本
STARTUP_SCRIPT="/tmp/three_dimensional_optimization_startup_${TIMESTAMP}.sh"
cat > "${STARTUP_SCRIPT}" << 'EOF'
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
EOF

chmod +x "${STARTUP_SCRIPT}"

echo "🚀 启动screen会话..."
echo "   会话名称: ${SESSION_NAME}"
echo ""

# 启动screen会话并自动分离
screen -dmS "${SESSION_NAME}" bash "${STARTUP_SCRIPT}"

# 等待会话启动
sleep 2

# 检查会话状态
if screen -list | grep -q "${SESSION_NAME}"; then
    echo "✅ Screen会话启动成功！"
    echo ""
    echo "📋 管理命令:"
    echo "   查看会话: screen -r ${SESSION_NAME}"
    echo "   分离会话: Ctrl+A, D"
    echo "   终止会话: screen -S ${SESSION_NAME} -X quit"
    echo ""
    echo "📊 监控命令:"
    echo "   实时查看结果: tail -f three_dimensional_optimization/results.csv"
    echo "   查看GPU状态: watch -n 30 nvidia-smi"
    echo ""
    echo "🎯 实验价值:"
    echo "   ✅ 首次真正的三维独立特征优化"
    echo "   ✅ 三个完全不同数据源的特征融合"
    echo "   ✅ 9×9=81个权重组合全面搜索"
    echo "   ✅ 寻找超越0.770的最优配置"
    
    echo ""
    echo "🔗 要查看实验进展，请运行:"
    echo "   screen -r ${SESSION_NAME}"
    
else
    echo "❌ Screen会话启动失败！"
    exit 1
fi

echo ""
echo "🔬 三维特征优化实验已启动！"
echo "💡 这将是最终的融合置信度算法优化！"
