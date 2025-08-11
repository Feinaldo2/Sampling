#!/bin/bash
# 修复版单个测试脚本

export CUDA_VISIBLE_DEVICES=0
cd /home/zhaoyifei/Sampling/slow-fast-sampling

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate slow_fast_sampling

echo "🧪 测试单个配置: 1.5|0.0|0.3"
echo "环境: $(which python3)"
echo "CUDA设备: $CUDA_VISIBLE_DEVICES"

python3 evaluation_script.py --model dream \
  --model_args "pretrained=/home/zhaoyifei/Sampling/Models/Dream-base,max_new_tokens=256,diffusion_steps=256,temperature=0.2,top_p=0.95,alg=entropy,alg_temp=0.0,add_bos_token=true,is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight=1.5|0.0|0.3,k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \
  --tasks gsm8k \
  --num_fewshot 8 \
  --batch_size 1 \
  --limit 5 \
  --output_path ./debug_test \
  --log_samples

echo "测试完成，返回码: $?"
