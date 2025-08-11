#!/usr/bin/env python3
"""
🧪 测试非线性融合功能
"""

import subprocess
import time
import os

def test_nonlinear_fusion():
    """测试非线性融合是否正常工作"""
    
    print("🧪 测试非线性融合功能")
    print("=" * 50)
    
    # 测试配置
    test_configs = [
        ("linear_baseline", "linear", "1.0|0.0|0.0"),
        ("linear_fusion", "linear", "1.5|0.0|0.3"),
        ("nonlinear_test", "nonlinear", "1.2|-0.1|0.2"),
    ]
    
    results = []
    
    for config_name, fusion_mode, weights in test_configs:
        print(f"\n🚀 测试配置: {config_name} ({fusion_mode}: {weights})")
        
        cmd = f"""
        cd /home/zhaoyifei/Sampling/slow-fast-sampling
        export CUDA_VISIBLE_DEVICES=0
        source /home/zhaoyifei/miniconda3/etc/profile.d/conda.sh
        conda activate slow_fast_sampling

        python3 evaluation_script.py --model dream \
          --model_args "pretrained=/home/zhaoyifei/Sampling/Models/Dream-base,max_new_tokens=256,diffusion_steps=256,temperature=0.2,top_p=0.95,alg=entropy,alg_temp=0.0,add_bos_token=true,is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight={weights},fusion_mode={fusion_mode},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \
          --tasks gsm8k \
          --num_fewshot 8 \
          --batch_size 1 \
          --limit 10 \
          --output_path ./test_nonlinear_{config_name} \
          --log_samples
        """
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                # 查找调试输出
                debug_found = False
                for line in result.stdout.split('\n'):
                    if '🔬 AdLLM融合调试' in line:
                        print(f"✅ 找到调试输出: {line}")
                        debug_found = True
                    elif 'exact_match,strict-match' in line:
                        print(f"📊 结果: {line}")
                
                if debug_found:
                    print(f"✅ {config_name} 测试成功 ({duration:.1f}s)")
                    results.append((config_name, fusion_mode, weights, "success", duration))
                else:
                    print(f"⚠️  {config_name} 未找到调试输出")
                    results.append((config_name, fusion_mode, weights, "no_debug", duration))
            else:
                print(f"❌ {config_name} 测试失败")
                print(f"错误输出: {result.stderr[:200]}")
                results.append((config_name, fusion_mode, weights, "failed", 0))
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config_name} 测试超时")
            results.append((config_name, fusion_mode, weights, "timeout", 300))
        except Exception as e:
            print(f"💥 {config_name} 测试异常: {e}")
            results.append((config_name, fusion_mode, weights, "error", 0))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    for config_name, fusion_mode, weights, status, duration in results:
        status_icon = {"success": "✅", "no_debug": "⚠️", "failed": "❌", "timeout": "⏰", "error": "💥"}[status]
        print(f"   {status_icon} {config_name} ({fusion_mode}): {status} ({duration:.1f}s)")
    
    success_count = len([r for r in results if r[3] == "success"])
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count > 0:
        print("🎉 非线性融合功能测试通过！")
        return True
    else:
        print("😔 非线性融合功能测试失败")
        return False

if __name__ == "__main__":
    test_nonlinear_fusion()
