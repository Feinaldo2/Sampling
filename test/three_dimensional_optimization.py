#!/usr/bin/env python3
"""
🔬 三维特征优化实验
对最优attention方式(attention_entropy)进行三维调参
维度: confidence(固定1.0) + entropy + attention_entropy
"""

import subprocess
import time
import csv
import os
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

class ThreeDimensionalOptimization:
    def __init__(self, sample_size=150):
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        
        # 已知的最佳结果作为基准
        self.current_best_score = 0.770  # attention_entropy_neg0.04_200
        self.current_best_weights = "1.00|0.00|-0.04"
        self.pmass_best_score = 0.755
        self.baseline_score = 0.66
        
        # 动态GPU分配
        self.gpu_queue = queue.Queue()
        for i in range(8):
            self.gpu_queue.put(i)
        
        # 创建结果目录
        os.makedirs("three_dimensional_optimization", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "three_dimensional_optimization/results.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'attention_type', 'fusion_mode', 'weights', 'samples', 'score', 'tps', 'duration', 'improvement', 'vs_baseline', 'vs_current_best', 'vs_pmass'])
    
    def get_gpu(self):
        return self.gpu_queue.get()
    
    def release_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)
    
    def get_three_dimensional_configs(self):
        """获取三维调参配置"""
        configs = []
        
        # 三维参数空间 (confidence固定为1.00)
        entropy_weights = [-0.10, -0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06]
        attention_entropy_weights = [-0.15, -0.12, -0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09]
        
        # 基准配置
        configs.append(("baseline", "pmass", "linear", "1.00|0.00|0.00"))
        configs.append(("current_best", "attention_entropy", "linear", "1.00|0.00|-0.04"))
        configs.append(("pmass_best", "pmass", "linear", "1.00|0.00|-0.02"))
        
        # 生成所有三维组合
        for entropy_w in entropy_weights:
            for attention_w in attention_entropy_weights:
                # 跳过当前已知最优配置
                if entropy_w == 0.00 and attention_w == -0.04:
                    continue
                
                config_name = f"3d_e{entropy_w:+.2f}_a{attention_w:+.2f}"
                weight_str = f"1.00|{entropy_w:.2f}|{attention_w:.2f}"
                configs.append((config_name, "attention_entropy", "linear", weight_str))
        
        return configs
    
    def run_single_test(self, config_name, attention_type, fusion_mode, weights):
        """运行单个测试"""
        gpu_id = self.get_gpu()
        
        try:
            with self.lock:
                print(f"🚀 GPU{gpu_id} 启动: {config_name} ({attention_type}: {weights}) - {self.sample_size}样本")
            
            # 创建测试脚本
            script_content = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu_id}
cd /home/zhaoyifei/Sampling/slow-fast-sampling

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate slow_fast_sampling

# 设置attention类型环境变量
export ATTENTION_TYPE={attention_type}

python3 evaluation_script.py --model dream \\
  --model_args "pretrained=/home/zhaoyifei/Sampling/Models/Dream-base,max_new_tokens=256,diffusion_steps=256,temperature=0.2,top_p=0.95,alg=entropy,alg_temp=0.0,add_bos_token=true,is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight={weights},fusion_mode={fusion_mode},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \\
  --tasks gsm8k \\
  --num_fewshot 8 \\
  --batch_size 1 \\
  --limit {self.sample_size} \\
  --output_path ./three_dimensional_optimization/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            script_path = f"/tmp/3d_opt_gpu{gpu_id}_{config_name}_{int(time.time())}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            start_time = time.time()
            result = subprocess.run(['bash', script_path], capture_output=True, text=True)  # 不限时间
            end_time = time.time()
            duration = int(end_time - start_time)
            
            os.remove(script_path)
            
            if result.returncode == 0:
                score = self.extract_score(result.stdout)
                tps = self.extract_tps(result.stdout)
                
                if score is not None:
                    improvement = (score - self.baseline_score) / self.baseline_score * 100
                    vs_baseline = score - self.baseline_score
                    vs_current_best = score - self.current_best_score
                    vs_pmass = score - self.pmass_best_score
                    
                    with self.lock:
                        result_data = {
                            'gpu_id': gpu_id,
                            'config_name': config_name,
                            'attention_type': attention_type,
                            'fusion_mode': fusion_mode,
                            'weights': weights,
                            'score': score,
                            'tps': tps,
                            'duration': duration,
                            'improvement': improvement,
                            'vs_baseline': vs_baseline,
                            'vs_current_best': vs_current_best,
                            'vs_pmass': vs_pmass
                        }
                        
                        self.results.append(result_data)
                        
                        # 写入CSV
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                gpu_id, config_name, attention_type, fusion_mode, weights, 
                                self.sample_size, score, tps, duration, improvement, vs_baseline, vs_current_best, vs_pmass
                            ])
                    
                    if vs_current_best > 0:
                        print(f"🎉 GPU{gpu_id} 新纪录: {config_name} - {score:.4f} (超越当前最佳 {vs_current_best:.4f})")
                    elif vs_pmass > 0:
                        print(f"✅ GPU{gpu_id} 超越pmass: {config_name} - {score:.4f} (超越pmass {vs_pmass:.4f})")
                    elif vs_baseline > 0:
                        print(f"📊 GPU{gpu_id} 超越baseline: {config_name} - {score:.4f} (+{improvement:.2f}%)")
                    else:
                        print(f"📊 GPU{gpu_id} 完成: {config_name} - {score:.4f} ({improvement:.2f}%)")
                    
                    return True
                else:
                    print(f"⚠️  GPU{gpu_id} 解析失败: {config_name}")
                    return False
            else:
                print(f"❌ GPU{gpu_id} 失败: {config_name}")
                return False
                
        except Exception as e:
            print(f"💥 GPU{gpu_id} 异常: {config_name} - {e}")
            return False
        finally:
            self.release_gpu(gpu_id)
    
    def extract_score(self, output):
        """提取分数"""
        try:
            for line in output.split('\n'):
                if 'strict-match' in line and 'exact_match' in line:
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        try:
                            score = float(part)
                            if 0 <= score <= 1:
                                return score
                        except:
                            continue
            return None
        except:
            return None
    
    def extract_tps(self, output):
        """提取TPS"""
        try:
            for line in output.split('\n'):
                if 'TPS' in line and 'Chars/Sec' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            tps = float(parts[-1].strip())
                            return tps
                        except:
                            pass
            return 0.0
        except:
            return 0.0
    
    def run_three_dimensional_optimization(self):
        """运行三维优化实验"""
        print("🔬 三维特征优化实验开始")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 样本数量: {self.sample_size}")
        print(f"🎯 目标: 在三维空间中找到最优权重组合")
        print(f"📈 当前最佳: {self.current_best_score:.4f} ({self.current_best_weights})")
        print("=" * 80)
        
        configs = self.get_three_dimensional_configs()
        
        print(f"📊 三维搜索配置:")
        print(f"   Confidence权重: 固定1.00")
        print(f"   Entropy权重: 9个值 [-0.10 到 +0.06]")
        print(f"   Attention_entropy权重: 9个值 [-0.15 到 +0.09]")
        print(f"   总配置数: {len(configs)} (包含3个基准)")
        print(f"⏰ 预计时间: {len(configs) * 25 // 8} 分钟 (8个GPU并行)")
        print("=" * 80)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for config_name, attention_type, fusion_mode, weights in configs:
                future = executor.submit(self.run_single_test, config_name, attention_type, fusion_mode, weights)
                futures.append((future, config_name))
            
            completed = 0
            total = len(futures)
            
            for future in as_completed([f[0] for f in futures]):
                completed += 1
                # 找到对应的config_name
                config_name = None
                for f, name in futures:
                    if f == future:
                        config_name = name
                        break
                
                try:
                    future.result()
                    print(f"📊 三维优化进度: {completed}/{total} - 完成 {config_name}")
                except Exception as e:
                    print(f"💥 {config_name} 异常: {e}")
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 100)
        print("🏁 三维特征优化实验完成！")
        print("=" * 100)
        
        if not self.results:
            print("😔 没有成功的结果")
            return
        
        # 按分数排序
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 三维优化结果 ({self.sample_size}样本):")
        print(f"{'排名':<4} {'配置名称':<25} {'权重配置':<15} {'分数':<8} {'vs 当前最佳':<12} {'vs pmass':<10}")
        print("-" * 85)
        
        for i, result in enumerate(self.results[:20], 1):
            score = result['score']
            vs_current_best = result['vs_current_best']
            vs_pmass = result['vs_pmass']
            
            if vs_current_best > 0:
                mark = "🎉"
                vs_current_str = f"+{vs_current_best:.4f}"
            elif vs_pmass > 0:
                mark = "✅"
                vs_current_str = f"{vs_current_best:.4f}"
            elif result['vs_baseline'] > 0:
                mark = "📊"
                vs_current_str = f"{vs_current_best:.4f}"
            else:
                mark = "  "
                vs_current_str = f"{vs_current_best:.4f}"
            
            vs_pmass_str = f"{vs_pmass:+.4f}"
            
            print(f"{mark}{i:<3} {result['config_name']:<25} {result['weights']:<15} {score:.4f} {vs_current_str:<12} {vs_pmass_str:<10}")
        
        # 显示最终最优配置
        if self.results:
            best = self.results[0]
            print(f"\n🏆 最终最优配置 ({self.sample_size}样本验证):")
            print(f"   配置名称: {best['config_name']}")
            print(f"   权重配置: {best['weights']}")
            print(f"   分数: {best['score']:.4f}")
            print(f"   超越当前最佳: {best['vs_current_best']:+.4f}")
            print(f"   超越pmass: {best['vs_pmass']:+.4f}")
            print(f"   超越baseline: {best['vs_baseline']:+.4f}")
            
            # 分析权重组合
            weight_parts = best['weights'].split('|')
            if len(weight_parts) == 3:
                conf_w, entropy_w, att_w = weight_parts
                print(f"\n🔍 最优权重分析:")
                print(f"   Confidence权重: {conf_w} (固定)")
                print(f"   Entropy权重: {entropy_w}")
                print(f"   Attention_entropy权重: {att_w}")
                
                if float(entropy_w) < 0:
                    print(f"   ✅ Entropy负权重: 惩罚不确定性")
                if float(att_w) < 0:
                    print(f"   ✅ Attention_entropy负权重: 惩罚注意力分散")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='三维特征优化实验')
    parser.add_argument('--samples', type=int, default=150, help='样本数量')
    
    args = parser.parse_args()
    
    experiment = ThreeDimensionalOptimization(sample_size=args.samples)
    experiment.run_three_dimensional_optimization()
