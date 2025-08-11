#!/usr/bin/env python3
"""
🔬 特征值分析实验
分析confidence和各种attention特征的数值分布
基于分布确定合理的权重范围
"""

import subprocess
import time
import csv
import os
import json
import numpy as np
from datetime import datetime
import threading
import queue
import re

class FeatureAnalysisExperiment:
    def __init__(self, sample_size=20):
        self.sample_size = sample_size  # 用较少样本快速分析
        self.results = []
        self.lock = threading.Lock()
        
        # 动态GPU分配
        self.gpu_queue = queue.Queue()
        for i in range(8):
            self.gpu_queue.put(i)
        
        # 创建结果目录
        os.makedirs("feature_analysis_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "feature_analysis_results/feature_stats.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['attention_type', 'confidence_mean', 'confidence_std', 'attention_mean', 'attention_std', 'confidence_range', 'attention_range', 'suggested_weight_range'])
    
    def get_gpu(self):
        return self.gpu_queue.get()
    
    def release_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)
    
    def get_analysis_configs(self):
        """获取分析配置 - 每种attention类型用零权重测试"""
        configs = []
        
        attention_types = [
            'pmass',
            'attention_entropy',
            'max_attention', 
            'self_attention',
            'attention_variance',
            'k_direction'
        ]
        
        # 每种attention类型都用零权重测试，获取原始特征值分布
        for attention_type in attention_types:
            config_name = f"{attention_type}_analysis"
            weights = "1.00|0.00|0.00"  # 零权重，只获取特征值
            configs.append((config_name, attention_type, "linear", weights))
        
        return configs
    
    def run_single_analysis(self, config_name, attention_type, fusion_mode, weights):
        """运行单个分析测试"""
        gpu_id = self.get_gpu()
        
        try:
            with self.lock:
                print(f"🔍 GPU{gpu_id} 分析: {config_name} ({attention_type}) - {self.sample_size}样本")
            
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
  --output_path ./feature_analysis_results/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            script_path = f"/tmp/feature_analysis_gpu{gpu_id}_{config_name}_{int(time.time())}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            start_time = time.time()
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=1200)
            end_time = time.time()
            duration = int(end_time - start_time)
            
            os.remove(script_path)
            
            if result.returncode == 0:
                # 解析融合调试信息，提取特征值
                feature_stats = self.extract_feature_stats(result.stdout, attention_type)
                
                if feature_stats:
                    with self.lock:
                        self.results.append({
                            'attention_type': attention_type,
                            'feature_stats': feature_stats,
                            'duration': duration
                        })
                        
                        print(f"✅ GPU{gpu_id} 完成分析: {config_name}")
                        print(f"   Confidence: 均值={feature_stats['confidence_mean']:.4f}, 标准差={feature_stats['confidence_std']:.4f}")
                        print(f"   Attention: 均值={feature_stats['attention_mean']:.4f}, 标准差={feature_stats['attention_std']:.4f}")
                    
                    return True
                else:
                    print(f"⚠️  GPU{gpu_id} 特征解析失败: {config_name}")
                    return False
            else:
                print(f"❌ GPU{gpu_id} 失败: {config_name}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ GPU{gpu_id} 超时: {config_name}")
            return False
        except Exception as e:
            print(f"💥 GPU{gpu_id} 异常: {config_name} - {e}")
            return False
        finally:
            self.release_gpu(gpu_id)
    
    def extract_feature_stats(self, output, attention_type):
        """从输出中提取特征统计信息"""
        try:
            confidence_values = []
            attention_values = []

            # 先打印输出用于调试
            print(f"🔍 调试输出 ({attention_type}):")
            lines = output.split('\n')
            debug_lines = [line for line in lines if ('融合调试' in line or 'conf=' in line or 'entropy=' in line or 'pmass=' in line)]
            for line in debug_lines[:5]:  # 只显示前5行
                print(f"   {line}")

            # 解析融合调试信息
            for i, line in enumerate(lines):
                if '🔬 AdLLM融合调试' in line and 'linear' in line:
                    # 查找下一行的特征值
                    if i + 1 < len(lines):
                        feature_line = lines[i + 1]
                        print(f"   特征行: {feature_line}")

                        # 解析confidence值
                        conf_match = re.search(r'conf=([0-9.]+)', feature_line)
                        if conf_match:
                            confidence_values.append(float(conf_match.group(1)))

                        # 解析attention特征值 (根据attention类型确定特征名)
                        if attention_type == 'pmass':
                            att_match = re.search(r'pmass=([0-9.]+)', feature_line)
                        elif attention_type == 'attention_entropy':
                            att_match = re.search(r'entropy=([0-9.]+)', feature_line)
                        else:
                            # 对于其他类型，尝试找到第三个特征值
                            parts = feature_line.split(',')
                            if len(parts) >= 3:
                                third_part = parts[2].strip()
                                att_match = re.search(r'=([0-9.]+)', third_part)
                            else:
                                att_match = None

                        if att_match:
                            attention_values.append(float(att_match.group(1)))
                            print(f"   提取到attention值: {att_match.group(1)}")
                        else:
                            print(f"   未找到attention值")
            
            if confidence_values and attention_values:
                confidence_mean = np.mean(confidence_values)
                confidence_std = np.std(confidence_values)
                attention_mean = np.mean(attention_values)
                attention_std = np.std(attention_values)
                
                confidence_range = f"[{min(confidence_values):.4f}, {max(confidence_values):.4f}]"
                attention_range = f"[{min(attention_values):.4f}, {max(attention_values):.4f}]"
                
                # 基于特征值分布建议权重范围
                # 权重应该使得 attention_weight * attention_feature 与 confidence 在同一数量级
                if attention_mean > 0:
                    # 建议权重范围：使attention项的贡献在confidence的±50%范围内
                    max_weight = (confidence_mean * 0.5) / attention_mean
                    min_weight = -(confidence_mean * 0.5) / attention_mean
                    suggested_range = f"[{min_weight:.2f}, {max_weight:.2f}]"
                else:
                    suggested_range = "[-0.10, 0.10]"  # 默认范围
                
                return {
                    'confidence_mean': confidence_mean,
                    'confidence_std': confidence_std,
                    'attention_mean': attention_mean,
                    'attention_std': attention_std,
                    'confidence_range': confidence_range,
                    'attention_range': attention_range,
                    'suggested_weight_range': suggested_range,
                    'confidence_values': confidence_values,
                    'attention_values': attention_values
                }
            
            return None
            
        except Exception as e:
            print(f"特征解析异常: {e}")
            return None
    
    def run_feature_analysis(self):
        """运行特征分析实验"""
        print("🔬 特征值分析实验开始")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 样本数量: {self.sample_size} (快速分析)")
        print(f"🎯 目标: 分析各attention特征的数值分布，确定合理权重范围")
        print("=" * 80)
        
        configs = self.get_analysis_configs()
        
        print(f"📊 分析配置数量: {len(configs)}")
        print(f"⏰ 预计时间: {len(configs) * 10 // 8} 分钟 (8个GPU并行)")
        print("=" * 80)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for config_name, attention_type, fusion_mode, weights in configs:
                future = executor.submit(self.run_single_analysis, config_name, attention_type, fusion_mode, weights)
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
                    print(f"📊 分析进度: {completed}/{total} - 完成 {config_name}")
                except Exception as e:
                    print(f"💥 {config_name} 异常: {e}")
        
        # 生成分析报告
        self.generate_analysis_report()
    
    def generate_analysis_report(self):
        """生成特征分析报告"""
        print("\n" + "=" * 100)
        print("🏁 特征值分析实验完成！")
        print("=" * 100)
        
        if not self.results:
            print("😔 没有分析结果")
            return
        
        print(f"📊 特征值分布分析:")
        print(f"{'Attention类型':<18} {'Confidence均值':<12} {'Attention均值':<12} {'建议权重范围':<20}")
        print("-" * 70)
        
        # 写入CSV并显示结果
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for result in self.results:
                att_type = result['attention_type']
                stats = result['feature_stats']
                
                print(f"{att_type:<18} {stats['confidence_mean']:<12.4f} {stats['attention_mean']:<12.4f} {stats['suggested_weight_range']:<20}")
                
                writer.writerow([
                    att_type,
                    stats['confidence_mean'],
                    stats['confidence_std'],
                    stats['attention_mean'],
                    stats['attention_std'],
                    stats['confidence_range'],
                    stats['attention_range'],
                    stats['suggested_weight_range']
                ])
        
        print(f"\n🔍 详细分析:")
        for result in self.results:
            att_type = result['attention_type']
            stats = result['feature_stats']
            
            print(f"\n{att_type}:")
            print(f"   Confidence: 均值={stats['confidence_mean']:.4f}, 标准差={stats['confidence_std']:.4f}, 范围={stats['confidence_range']}")
            print(f"   Attention:  均值={stats['attention_mean']:.4f}, 标准差={stats['attention_std']:.4f}, 范围={stats['attention_range']}")
            print(f"   建议权重范围: {stats['suggested_weight_range']}")
        
        print(f"\n💡 基于分析结果的建议:")
        print(f"   1. 权重范围应该基于特征值的实际分布")
        print(f"   2. 权重 × attention特征值 应与 confidence 在同一数量级")
        print(f"   3. 建议的权重范围已保存到 {self.csv_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='特征值分析实验')
    parser.add_argument('--samples', type=int, default=20, help='分析样本数量')
    
    args = parser.parse_args()
    
    experiment = FeatureAnalysisExperiment(sample_size=args.samples)
    experiment.run_feature_analysis()
