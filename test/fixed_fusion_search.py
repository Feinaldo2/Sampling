#!/usr/bin/env python3
"""
🚀 修复版融合策略搜索 - 使用正确的conda激活方式
"""

import subprocess
import time
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class FixedFusionSearch:
    def __init__(self, sample_size=50):
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        self.baseline_score = 0.7333
        
        # 创建结果目录
        os.makedirs("fixed_fusion_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "fixed_fusion_results/results.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'weights', 'samples', 'score', 'tps', 'duration', 'improvement'])
    
    def get_test_configs(self):
        """获取测试配置"""
        return [
            # 基线和基础配置
            ("baseline", "1.0|0.0|0.0"),
            ("conf_15_attn_03", "1.5|0.0|0.3"),
            ("conf_18_attn_04", "1.8|0.0|0.4"),
            ("conf_20_attn_05", "2.0|0.0|0.5"),
            
            # 熵惩罚配置
            ("conf_15_ent_02_attn_04", "1.5|-0.2|0.4"),
            ("conf_18_ent_03_attn_05", "1.8|-0.3|0.5"),
            ("conf_20_ent_04_attn_06", "2.0|-0.4|0.6"),
            
            # 高潜力配置
            ("conf_22_attn_03", "2.2|0.0|0.3"),
            ("conf_25_attn_04", "2.5|0.0|0.4"),
            ("attention_dominant", "1.0|0.0|1.0"),
            
            # 创新配置
            ("innovative_1", "1.6|-0.15|0.45"),
            ("innovative_2", "2.3|0.0|0.35"),
        ]
    
    def run_single_test(self, gpu_id, config_name, weights):
        """在指定GPU上运行单个测试"""
        with self.lock:
            print(f"🚀 GPU{gpu_id} 启动: {config_name} ({weights})")
        
        try:
            # 创建修复版脚本
            script_content = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu_id}
cd /home/zhaoyifei/Sampling/slow-fast-sampling

# 正确激活conda环境
eval "$(conda shell.bash hook)"
conda activate slow_fast_sampling

python3 evaluation_script.py --model dream \\
  --model_args "pretrained=/home/zhaoyifei/Sampling/Models/Dream-base,max_new_tokens=256,diffusion_steps=256,temperature=0.2,top_p=0.95,alg=entropy,alg_temp=0.0,add_bos_token=true,is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight={weights},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \\
  --tasks gsm8k \\
  --num_fewshot 8 \\
  --batch_size 1 \\
  --limit {self.sample_size} \\
  --output_path ./fixed_fusion_results/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            # 写入临时脚本
            script_path = f"/tmp/fixed_test_gpu{gpu_id}_{config_name}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            start_time = time.time()
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=1800)  # 30分钟超时
            end_time = time.time()
            duration = int(end_time - start_time)
            
            # 清理临时脚本
            os.remove(script_path)
            
            if result.returncode == 0:
                score = self.extract_score(result.stdout)
                tps = self.extract_tps(result.stdout)
                
                if score is not None:
                    improvement = (score - self.baseline_score) / self.baseline_score * 100
                    
                    with self.lock:
                        self.results.append({
                            'gpu_id': gpu_id,
                            'config_name': config_name,
                            'weights': weights,
                            'score': score,
                            'tps': tps,
                            'duration': duration,
                            'improvement': improvement
                        })
                        
                        # 写入CSV
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                gpu_id, config_name, weights, self.sample_size,
                                score, tps, duration, improvement
                            ])
                    
                    if improvement > 0:
                        print(f"🎉 GPU{gpu_id} 成功: {config_name} - {score:.4f} (+{improvement:.2f}%)")
                    else:
                        print(f"✅ GPU{gpu_id} 完成: {config_name} - {score:.4f} ({improvement:.2f}%)")
                    
                    return True
                else:
                    print(f"⚠️  GPU{gpu_id} 解析失败: {config_name}")
                    return False
            else:
                print(f"❌ GPU{gpu_id} 失败: {config_name}")
                print(f"错误: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ GPU{gpu_id} 超时: {config_name}")
            return False
        except Exception as e:
            print(f"💥 GPU{gpu_id} 异常: {config_name} - {e}")
            return False
    
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
    
    def run_search(self, max_parallel=4):
        """运行搜索 - 使用较少的并行数避免资源冲突"""
        configs = self.get_test_configs()
        
        print("🚀 修复版AdLLM融合策略搜索")
        print(f"📊 样本数量: {self.sample_size}")
        print(f"🎯 目标: 超越baseline ({self.baseline_score})")
        print(f"🚀 并行GPU数: {max_parallel}")
        print(f"📋 配置数量: {len(configs)}")
        print(f"⏱️  预计时间: {len(configs) * self.sample_size / 10 / max_parallel:.1f}分钟")
        print("=" * 80)
        
        input("按Enter键开始搜索...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_config = {}
            for i, (config_name, weights) in enumerate(configs):
                gpu_id = i % max_parallel
                future = executor.submit(self.run_single_test, gpu_id, config_name, weights)
                future_to_config[future] = (gpu_id, config_name, weights)
            
            completed = 0
            total = len(future_to_config)
            
            for future in as_completed(future_to_config):
                gpu_id, config_name, weights = future_to_config[future]
                completed += 1
                try:
                    future.result()
                    print(f"📊 进度: {completed}/{total}")
                except Exception as e:
                    print(f"💥 GPU{gpu_id} 异常: {config_name} - {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("=" * 80)
        print(f"🏁 搜索完成！总用时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        
        self.generate_report()
    
    def generate_report(self):
        """生成报告"""
        if not self.results:
            print("😔 没有成功的结果")
            return
        
        print("\n📊 修复版融合策略搜索报告")
        print("=" * 100)
        
        # 按分数排序
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"{'排名':<4} {'GPU':<4} {'配置名称':<25} {'权重配置':<15} {'分数':<12} {'提升':<10} {'TPS':<8}")
        print("-" * 100)
        
        for i, result in enumerate(self.results, 1):
            score = result['score']
            improvement = result['improvement']
            tps = result['tps'] or 0
            gpu_id = result['gpu_id']
            
            if improvement > 0:
                mark = "🎉"
                improvement_str = f"+{improvement:.1f}%"
            else:
                mark = "  "
                improvement_str = f"{improvement:.1f}%"
            
            print(f"{mark}{i:<3} GPU{gpu_id:<3} {result['config_name']:<25} {result['weights']:<15} {score:.4f}    {improvement_str:<10} {tps:<8.1f}")
        
        # 统计
        better_than_baseline = [r for r in self.results if r['score'] > self.baseline_score]
        
        print(f"\n📈 统计:")
        print(f"   总测试: {len(self.results)}")
        print(f"   超越baseline: {len(better_than_baseline)}")
        print(f"   成功率: {len(better_than_baseline)/len(self.results)*100:.1f}%")
        
        if better_than_baseline:
            best = better_than_baseline[0]
            print(f"\n🏆 最佳结果:")
            print(f"   配置: {best['config_name']}")
            print(f"   权重: {best['weights']}")
            print(f"   分数: {best['score']:.4f}")
            print(f"   提升: +{best['improvement']:.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='修复版AdLLM融合策略搜索')
    parser.add_argument('--samples', type=int, default=50, help='每个配置的样本数量')
    parser.add_argument('--parallel', type=int, default=4, help='并行GPU数量')
    
    args = parser.parse_args()
    
    searcher = FixedFusionSearch(sample_size=args.samples)
    searcher.run_search(max_parallel=args.parallel)
