#!/usr/bin/env python3
"""
🔬 聚焦Attention搜索实验
基于之前结果，聚焦于有希望的attention方式和权重范围
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

class FocusedAttentionSearch:
    def __init__(self, sample_size=50):
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        self.baseline_score = 0.7333
        self.target_score = 0.74  # 需要超越的分数
        
        # 动态GPU分配
        self.gpu_queue = queue.Queue()
        for i in range(8):
            self.gpu_queue.put(i)
        
        # 创建结果目录
        os.makedirs("focused_attention_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "focused_attention_results/results.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'attention_type', 'fusion_mode', 'weights', 'samples', 'score', 'tps', 'duration', 'improvement', 'vs_target', 'phase'])
    
    def get_gpu(self):
        return self.gpu_queue.get()
    
    def release_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)
    
    def generate_focused_configs(self):
        """基于之前结果生成聚焦的配置"""
        configs = []
        
        # 基线对照
        configs.append(("baseline", "pmass", "linear", "1.00|0.00|0.00"))
        configs.append(("entropy_best", "pmass", "linear", "1.00|-0.05|0.00"))
        configs.append(("pmass_best", "pmass", "linear", "1.00|0.00|-0.02"))
        
        # 1. 新的attention方式：k_direction和q_direction_prompt
        # 基于pmass最佳权重(-0.02)的经验，测试相似范围
        new_attention_weights = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        
        for att_type in ['k_direction', 'q_direction_prompt']:
            for w in new_attention_weights:
                configs.append((f"{att_type}_{abs(w):.2f}{'_neg' if w < 0 else ''}", 
                              att_type, "linear", f"1.00|0.00|{w:.2f}"))
        
        # 2. 已知表现好的attention方式的精细化搜索
        # attention_entropy和relative_pmass在-0.02时达到0.74
        good_attention_types = ['attention_entropy', 'relative_pmass']
        refined_weights = [-0.04, -0.03, -0.025, -0.02, -0.015, -0.01]
        
        for att_type in good_attention_types:
            for w in refined_weights:
                configs.append((f"{att_type}_refined_{abs(w):.3f}", 
                              att_type, "linear", f"1.00|0.00|{w:.2f}"))
        
        # 3. 三元组合：最有希望的attention + entropy组合
        # 基于已知entropy最佳范围[-0.02, -0.07]
        entropy_weights = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
        promising_combinations = [
            ('pmass', [-0.04, -0.03, -0.025, -0.02, -0.015, -0.01, 0.01]),  # 🔥 最重要：pmass权重也变化
            ('k_direction', [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]),
            ('q_direction_prompt', [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]),
            ('attention_entropy', [-0.03, -0.02, -0.01]),
            ('relative_pmass', [-0.03, -0.02, -0.01]),
        ]
        
        for att_type, att_weights in promising_combinations:
            for e_w in entropy_weights:
                for a_w in att_weights:
                    configs.append((f"{att_type}_combo_e{abs(e_w):.2f}_a{abs(a_w):.2f}{'_neg' if a_w < 0 else ''}", 
                                  att_type, "linear", f"1.00|{e_w:.2f}|{a_w:.2f}"))
        
        return configs
    
    def generate_best_refinement_configs(self):
        """基于第一阶段最佳结果的进一步精细化"""
        configs = []
        
        # 读取第一阶段最佳结果
        best_results = self.get_current_best_results()
        
        for result in best_results[:3]:  # 对前3名进行精细化
            att_type = result['attention_type']
            weights = result['weights'].split('|')
            ent_w = float(weights[1])
            att_w = float(weights[2])
            
            # 在最佳结果周围进行微调 (0.005步长)
            ent_deltas = [-0.01, -0.005, 0.005, 0.01] if ent_w != 0 else [0]
            att_deltas = [-0.01, -0.005, 0.005, 0.01] if att_w != 0 else [0]
            
            for e_delta in ent_deltas:
                for a_delta in att_deltas:
                    if e_delta == 0 and a_delta == 0:
                        continue
                    
                    new_e_w = round(ent_w + e_delta, 3)
                    new_a_w = round(att_w + a_delta, 3)
                    
                    # 确保在合理范围内
                    if -0.15 <= new_e_w <= 0 and -0.10 <= new_a_w <= 0.10:
                        configs.append((f"{att_type}_micro_e{abs(new_e_w):.3f}_a{abs(new_a_w):.3f}{'_neg' if new_a_w < 0 else ''}", 
                                      att_type, "linear", f"1.00|{new_e_w:.2f}|{new_a_w:.2f}"))
        
        return configs[:30]  # 限制数量
    
    def get_current_best_results(self):
        """获取当前最佳结果"""
        results = []
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题
                for row in reader:
                    if len(row) >= 13:
                        results.append({
                            'config_name': row[2],
                            'attention_type': row[3],
                            'weights': row[5],
                            'score': float(row[7]),
                            'vs_target': float(row[11])
                        })
        except:
            return []
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def run_single_test(self, config_name, attention_type, fusion_mode, weights, phase):
        """运行单个测试"""
        gpu_id = self.get_gpu()
        
        try:
            with self.lock:
                print(f"🚀 GPU{gpu_id} 启动: {config_name} ({attention_type}: {weights})")
            
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
  --output_path ./focused_attention_results/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            script_path = f"/tmp/focused_attn_gpu{gpu_id}_{config_name}_{int(time.time())}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            start_time = time.time()
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=1800)
            end_time = time.time()
            duration = int(end_time - start_time)
            
            os.remove(script_path)
            
            if result.returncode == 0:
                score = self.extract_score(result.stdout)
                tps = self.extract_tps(result.stdout)
                
                if score is not None:
                    improvement = (score - self.baseline_score) / self.baseline_score * 100
                    vs_target = score - self.target_score
                    
                    with self.lock:
                        self.results.append({
                            'gpu_id': gpu_id,
                            'config_name': config_name,
                            'attention_type': attention_type,
                            'fusion_mode': fusion_mode,
                            'weights': weights,
                            'score': score,
                            'tps': tps,
                            'duration': duration,
                            'improvement': improvement,
                            'vs_target': vs_target,
                            'phase': phase
                        })
                        
                        # 写入CSV
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                gpu_id, config_name, attention_type, fusion_mode, weights, 
                                self.sample_size, score, tps, duration, improvement, vs_target, phase
                            ])
                    
                    if vs_target > 0:
                        print(f"🎉 GPU{gpu_id} 超越目标: {config_name} - {score:.4f} (超越{vs_target:.4f})")
                    elif improvement > 0:
                        print(f"✅ GPU{gpu_id} 超越baseline: {config_name} - {score:.4f} (+{improvement:.2f}%)")
                    else:
                        print(f"📊 GPU{gpu_id} 完成: {config_name} - {score:.4f} ({improvement:.2f}%)")
                    
                    return True
                else:
                    print(f"⚠️  GPU{gpu_id} 解析失败: {config_name}")
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
    
    def run_phase(self, phase_name, configs, max_parallel=8):
        """运行一个阶段的实验"""
        if not configs:
            print(f"⏭️  跳过{phase_name}：无配置")
            return True
            
        print(f"\n🚀 开始{phase_name}")
        print(f"📊 配置数量: {len(configs)}")
        print(f"🎯 目标: 超越分数 {self.target_score:.4f}")
        print("=" * 80)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = []
            for config_name, attention_type, fusion_mode, weights in configs:
                future = executor.submit(self.run_single_test, config_name, attention_type, fusion_mode, weights, phase_name)
                futures.append(future)
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                completed += 1
                try:
                    future.result()
                    print(f"📊 {phase_name} 进度: {completed}/{total}")
                except Exception as e:
                    print(f"💥 任务异常: {e}")
        
        return True
    
    def run_focused_search(self):
        """运行聚焦搜索实验"""
        print("🔬 聚焦Attention搜索实验开始")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 样本数量: {self.sample_size}")
        print(f"🎯 目标: 超越分数 {self.target_score:.4f}")
        print(f"⚙️  权重格式: 2位小数，Confidence=1.00")
        print("=" * 80)
        
        # 阶段1：聚焦搜索
        configs = self.generate_focused_configs()
        self.run_phase("阶段1：聚焦搜索", configs)
        
        # 阶段2：最佳结果精细化
        refined_configs = self.generate_best_refinement_configs()
        if refined_configs:
            self.run_phase("阶段2：最佳结果精细化", refined_configs)
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 100)
        print("🏁 聚焦Attention搜索实验完成！")
        print("=" * 100)
        
        if not self.results:
            print("😔 没有成功的结果")
            return
        
        # 按分数排序
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        # 超越目标的配置
        better_than_target = [r for r in self.results if r['vs_target'] > 0]
        
        print(f"📊 实验统计:")
        print(f"   总配置数: {len(self.results)}")
        print(f"   超越目标: {len(better_than_target)}")
        
        # 显示前15名
        print(f"\n🏆 前15名结果:")
        print(f"{'排名':<4} {'配置名称':<35} {'Attention类型':<20} {'权重配置':<15} {'分数':<8} {'vs目标':<8}")
        print("-" * 100)
        
        for i, result in enumerate(self.results[:15], 1):
            score = result['score']
            vs_target = result['vs_target']
            
            if vs_target > 0:
                mark = "🎉"
                vs_target_str = f"+{vs_target:.4f}"
            else:
                mark = "  "
                vs_target_str = f"{vs_target:.4f}"
            
            print(f"{mark}{i:<3} {result['config_name']:<35} {result['attention_type']:<20} {result['weights']:<15} {score:.4f} {vs_target_str:<8}")
        
        # 最佳配置详情
        if better_than_target:
            best = better_than_target[0]
            print(f"\n🎉 发现超越目标的配置!")
            print(f"   最佳配置: {best['config_name']}")
            print(f"   Attention类型: {best['attention_type']}")
            print(f"   权重配置: {best['weights']}")
            print(f"   分数: {best['score']:.4f}")
            print(f"   超越目标: +{best['vs_target']:.4f}")
        else:
            print(f"\n📊 未找到超越目标的配置")
            if self.results:
                best = self.results[0]
                print(f"   最佳配置: {best['config_name']} - {best['score']:.4f}")
                print(f"   与目标差距: {best['vs_target']:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='聚焦Attention搜索实验')
    parser.add_argument('--samples', type=int, default=50, help='每个配置的样本数量')
    
    args = parser.parse_args()
    
    experiment = FocusedAttentionSearch(sample_size=args.samples)
    experiment.run_focused_search()
