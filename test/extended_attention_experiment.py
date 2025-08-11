#!/usr/bin/env python3
"""
🔬 扩展Attention实验
1. 测试新的attention计算方式：k_direction, q_direction_prompt
2. 基于阶段1结果进行进一步探索
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

class ExtendedAttentionExperiment:
    def __init__(self, sample_size=50):
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        self.baseline_score = 0.7333
        self.pmass_best_score = 0.74  # pmass的最佳分数
        
        # 动态GPU分配
        self.gpu_queue = queue.Queue()
        for i in range(8):
            self.gpu_queue.put(i)
        
        # 创建结果目录
        os.makedirs("extended_attention_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "extended_attention_results/results.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'attention_type', 'fusion_mode', 'weights', 'samples', 'score', 'tps', 'duration', 'improvement', 'vs_pmass_best', 'phase'])
    
    def get_gpu(self):
        return self.gpu_queue.get()
    
    def release_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)
    
    def get_new_attention_configs(self):
        """获取新的attention计算方式配置"""
        configs = []
        
        # 基线对照
        configs.append(("baseline", "pmass", "linear", "1.0|0.0|0.0"))
        configs.append(("pmass_best", "pmass", "linear", "1.0|0.0|-0.02"))
        configs.append(("entropy_best", "pmass", "linear", "1.0|-0.05|0.0"))
        
        # 1. k_direction: 被关注程度 (权重限制为2位小数)
        k_direction_weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20,
                              -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.12, -0.15]
        for w in k_direction_weights:
            configs.append((f"k_dir_{abs(w):.2f}{'_neg' if w < 0 else ''}", "k_direction", "linear", f"1.00|0.00|{w:.2f}"))

        # 2. q_direction_prompt: 对prompt的查询强度 (权重限制为2位小数)
        q_direction_weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20,
                              -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.12, -0.15]
        for w in q_direction_weights:
            configs.append((f"q_prompt_{abs(w):.2f}{'_neg' if w < 0 else ''}", "q_direction_prompt", "linear", f"1.00|0.00|{w:.2f}"))
        
        return configs
    
    def get_refined_exploration_configs(self):
        """基于阶段1结果的精细化探索配置"""
        configs = []
        
        # 这里会根据当前实验的阶段1结果动态生成
        # 暂时先定义一些基于经验的配置
        
        # 基于pmass最佳结果(-0.02)的邻域搜索 (2位小数)
        pmass_refined_weights = [-0.01, -0.02, -0.03, -0.04]
        for w in pmass_refined_weights:
            configs.append((f"pmass_refined_{abs(w):.2f}", "pmass", "linear", f"1.00|0.00|{w:.2f}"))

        # 基于entropy最佳结果(-0.05)的邻域搜索 (2位小数)
        entropy_refined_weights = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]
        for w in entropy_refined_weights:
            configs.append((f"entropy_refined_{abs(w):.2f}", "pmass", "linear", f"1.00|{w:.2f}|0.00"))

        # 三元组合搜索 (confidence + entropy + attention) - 2位小数
        triple_configs = [
            ("triple_1", "pmass", "linear", "1.00|-0.03|-0.01"),
            ("triple_2", "pmass", "linear", "1.00|-0.04|-0.02"),
            ("triple_3", "pmass", "linear", "1.00|-0.06|-0.02"),
            ("triple_4", "k_direction", "linear", "1.00|-0.05|0.05"),
            ("triple_5", "q_direction_prompt", "linear", "1.00|-0.05|0.05"),
            ("triple_6", "pmass", "linear", "1.00|-0.02|-0.01"),
            ("triple_7", "pmass", "linear", "1.00|-0.07|-0.03"),
        ]
        configs.extend(triple_configs)
        
        return configs
    
    def analyze_stage1_results(self):
        """分析当前实验的阶段1结果，生成针对性的探索配置"""
        # 读取当前实验结果
        stage1_results = []
        try:
            with open("alternative_attention_results/results.csv", 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题
                for row in reader:
                    if len(row) >= 13 and row[12] == "阶段1：替代Attention特征搜索":
                        stage1_results.append({
                            'config_name': row[2],
                            'attention_type': row[3],
                            'weights': row[5],
                            'score': float(row[7]),
                            'vs_pmass_best': float(row[11])
                        })
        except:
            print("⚠️  无法读取阶段1结果，使用默认配置")
            return []
        
        if not stage1_results:
            return []
        
        # 按分数排序，找出最佳的attention类型
        stage1_results.sort(key=lambda x: x['score'], reverse=True)
        
        configs = []
        print(f"\n📊 基于阶段1结果生成精细化配置:")
        
        # 对每种attention类型，找出最佳权重并进行邻域搜索
        attention_types = {}
        for result in stage1_results:
            att_type = result['attention_type']
            if att_type not in attention_types:
                attention_types[att_type] = []
            attention_types[att_type].append(result)
        
        for att_type, results in attention_types.items():
            if att_type == 'pmass':  # 跳过pmass，已经充分测试
                continue
                
            best_result = max(results, key=lambda x: x['score'])
            if best_result['score'] > 0.70:  # 只对表现较好的进行精细化
                best_weight = float(best_result['weights'].split('|')[2])
                print(f"   {att_type}: 最佳权重 {best_weight}, 分数 {best_result['score']:.4f}")
                
                # 邻域搜索 (确保2位小数)
                if best_weight > 0:
                    neighbors = [round(best_weight * 0.8, 2), round(best_weight * 0.9, 2),
                               round(best_weight * 1.1, 2), round(best_weight * 1.2, 2)]
                else:
                    neighbors = [round(best_weight * 1.2, 2), round(best_weight * 1.1, 2),
                               round(best_weight * 0.9, 2), round(best_weight * 0.8, 2)]

                for w in neighbors:
                    configs.append((f"{att_type}_refined_{abs(w):.2f}{'_neg' if w < 0 else ''}",
                                  att_type, "linear", f"1.00|0.00|{w:.2f}"))

                # 与entropy组合 (确保2位小数)
                entropy_weights = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
                for e_w in entropy_weights:
                    configs.append((f"{att_type}_entropy_combo_{abs(e_w):.2f}",
                                  att_type, "linear", f"1.00|{e_w:.2f}|{best_weight:.2f}"))
        
        return configs[:20]  # 限制数量
    
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
  --output_path ./extended_attention_results/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            script_path = f"/tmp/ext_attn_gpu{gpu_id}_{config_name}_{int(time.time())}.sh"
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
                    vs_pmass_best = score - self.pmass_best_score
                    
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
                            'vs_pmass_best': vs_pmass_best,
                            'phase': phase
                        })
                        
                        # 写入CSV
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                gpu_id, config_name, attention_type, fusion_mode, weights, 
                                self.sample_size, score, tps, duration, improvement, vs_pmass_best, phase
                            ])
                    
                    if vs_pmass_best > 0:
                        print(f"🎉 GPU{gpu_id} 超越pmass: {config_name} - {score:.4f} (超越{vs_pmass_best:.4f})")
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
        print(f"🎯 目标: 超越pmass最佳分数 {self.pmass_best_score:.4f}")
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
    
    def run_extended_experiment(self):
        """运行扩展attention实验"""
        print("🔬 扩展Attention实验开始")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 样本数量: {self.sample_size}")
        print(f"🎯 目标: 超越pmass最佳分数 {self.pmass_best_score:.4f}")
        print("=" * 80)
        
        # 阶段1：新的attention计算方式
        configs = self.get_new_attention_configs()
        self.run_phase("阶段1：新Attention方式测试", configs)
        
        # 阶段2：基于现有结果的精细化探索
        refined_configs = self.analyze_stage1_results()
        if refined_configs:
            self.run_phase("阶段2：基于结果的精细化探索", refined_configs)
        
        # 阶段3：进一步的精细化探索
        further_configs = self.get_refined_exploration_configs()
        self.run_phase("阶段3：进一步精细化探索", further_configs)
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终实验报告"""
        print("\n" + "=" * 100)
        print("🏁 扩展Attention实验完成！")
        print("=" * 100)
        
        if not self.results:
            print("😔 没有成功的结果")
            return
        
        # 按分数排序
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        # 超越pmass的配置
        better_than_pmass = [r for r in self.results if r['vs_pmass_best'] > 0]
        
        print(f"📊 实验统计:")
        print(f"   总配置数: {len(self.results)}")
        print(f"   超越pmass: {len(better_than_pmass)}")
        
        # 显示前10名
        print(f"\n🏆 前10名结果:")
        print(f"{'排名':<4} {'配置名称':<30} {'Attention类型':<20} {'权重':<15} {'分数':<8} {'vs Pmass':<10}")
        print("-" * 100)
        
        for i, result in enumerate(self.results[:10], 1):
            score = result['score']
            vs_pmass = result['vs_pmass_best']
            
            if vs_pmass > 0:
                mark = "🎉"
                vs_pmass_str = f"+{vs_pmass:.4f}"
            else:
                mark = "  "
                vs_pmass_str = f"{vs_pmass:.4f}"
            
            print(f"{mark}{i:<3} {result['config_name']:<30} {result['attention_type']:<20} {result['weights']:<15} {score:.4f} {vs_pmass_str:<10}")
        
        if better_than_pmass:
            best = better_than_pmass[0]
            print(f"\n🎉 发现更好的attention计算方式!")
            print(f"   最佳方式: {best['attention_type']}")
            print(f"   最佳配置: {best['config_name']}")
            print(f"   权重: {best['weights']}")
            print(f"   分数: {best['score']:.4f}")
            print(f"   超越pmass: +{best['vs_pmass_best']:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='扩展Attention实验')
    parser.add_argument('--samples', type=int, default=50, help='每个配置的样本数量')
    
    args = parser.parse_args()
    
    experiment = ExtendedAttentionExperiment(sample_size=args.samples)
    experiment.run_extended_experiment()
