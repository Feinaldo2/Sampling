#!/usr/bin/env python3
"""
🚀 修复版AdLLM融合策略20小时全面实验系统
修复GPU分配问题，使用动态GPU分配避免OOM
"""

import subprocess
import time
import csv
import os
import json
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import itertools
import random
import queue
from typing import List, Tuple, Dict

class FixedComprehensive20HExperiment:
    def __init__(self, total_hours=20, sample_size=30):
        self.total_hours = total_hours
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        self.baseline_score = 0.7333
        self.start_time = time.time()
        self.best_score = 0.0
        self.best_config = None
        
        # 动态GPU分配
        self.gpu_queue = queue.Queue()
        for i in range(8):  # 8个GPU
            self.gpu_queue.put(i)
        
        # 创建结果目录
        os.makedirs("fixed_comprehensive_20h_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "fixed_comprehensive_20h_results/results.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'fusion_mode', 'weights', 'samples', 'score', 'tps', 'duration', 'improvement', 'phase'])
        
        # 保存实验状态
        self.state_file = "fixed_comprehensive_20h_results/experiment_state.json"
        self.save_state()
    
    def save_state(self):
        """保存实验状态"""
        state = {
            'start_time': self.start_time,
            'total_hours': self.total_hours,
            'sample_size': self.sample_size,
            'best_score': self.best_score,
            'best_config': self.best_config,
            'completed_configs': len(self.results)
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_remaining_time(self):
        """获取剩余时间（小时）"""
        elapsed = time.time() - self.start_time
        remaining = self.total_hours * 3600 - elapsed
        return max(0, remaining / 3600)
    
    def get_gpu(self):
        """获取可用GPU"""
        return self.gpu_queue.get()
    
    def release_gpu(self, gpu_id):
        """释放GPU"""
        self.gpu_queue.put(gpu_id)
    
    def phase1_baseline_verification(self):
        """阶段1：基线验证和基础参数探索"""
        print("🔬 阶段1：基线验证和基础参数探索")
        configs = [
            # 基线验证
            ("baseline", "linear", "1.0|0.0|0.0"),
            
            # 温和的置信度调整
            ("conf_11", "linear", "1.1|0.0|0.0"),
            ("conf_12", "linear", "1.2|0.0|0.0"),
            ("conf_13", "linear", "1.3|0.0|0.0"),
            ("conf_14", "linear", "1.4|0.0|0.0"),
            
            # 温和的attention权重
            ("attn_01", "linear", "1.0|0.0|0.1"),
            ("attn_02", "linear", "1.0|0.0|0.2"),
            ("attn_03", "linear", "1.0|0.0|0.3"),
            
            # 温和的熵惩罚
            ("ent_penalty_01", "linear", "1.0|-0.1|0.0"),
            ("ent_penalty_02", "linear", "1.0|-0.2|0.0"),
            
            # 组合策略
            ("combo_1", "linear", "1.2|0.0|0.1"),
            ("combo_2", "linear", "1.1|-0.1|0.1"),
        ]
        return configs
    
    def phase2_grid_search(self):
        """阶段2：网格搜索"""
        print("🔬 阶段2：精细网格搜索")
        configs = []
        
        # 基于阶段1结果的精细搜索
        conf_weights = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        entropy_weights = [0.0, -0.05, -0.1, -0.15, -0.2]
        pmass_weights = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        for i, (w1, w2, w3) in enumerate(itertools.product(conf_weights, entropy_weights, pmass_weights)):
            if i >= 50:  # 限制数量
                break
            configs.append((f"grid_{i}", "linear", f"{w1}|{w2}|{w3}"))
        
        return configs
    
    def phase3_nonlinear_exploration(self):
        """阶段3：非线性融合探索"""
        print("🔬 阶段3：非线性融合探索")
        configs = []
        
        # 非线性配置 - 小心选择参数避免exp爆炸
        nonlinear_configs = [
            ("nonlin_1", "nonlinear", "1.0|-0.1|0.1"),
            ("nonlin_2", "nonlinear", "1.1|-0.05|0.15"),
            ("nonlin_3", "nonlinear", "1.2|0.0|0.1"),
            ("nonlin_4", "nonlinear", "0.9|-0.1|0.2"),
            ("nonlin_5", "nonlinear", "1.0|-0.15|0.05"),
            ("nonlin_6", "nonlinear", "1.3|-0.05|0.1"),
            ("nonlin_7", "nonlinear", "1.1|-0.08|0.12"),
            ("nonlin_8", "nonlinear", "1.05|-0.12|0.18"),
        ]
        
        configs.extend(nonlinear_configs)
        return configs
    
    def phase4_evolutionary_search(self):
        """阶段4：进化算法搜索"""
        print("🔬 阶段4：基于最佳结果的进化搜索")
        configs = []
        
        if self.best_config:
            # 基于最佳配置的变异
            best_weights = [float(x) for x in self.best_config.split('|')]
            
            for i in range(20):
                # 添加随机扰动
                w1 = best_weights[0] + random.uniform(-0.2, 0.2)
                w2 = best_weights[1] + random.uniform(-0.1, 0.1)
                w3 = best_weights[2] + random.uniform(-0.1, 0.1)
                
                # 确保合理范围
                w1 = max(0.5, min(2.0, w1))
                w2 = max(-0.5, min(0.2, w2))
                w3 = max(0.0, min(1.0, w3))
                
                configs.append((f"evo_{i}", "linear", f"{w1:.3f}|{w2:.3f}|{w3:.3f}"))
        else:
            # 如果没有好的结果，使用随机搜索
            for i in range(20):
                w1 = random.uniform(0.8, 1.5)
                w2 = random.uniform(-0.3, 0.1)
                w3 = random.uniform(0.0, 0.5)
                configs.append((f"random_{i}", "linear", f"{w1:.3f}|{w2:.3f}|{w3:.3f}"))
        
        return configs
    
    def phase5_fine_tuning(self):
        """阶段5：精细调优"""
        print("🔬 阶段5：基于最佳结果的精细调优")
        configs = []
        
        if self.best_config:
            best_weights = [float(x) for x in self.best_config.split('|')]
            
            # 在最佳配置周围进行精细搜索
            for i in range(15):
                w1 = best_weights[0] + random.uniform(-0.05, 0.05)
                w2 = best_weights[1] + random.uniform(-0.02, 0.02)
                w3 = best_weights[2] + random.uniform(-0.02, 0.02)
                
                configs.append((f"fine_{i}", "linear", f"{w1:.4f}|{w2:.4f}|{w3:.4f}"))
        
        return configs
    
    def run_single_test_with_dynamic_gpu(self, config_name, fusion_mode, weights, phase):
        """使用动态GPU分配运行单个测试"""
        # 获取可用GPU
        gpu_id = self.get_gpu()
        
        try:
            with self.lock:
                remaining_hours = self.get_remaining_time()
                print(f"🚀 GPU{gpu_id} 启动: {config_name} ({fusion_mode}: {weights}) [剩余{remaining_hours:.1f}h]")
            
            # 创建测试脚本
            script_content = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu_id}
cd /home/zhaoyifei/Sampling/slow-fast-sampling

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate slow_fast_sampling

python3 evaluation_script.py --model dream \\
  --model_args "pretrained=/home/zhaoyifei/Sampling/Models/Dream-base,max_new_tokens=256,diffusion_steps=256,temperature=0.2,top_p=0.95,alg=entropy,alg_temp=0.0,add_bos_token=true,is_feature_cache=False,is_cfg_cache=False,use_attention_fusion=True,fusion_type=static,static_weight={weights},fusion_mode={fusion_mode},k_exploration_steps=6,cycle_length_stability_window=2,use_fast_attention=True" \\
  --tasks gsm8k \\
  --num_fewshot 8 \\
  --batch_size 1 \\
  --limit {self.sample_size} \\
  --output_path ./fixed_comprehensive_20h_results/gpu{gpu_id}_{config_name} \\
  --log_samples
"""
            
            script_path = f"/tmp/fixed_comp_test_gpu{gpu_id}_{config_name}_{int(time.time())}.sh"
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
                    
                    with self.lock:
                        self.results.append({
                            'gpu_id': gpu_id,
                            'config_name': config_name,
                            'fusion_mode': fusion_mode,
                            'weights': weights,
                            'score': score,
                            'tps': tps,
                            'duration': duration,
                            'improvement': improvement,
                            'phase': phase
                        })
                        
                        # 更新最佳结果
                        if score > self.best_score:
                            self.best_score = score
                            self.best_config = weights
                            print(f"🏆 新的最佳结果: {config_name} - {score:.4f} (+{improvement:.2f}%)")
                        
                        # 写入CSV
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                gpu_id, config_name, fusion_mode, weights, 
                                self.sample_size, score, tps, duration, improvement, phase
                            ])
                        
                        self.save_state()
                    
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
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ GPU{gpu_id} 超时: {config_name}")
            return False
        except Exception as e:
            print(f"💥 GPU{gpu_id} 异常: {config_name} - {e}")
            return False
        finally:
            # 确保GPU被释放
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
    
    def run_phase_with_dynamic_gpu(self, phase_name, configs, max_parallel=8):
        """使用动态GPU分配运行阶段"""
        if self.get_remaining_time() <= 0:
            print("⏰ 时间已用完，停止实验")
            return False
        
        print(f"\n🚀 开始{phase_name}")
        print(f"📊 配置数量: {len(configs)}")
        print(f"⏱️  剩余时间: {self.get_remaining_time():.1f}小时")
        print(f"🔧 使用动态GPU分配，避免资源冲突")
        print("=" * 80)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # 提交所有任务，使用动态GPU分配
            futures = []
            for config_name, fusion_mode, weights in configs:
                if self.get_remaining_time() <= 0:
                    break
                
                future = executor.submit(self.run_single_test_with_dynamic_gpu, config_name, fusion_mode, weights, phase_name)
                futures.append(future)
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                if self.get_remaining_time() <= 0:
                    print("⏰ 时间已用完，停止当前阶段")
                    break
                
                completed += 1
                try:
                    future.result()
                    print(f"📊 {phase_name} 进度: {completed}/{total}")
                except Exception as e:
                    print(f"💥 任务异常: {e}")
        
        return True
    
    def run_comprehensive_experiment(self):
        """运行20小时全面实验"""
        print("🚀 修复版AdLLM融合策略20小时全面实验开始")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ 预计结束: {(datetime.now() + timedelta(hours=self.total_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 样本数量: {self.sample_size}")
        print(f"🔧 修复: 使用动态GPU分配避免OOM")
        print("=" * 80)
        
        # 阶段1：基线验证
        if self.get_remaining_time() > 0:
            configs = self.phase1_baseline_verification()
            self.run_phase_with_dynamic_gpu("阶段1：基线验证", configs)
        
        # 阶段2：网格搜索
        if self.get_remaining_time() > 0:
            configs = self.phase2_grid_search()
            self.run_phase_with_dynamic_gpu("阶段2：网格搜索", configs)
        
        # 阶段3：非线性探索
        if self.get_remaining_time() > 0:
            configs = self.phase3_nonlinear_exploration()
            self.run_phase_with_dynamic_gpu("阶段3：非线性探索", configs)
        
        # 阶段4：进化搜索
        if self.get_remaining_time() > 0:
            configs = self.phase4_evolutionary_search()
            self.run_phase_with_dynamic_gpu("阶段4：进化搜索", configs)
        
        # 阶段5：精细调优
        if self.get_remaining_time() > 0:
            configs = self.phase5_fine_tuning()
            self.run_phase_with_dynamic_gpu("阶段5：精细调优", configs)
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 100)
        print("🏁 修复版20小时全面实验完成！")
        print("=" * 100)
        
        if not self.results:
            print("😔 没有成功的结果")
            return
        
        # 按分数排序
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 实验统计:")
        print(f"   总配置数: {len(self.results)}")
        print(f"   实验时长: {(time.time() - self.start_time) / 3600:.1f}小时")
        
        # 显示前10名
        print(f"\n🏆 前10名结果:")
        for i, result in enumerate(self.results[:10], 1):
            score = result['score']
            improvement = result['improvement']
            
            if improvement > 0:
                mark = "🎉"
                improvement_str = f"+{improvement:.1f}%"
            else:
                mark = "  "
                improvement_str = f"{improvement:.1f}%"
            
            print(f"{mark}{i:2d}. {result['config_name']:<20} ({result['fusion_mode']:<8}) {result['weights']:<15} {score:.4f} {improvement_str}")
        
        # 最佳结果详情
        if self.results[0]['score'] > self.baseline_score:
            best = self.results[0]
            improvement = best['improvement']
            print(f"\n🏆 最终最佳结果:")
            print(f"   配置: {best['config_name']}")
            print(f"   模式: {best['fusion_mode']}")
            print(f"   权重: {best['weights']}")
            print(f"   分数: {best['score']:.4f}")
            print(f"   提升: +{improvement:.2f}%")
            print(f"   阶段: {best['phase']}")
        else:
            print(f"\n😔 没有配置超越baseline ({self.baseline_score})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='修复版AdLLM融合策略20小时全面实验')
    parser.add_argument('--hours', type=int, default=20, help='实验总时长（小时）')
    parser.add_argument('--samples', type=int, default=30, help='每个配置的样本数量')
    
    args = parser.parse_args()
    
    experiment = FixedComprehensive20HExperiment(total_hours=args.hours, sample_size=args.samples)
    experiment.run_comprehensive_experiment()
