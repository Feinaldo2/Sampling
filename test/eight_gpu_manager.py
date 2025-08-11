#!/usr/bin/env python3
"""
🚀 8-GPU并行测试管理器
充分利用所有8张GPU同时运行不同的测试配置
"""

import subprocess
import time
import json
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class EightGPUManager:
    def __init__(self, sample_size=150):
        self.sample_size = sample_size
        self.results = []
        self.lock = threading.Lock()
        self.gpu_status = {i: 'idle' for i in range(8)}  # 跟踪每个GPU状态
        
        # 创建结果目录
        os.makedirs("gpu_test_results", exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = "gpu_test_results/summary.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_id', 'config_name', 'weights', 'samples', 'score', 'tps', 'duration'])
    
    def run_single_gpu_test(self, gpu_id, config_name, weights):
        """在指定GPU上运行单个测试"""
        with self.lock:
            self.gpu_status[gpu_id] = f'running_{config_name}'
        
        print(f"🚀 GPU{gpu_id} 启动测试: {config_name} ({weights})")
        
        try:
            # 运行测试
            cmd = f"bash gpu_specific_test.sh {gpu_id} '{weights}' '{config_name}' {self.sample_size}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            if result.returncode == 0:
                # 解析结果
                output = result.stdout
                score = None
                tps = None
                duration = None
                
                for line in output.split('\n'):
                    if line.startswith('分数:'):
                        try:
                            score = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif line.startswith('TPS:'):
                        try:
                            tps = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif line.startswith('用时:'):
                        try:
                            duration = int(line.split(':')[1].split('秒')[0].strip())
                        except:
                            pass
                
                with self.lock:
                    self.results.append({
                        'gpu_id': gpu_id,
                        'config_name': config_name,
                        'weights': weights,
                        'score': score,
                        'tps': tps,
                        'duration': duration,
                        'status': 'success'
                    })
                    self.gpu_status[gpu_id] = 'completed'
                
                baseline_score = 0.7333
                if score is not None:
                    if score > baseline_score:
                        improvement = (score - baseline_score) / baseline_score * 100
                        print(f"🎉 GPU{gpu_id} 完成: {config_name} - 分数: {score:.4f} (+{improvement:.2f}% vs baseline)")
                    else:
                        decline = (baseline_score - score) / baseline_score * 100
                        print(f"✅ GPU{gpu_id} 完成: {config_name} - 分数: {score:.4f} (-{decline:.2f}% vs baseline)")
                else:
                    print(f"⚠️  GPU{gpu_id} 完成但解析失败: {config_name}")
                
                return True
                
            else:
                print(f"❌ GPU{gpu_id} 测试失败: {config_name}")
                with self.lock:
                    self.results.append({
                        'gpu_id': gpu_id,
                        'config_name': config_name,
                        'weights': weights,
                        'score': None,
                        'tps': None,
                        'duration': None,
                        'status': 'failed'
                    })
                    self.gpu_status[gpu_id] = 'failed'
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ GPU{gpu_id} 测试超时: {config_name}")
            with self.lock:
                self.results.append({
                    'gpu_id': gpu_id,
                    'config_name': config_name,
                    'weights': weights,
                    'score': None,
                    'tps': None,
                    'duration': None,
                    'status': 'timeout'
                })
                self.gpu_status[gpu_id] = 'timeout'
            return False
        except Exception as e:
            print(f"💥 GPU{gpu_id} 测试异常: {config_name} - {e}")
            with self.lock:
                self.results.append({
                    'gpu_id': gpu_id,
                    'config_name': config_name,
                    'weights': weights,
                    'score': None,
                    'tps': None,
                    'duration': None,
                    'status': 'error'
                })
                self.gpu_status[gpu_id] = 'error'
            return False
        finally:
            with self.lock:
                if self.gpu_status[gpu_id].startswith('running_'):
                    self.gpu_status[gpu_id] = 'idle'
    
    def run_eight_gpu_tests(self, test_configs):
        """8-GPU并行测试"""
        print(f"🎯 开始8-GPU并行测试，共{len(test_configs)}个配置")
        print(f"📊 每个测试样本数: {self.sample_size}")
        print(f"🚀 使用GPU: 0,1,2,3,4,5,6,7")
        print("=" * 80)
        
        start_time = time.time()
        
        # 分配配置到GPU
        gpu_assignments = []
        for i, (config_name, weights) in enumerate(test_configs):
            gpu_id = i % 8  # 轮询分配到8个GPU
            gpu_assignments.append((gpu_id, config_name, weights))
        
        print("📋 GPU分配:")
        for gpu_id, config_name, weights in gpu_assignments:
            print(f"   GPU{gpu_id}: {config_name} ({weights})")
        print()
        
        # 启动并行测试
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_gpu_test, gpu_id, config_name, weights): (gpu_id, config_name, weights)
                for gpu_id, config_name, weights in gpu_assignments
            }
            
            # 等待完成
            completed = 0
            total = len(future_to_config)
            
            for future in as_completed(future_to_config):
                gpu_id, config_name, weights = future_to_config[future]
                completed += 1
                try:
                    future.result()
                    print(f"📊 进度: {completed}/{total} 完成")
                except Exception as e:
                    print(f"💥 GPU{gpu_id} 任务异常: {config_name} - {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("=" * 80)
        print(f"🏁 所有8-GPU测试完成！总用时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n📊 8-GPU测试结果报告")
        print("=" * 100)
        
        # 按分数排序
        successful_results = [r for r in self.results if r['status'] == 'success' and r['score'] is not None]
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"{'排名':<4} {'GPU':<4} {'配置名称':<25} {'权重配置':<15} {'分数':<12} {'TPS':<8} {'用时(s)':<8}")
        print("-" * 100)
        
        baseline_score = 0.7333
        
        for i, result in enumerate(successful_results, 1):
            score = result['score']
            tps = result['tps'] or 0
            duration = result['duration'] or 0
            gpu_id = result['gpu_id']
            
            # 标记是否超越baseline
            if score > baseline_score:
                mark = "🎉"
                improvement = (score - baseline_score) / baseline_score * 100
                score_str = f"{score:.4f} (+{improvement:.1f}%)"
            else:
                mark = "  "
                decline = (baseline_score - score) / baseline_score * 100
                score_str = f"{score:.4f} (-{decline:.1f}%)"
            
            print(f"{mark}{i:<3} GPU{gpu_id:<3} {result['config_name']:<25} {result['weights']:<15} {score_str:<12} {tps:<8.2f} {duration:<8}")
        
        # 按GPU分组显示
        print(f"\n📊 按GPU分组结果:")
        for gpu_id in range(8):
            gpu_results = [r for r in self.results if r['gpu_id'] == gpu_id]
            if gpu_results:
                result = gpu_results[0]
                status = result['status']
                if status == 'success' and result['score'] is not None:
                    score = result['score']
                    if score > baseline_score:
                        improvement = (score - baseline_score) / baseline_score * 100
                        print(f"   GPU{gpu_id}: {result['config_name']} - {score:.4f} (+{improvement:.1f}%) ✅")
                    else:
                        decline = (baseline_score - score) / baseline_score * 100
                        print(f"   GPU{gpu_id}: {result['config_name']} - {score:.4f} (-{decline:.1f}%)")
                else:
                    print(f"   GPU{gpu_id}: {result['config_name']} - {status} ❌")
        
        # 统计信息
        total_tests = len(self.results)
        successful_tests = len(successful_results)
        better_than_baseline = len([r for r in successful_results if r['score'] > baseline_score])
        
        print(f"\n📈 统计信息:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功测试: {successful_tests}")
        print(f"   超越baseline({baseline_score}): {better_than_baseline}")
        print(f"   成功率: {successful_tests/total_tests*100:.1f}%")
        
        if better_than_baseline > 0:
            best_result = successful_results[0]
            improvement = (best_result['score'] - baseline_score) / baseline_score * 100
            print(f"\n🏆 最佳结果:")
            print(f"   GPU{best_result['gpu_id']}: {best_result['config_name']} ({best_result['weights']})")
            print(f"   分数: {best_result['score']:.4f}")
            print(f"   提升: +{improvement:.2f}%")
        else:
            print(f"\n😔 没有配置超越baseline ({baseline_score})")

def get_eight_gpu_configs():
    """8个GPU的测试配置"""
    return [
        # GPU 0-7 分别测试不同配置
        ("baseline", "1.0|0.0|0.0"),                           # GPU 0
        ("high_conf_strong_attention", "1.5|0.0|0.3"),        # GPU 1
        ("high_conf_very_strong_attention", "1.8|0.0|0.4"),   # GPU 2
        ("high_conf_extreme_attention", "2.0|0.0|0.5"),       # GPU 3
        ("enhanced_conf_entropy_penalty", "1.5|-0.2|0.4"),    # GPU 4
        ("strong_conf_entropy_penalty", "1.8|-0.3|0.5"),      # GPU 5
        ("extreme_conf_entropy_penalty", "2.0|-0.4|0.6"),     # GPU 6
        ("attention_dominant", "1.0|0.0|1.2"),                # GPU 7
        
        # 如果有更多配置，会轮询分配
        ("ultra_conf_attention", "2.5|0.0|0.8"),              # GPU 0 (第二轮)
        ("balanced_extreme", "1.5|0.0|1.0"),                  # GPU 1 (第二轮)
    ]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='8-GPU并行测试AdLLM融合策略')
    parser.add_argument('--samples', type=int, default=150, help='每个配置的样本数量')
    
    args = parser.parse_args()
    
    print("🚀 AdLLM 8-GPU并行测试系统")
    print(f"📊 样本数量: {args.samples}")
    print(f"🎯 目标: 超越baseline (0.7333)")
    print(f"🚀 GPU数量: 8张 (GPU 0-7)")
    
    # 创建测试管理器
    manager = EightGPUManager(sample_size=args.samples)
    
    # 获取测试配置
    test_configs = get_eight_gpu_configs()
    
    print(f"📋 测试配置: {len(test_configs)}个")
    print(f"⏱️  预计时间: {args.samples / 10:.1f}分钟 (并行执行)")
    
    # 确认开始
    input("\n按Enter键开始8-GPU并行测试...")
    
    # 运行测试
    manager.run_eight_gpu_tests(test_configs)
