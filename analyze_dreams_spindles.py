#!/usr/bin/env python3
"""
统计DREAMS纺锤波标注的分布情况
检查原始标注质量和正样本分布
"""

import os
import numpy as np
from collections import defaultdict

# 数据集路径
RAW_DIR = r"C:\Users\严晶晶\Desktop\UniTS_cursor\dataset\DatabaseSpindles"

# 采样率
FS = 256

def load_spindle_annotations(label_path):
    """加载纺锤波标注，返回(onset, duration)列表"""
    spindles = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    onset = float(parts[0])
                    duration = float(parts[1])
                    spindles.append((onset, duration))
                except ValueError:
                    continue
    return spindles

def analyze_spindle_distribution():
    """分析纺锤波分布"""
    results = {}

    for excerpt_id in range(1, 9):
        # 读取自动检测标注
        auto_path = os.path.join(RAW_DIR, f"Automatic_detection_excerpt{excerpt_id}.txt")
        # 读取视觉评分标注
        vis1_path = os.path.join(RAW_DIR, f"Visual_scoring1_excerpt{excerpt_id}.txt")
        vis2_path = os.path.join(RAW_DIR, f"Visual_scoring2_excerpt{excerpt_id}.txt")

        # 读取EEG长度（从txt文件）
        eeg_path = os.path.join(RAW_DIR, f"excerpt{excerpt_id}.txt")
        with open(eeg_path, 'r') as f:
            eeg_length = sum(1 for line in f if line.strip() and not line.startswith('['))

        total_duration_sec = eeg_length / FS

        # 分析标注
        auto_spindles = load_spindle_annotations(auto_path) if os.path.exists(auto_path) else []
        vis1_spindles = load_spindle_annotations(vis1_path) if os.path.exists(vis1_path) else []
        vis2_spindles = load_spindle_annotations(vis2_path) if os.path.exists(vis2_path) else []

        # 计算统计
        for name, spindles in [("auto", auto_spindles), ("vis1", vis1_spindles), ("vis2", vis2_spindles)]:
            if not spindles:
                continue

            durations = [d for _, d in spindles]
            total_spindle_time = sum(durations)
            spindle_percentage = (total_spindle_time / total_duration_sec) * 100

            results[f"excerpt{excerpt_id}_{name}"] = {
                "num_spindles": len(spindles),
                "total_duration_sec": total_spindle_time,
                "avg_duration_sec": np.mean(durations) if durations else 0,
                "min_duration_sec": min(durations) if durations else 0,
                "max_duration_sec": max(durations) if durations else 0,
                "spindle_percentage": spindle_percentage,
                "total_recording_sec": total_duration_sec
            }

    return results

def print_summary(results):
    """打印汇总统计"""
    print("=== DREAMS纺锤波标注分布分析 ===\n")

    # 按excerpt分组
    excerpt_stats = defaultdict(list)

    for key, stats in results.items():
        excerpt_id = key.split('_')[0]
        label_type = key.split('_')[1]
        excerpt_stats[excerpt_id].append((label_type, stats))

    for excerpt_id in sorted(excerpt_stats.keys(), key=lambda x: int(x.replace('excerpt', ''))):
        print(f"📊 {excerpt_id.upper()}")
        for label_type, stats in excerpt_stats[excerpt_id]:
            print(f"  {label_type.upper()}: {stats['num_spindles']} 个纺锤波, "
                  f"总时长 {stats['total_duration_sec']:.1f}s ({stats['spindle_percentage']:.2f}%), "
                  f"平均 {stats['avg_duration_sec']:.2f}s, "
                  f"范围 [{stats['min_duration_sec']:.2f}, {stats['max_duration_sec']:.2f}]s")
        print()

    # 整体统计
    print("=== 整体统计 ===")

    for label_type in ["auto", "vis1", "vis2"]:
        type_stats = [stats for key, stats in results.items() if key.endswith(f"_{label_type}")]

        if not type_stats:
            continue

        total_spindles = sum(s['num_spindles'] for s in type_stats)
        total_duration = sum(s['total_duration_sec'] for s in type_stats)
        avg_duration = np.mean([s['avg_duration_sec'] for s in type_stats if s['avg_duration_sec'] > 0])
        total_recording = sum(s['total_recording_sec'] for s in type_stats)
        overall_percentage = (total_duration / total_recording) * 100

        print(f"{label_type.upper()}: 总共 {total_spindles} 个纺锤波, "
              f"总时长 {total_duration:.1f}s ({overall_percentage:.2f}%), "
              f"平均持续时间 {avg_duration:.2f}s")

    # 训练/验证/测试集划分
    train_ids = [1,2,3,4,5]
    val_ids = [6]
    test_ids = [7,8]

    print("\n=== 训练/验证/测试集统计 (使用VIS1标注) ===")

    for split_name, ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        split_stats = [results.get(f"excerpt{i}_vis1") for i in ids]
        split_stats = [s for s in split_stats if s]

        if not split_stats:
            continue

        total_spindles = sum(s['num_spindles'] for s in split_stats)
        total_duration = sum(s['total_duration_sec'] for s in split_stats)
        total_recording = sum(s['total_recording_sec'] for s in split_stats)
        percentage = (total_duration / total_recording) * 100

        print(f"{split_name}: {total_spindles} 个纺锤波, "
              f"总时长 {total_duration:.1f}s ({percentage:.2f}%)")

def plot_distributions(results):
    """绘制分布图（暂时跳过）"""
    print("⚠️ matplotlib兼容性问题，跳过绘图")

if __name__ == "__main__":
    results = analyze_spindle_distribution()
    print_summary(results)
    plot_distributions(results)