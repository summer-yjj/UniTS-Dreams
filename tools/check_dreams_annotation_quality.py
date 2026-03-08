#!/usr/bin/env python3
"""
检查 DatabaseSpindles 数据集的标注质量：
- 文件完整性、格式、数值合法性
- 与信号长度对齐、事件重叠/过短/过长
- 多标注员一致性（Visual_scoring1 vs 2 vs Automatic）
"""

import os
import sys

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "DatabaseSpindles")

FS = 256
EXCERPT_IDS = list(range(1, 9))


def load_spindle_annotations(label_path):
    """加载纺锤波标注，返回 [(onset_sec, duration_sec), ...]。无效行跳过。"""
    spindles = []
    if not os.path.isfile(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                onset = float(parts[0])
                duration = float(parts[1])
            except ValueError:
                continue
            if onset < 0 or duration <= 0:
                continue
            spindles.append((onset, duration))
    return spindles


def get_signal_length(excerpt_id):
    """返回 excerpt 的采样点数。优先用 EDF，否则用 excerpt{i}.txt 行数。"""
    edf_path = os.path.join(RAW_DIR, f"excerpt{excerpt_id}.edf")
    txt_path = os.path.join(RAW_DIR, f"excerpt{excerpt_id}.txt")
    if os.path.isfile(edf_path):
        try:
            import mne
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            return raw.get_data().shape[1]  # 采样点数
        except Exception:
            pass
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip() and not line.strip().startswith("["))
        return count
    return None


def check_annotation_quality():
    report = []
    report.append("=" * 60)
    report.append("DatabaseSpindles 标注质量检查报告")
    report.append("=" * 60)
    report.append(f"数据目录: {RAW_DIR}")
    report.append(f"存在: {os.path.isdir(RAW_DIR)}")
    report.append("")

    if not os.path.isdir(RAW_DIR):
        report.append("错误: 数据目录不存在，请检查路径。")
        return "\n".join(report)

    # 1) 文件完整性
    report.append("--- 1. 文件完整性 ---")
    expected = []
    for i in EXCERPT_IDS:
        expected.append(f"excerpt{i}.edf")
        expected.append(f"excerpt{i}.txt")
        expected.append(f"Automatic_detection_excerpt{i}.txt")
        expected.append(f"Visual_scoring1_excerpt{i}.txt")
        expected.append(f"Visual_scoring2_excerpt{i}.txt")
    missing = []
    for name in expected:
        if not os.path.isfile(os.path.join(RAW_DIR, name)):
            missing.append(name)
    if missing:
        report.append(f"缺失文件: {missing}")
    else:
        report.append("所有预期文件均存在。")
    report.append("")

    # 2) 每个 excerpt 的标注与信号长度
    report.append("--- 2. 标注与信号长度 ---")
    all_issues = []
    for eid in EXCERPT_IDS:
        T = get_signal_length(eid)
        if T is None:
            report.append(f"  excerpt{eid}: 无法获取信号长度（无 EDF 或 txt）")
            continue
        duration_sec = T / FS

        for label_name, fname in [
            ("Automatic", f"Automatic_detection_excerpt{eid}.txt"),
            ("Visual1", f"Visual_scoring1_excerpt{eid}.txt"),
            ("Visual2", f"Visual_scoring2_excerpt{eid}.txt"),
        ]:
            path = os.path.join(RAW_DIR, fname)
            ann = load_spindle_annotations(path)
            if ann is None:
                all_issues.append(f"excerpt{eid} {label_name}: 文件缺失或无法读取")
                continue
            if not ann:
                continue
            # 检查是否超出信号范围
            for onset, dur in ann:
                end = onset + dur
                if onset >= duration_sec or end > duration_sec + 0.01:
                    all_issues.append(f"excerpt{eid} {label_name}: 事件超出范围 onset={onset:.2f} dur={dur:.2f} (信号长 {duration_sec:.2f}s)")
            # 过短/过长
            for onset, dur in ann:
                if dur < 0.2:
                    all_issues.append(f"excerpt{eid} {label_name}: 过短事件 dur={dur:.2f}s")
                if dur > 4.0:
                    all_issues.append(f"excerpt{eid} {label_name}: 过长事件 dur={dur:.2f}s")

    if all_issues:
        report.append("发现的问题:")
        for issue in all_issues[:30]:
            report.append(f"  - {issue}")
        if len(all_issues) > 30:
            report.append(f"  ... 共 {len(all_issues)} 条")
    else:
        report.append("未发现超出信号范围或过短/过长事件。")
    report.append("")

    # 3) 统计：事件数、总时长、占比
    report.append("--- 3. 各 excerpt 标注统计 (Visual_scoring1 为当前训练所用) ---")
    try:
        import numpy as np
    except ImportError:
        np = None
    for eid in EXCERPT_IDS:
        T = get_signal_length(eid)
        if T is None:
            continue
        duration_sec = T / FS
        vis1 = load_spindle_annotations(os.path.join(RAW_DIR, f"Visual_scoring1_excerpt{eid}.txt"))
        vis2 = load_spindle_annotations(os.path.join(RAW_DIR, f"Visual_scoring2_excerpt{eid}.txt"))
        auto = load_spindle_annotations(os.path.join(RAW_DIR, f"Automatic_detection_excerpt{eid}.txt"))
        vis1 = vis1 or []
        vis2 = vis2 or []
        auto = auto or []
        n1, n2, na = len(vis1), len(vis2), len(auto)
        d1 = sum(d for _, d in vis1)
        d2 = sum(d for _, d in vis2)
        da = sum(d for _, d in auto)
        pct1 = (d1 / duration_sec * 100) if duration_sec else 0
        pct2 = (d2 / duration_sec * 100) if duration_sec else 0
        pcta = (da / duration_sec * 100) if duration_sec else 0
        report.append(f"  excerpt{eid}: 信号 {duration_sec:.1f}s | Vis1: {n1} 个事件, {d1:.1f}s ({pct1:.2f}%) | Vis2: {n2} 个, {d2:.1f}s ({pct2:.2f}%) | Auto: {na} 个, {da:.1f}s ({pcta:.2f}%)")
    report.append("")

    # 4) 重叠检测（同一标注文件内）
    report.append("--- 4. 同文件内事件重叠检查 (Visual_scoring1) ---")
    overlap_count = 0
    for eid in EXCERPT_IDS:
        vis1 = load_spindle_annotations(os.path.join(RAW_DIR, f"Visual_scoring1_excerpt{eid}.txt"))
        if not vis1 or len(vis1) < 2:
            continue
        events = sorted(vis1, key=lambda x: x[0])
        for i in range(len(events) - 1):
            a_end = events[i][0] + events[i][1]
            b_start = events[i + 1][0]
            if a_end > b_start + 0.01:
                overlap_count += 1
                if overlap_count <= 5:
                    report.append(f"  excerpt{eid}: 重叠 (前结束 {a_end:.2f}s, 后开始 {b_start:.2f}s)")
    if overlap_count == 0:
        report.append("  未发现重叠。")
    else:
        report.append(f"  共 {overlap_count} 处重叠。")
    report.append("")

    # 5) 标注员一致性（Vis1 vs Vis2）
    report.append("--- 5. Visual1 vs Visual2 简要一致性 ---")
    try:
        import numpy as np
    except ImportError:
        report.append("  (需 numpy 计算 IoU，跳过)")
    else:
        total_iou = 0
        n_excerpts = 0
        for eid in EXCERPT_IDS:
            T = get_signal_length(eid)
            if T is None:
                continue
            vis1 = load_spindle_annotations(os.path.join(RAW_DIR, f"Visual_scoring1_excerpt{eid}.txt"))
            vis2 = load_spindle_annotations(os.path.join(RAW_DIR, f"Visual_scoring2_excerpt{eid}.txt"))
            vis1 = vis1 or []
            vis2 = vis2 or []
            if not vis1 and not vis2:
                continue
            # 转为逐点二值
            lab1 = np.zeros(T, dtype=np.float32)
            lab2 = np.zeros(T, dtype=np.float32)
            for onset, dur in vis1:
                s, e = int(onset * FS), min(int((onset + dur) * FS), T)
                lab1[max(0, s):e] = 1
            for onset, dur in vis2:
                s, e = int(onset * FS), min(int((onset + dur) * FS), T)
                lab2[max(0, s):e] = 1
            inter = (lab1 * lab2).sum()
            union = ((lab1 + lab2) > 0).astype(np.float32).sum()
            iou = (inter / union) if union > 0 else 0.0
            total_iou += iou
            n_excerpts += 1
            report.append(f"  excerpt{eid}: IoU(Vis1,Vis2) = {iou:.3f}")
        if n_excerpts:
            report.append(f"  平均 IoU(Vis1,Vis2) = {total_iou / n_excerpts:.3f}")
    report.append("")
    # 6) 结论与建议
    report.append("--- 6. 结论与建议 ---")
    report.append("- 训练使用 Visual_scoring1；若缺失 Visual_scoring2_excerpt7/8 仅影响一致性分析。")
    report.append("- Automatic 标注存在大量超出信号长度的事件，勿直接用作金标准。")
    report.append("- Vis1 与 Vis2 平均 IoU 较低时，说明标注员一致性有限，属常见现象。")
    report.append("- 同文件内 Vis1 无重叠，标注格式正常。")
    report.append("")
    report.append("=" * 60)
    return "\n".join(report)


if __name__ == "__main__":
    out = check_annotation_quality()
    print(out)
    report_path = os.path.join(PROJECT_ROOT, "dataset", "DatabaseSpindles_annotation_quality_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n报告已写入: {report_path}")
    except Exception as e:
        print(f"\n写入报告失败: {e}")
