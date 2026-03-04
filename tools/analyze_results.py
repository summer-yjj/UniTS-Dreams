"""
测试结果定量总结与案例集：读取 results/<setting>/ 下 results.json、per_sample_metrics.csv、
pred_labels.npy、gt_labels.npy、probs_cls1.npy，生成 analysis/*.png 与 cases/best_20、worst_20。
"""
from __future__ import division
import os
import sys
import json
import csv
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from utils.event_post import pred_to_events, gt_to_events
from utils.event_metrics import event_precision_recall_f1


def run_analysis(results_dir, num_classes=2, fs=256):
    analysis_dir = os.path.join(results_dir, "analysis")
    cases_best_dir = os.path.join(results_dir, "cases", "best_20")
    cases_worst_dir = os.path.join(results_dir, "cases", "worst_20")
    vis_dir = os.path.join(results_dir, "vis")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(cases_best_dir, exist_ok=True)
    os.makedirs(cases_worst_dir, exist_ok=True)

    # 加载
    with open(os.path.join(results_dir, "results.json"), "r", encoding="utf-8") as f:
        results = json.load(f)
    pred_labels = np.load(os.path.join(results_dir, "pred_labels.npy"))
    gt_labels = np.load(os.path.join(results_dir, "gt_labels.npy"))
    per_sample_path = os.path.join(results_dir, "per_sample_metrics.csv")
    per_sample_rows = []
    if os.path.isfile(per_sample_path):
        with open(per_sample_path, "r", encoding="utf-8") as f:
            per_sample_rows = list(csv.DictReader(f))

    probs_cls1_path = os.path.join(results_dir, "probs_cls1.npy")
    probs_cls1 = None
    if os.path.isfile(probs_cls1_path):
        probs_cls1 = np.load(probs_cls1_path)

    analysis_summary = {}

    # 1) 混淆矩阵（逐点，多类）
    pred_flat = pred_labels.flatten()
    gt_flat = gt_labels.flatten()
    valid = (gt_flat >= 0) & (gt_flat < num_classes)
    pred_flat = pred_flat[valid]
    gt_flat = gt_flat[valid].astype(int)
    cm = confusion_matrix(gt_flat, pred_flat, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(["bg" if i == 0 else "cls{}".format(i) for i in range(num_classes)])
    ax.set_yticklabels(["bg" if i == 0 else "cls{}".format(i) for i in range(num_classes)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("Point-wise confusion matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    analysis_summary["confusion_matrix"] = cm.tolist()

    # 2) 每类 point_f1 与 event_f1 条形图
    per_class = results.get("per_class", {})
    if per_class:
        classes = sorted(per_class.keys(), key=int)
        event_f1s = [per_class[c].get("event_f1", 0) for c in classes]
        labels = ["class {}".format(c) for c in classes]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
        bars = ax.bar(x, event_f1s, color="steelblue", edgecolor="black")
        ax.set_xticks(x)
        ax.set_ylabel("Event F1")
        ax.set_xticklabels(labels)
        ax.set_title("Per-class event F1")
        ax.set_ylim(0, 1.05)
        for b, v in zip(bars, event_f1s):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, "{:.3f}".format(v), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 3) 边界误差直方图（mean_boundary_error_ms）
    mbe_list = []
    for r in per_sample_rows:
        v = r.get("mean_boundary_error_ms")
        if v != "" and v is not None:
            try:
                mbe_list.append(float(v))
            except (ValueError, TypeError):
                pass
    if mbe_list:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(mbe_list, bins=min(50, len(mbe_list)), color="coral", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Mean boundary error (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of per-sample mean boundary error")
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "boundary_error_hist.png"), dpi=150, bbox_inches="tight")
        plt.close()
        analysis_summary["boundary_error_mean"] = float(np.nanmean(mbe_list))
        analysis_summary["boundary_error_median"] = float(np.nanmedian(mbe_list))

    # 4) F1 vs 阈值（spindle 类），并输出推荐阈值
    if probs_cls1 is not None and probs_cls1.size > 0:
        thresholds = np.linspace(0.05, 0.95, 19)
        f1_point = []
        f1_event = []
        for th in thresholds:
            pred_bin = (probs_cls1 >= th).astype(np.int32)
            # 逐点 F1（只对 class 1）
            gt_bin = (gt_labels == 1).astype(np.int32)
            pred_flat = pred_bin.flatten()
            gt_flat = gt_bin.flatten()
            f1p = f1_score(gt_flat, pred_flat, zero_division=0)
            f1_point.append(f1p)
            # 事件级 F1：对每个样本做 event 匹配后平均
            ev_f1s = []
            for i in range(pred_bin.shape[0]):
                pred_ev = pred_to_events(pred_bin[i], min_len=1, merge_gap=0)
                gt_ev = gt_to_events(gt_labels[i], positive_class=1, min_len=1, merge_gap=0)
                _, _, f1e = event_precision_recall_f1(pred_ev, gt_ev, iou_thr=0.3)
                ev_f1s.append(f1e)
            f1_event.append(np.nanmean(ev_f1s) if ev_f1s else 0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(thresholds, f1_point, "b-o", label="Point F1 (class 1)", markersize=4)
        ax.plot(thresholds, f1_event, "r-s", label="Event F1 (class 1)", markersize=4)
        ax.set_xlabel("Threshold (prob class 1)")
        ax.set_ylabel("F1")
        ax.set_title("F1 vs threshold (spindle)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "f1_vs_threshold.png"), dpi=150, bbox_inches="tight")
        plt.close()
        best_idx = np.argmax(f1_event)
        best_th = float(thresholds[best_idx])
        analysis_summary["recommended_threshold"] = best_th
        analysis_summary["recommended_event_f1"] = float(f1_event[best_idx])

    # 5) best_20 / worst_20 及复制 vis
    if per_sample_rows:
        valid_rows = [r for r in per_sample_rows if r.get("event_f1") != "" and r.get("event_f1") is not None]
        try:
            valid_rows = [(r, float(r["event_f1"])) for r in valid_rows]
        except (ValueError, TypeError):
            valid_rows = []
        if valid_rows:
            valid_rows.sort(key=lambda x: x[1], reverse=True)
            best_20 = [x[0] for x in valid_rows[:20]]
            worst_20 = [x[0] for x in valid_rows[-20:]]
            analysis_summary["best_20"] = [{"sample_idx": int(r["sample_idx"]), "excerpt_id": r["excerpt_id"], "event_f1": r["event_f1"]} for r in best_20]
            analysis_summary["worst_20"] = [{"sample_idx": int(r["sample_idx"]), "excerpt_id": r["excerpt_id"], "event_f1": r["event_f1"]} for r in worst_20]
            vis_info = {}
            if os.path.isfile(os.path.join(results_dir, "vis_info.json")):
                with open(os.path.join(results_dir, "vis_info.json"), "r") as f:
                    vis_info = json.load(f)
            n_vis = vis_info.get("n_vis", 20)
            for r in best_20:
                idx = int(r["sample_idx"])
                eid = r["excerpt_id"]
                if idx < n_vis:
                    src = os.path.join(vis_dir, "seg_{}_{}.png".format(eid, idx))
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(cases_best_dir, "seg_{}_{}.png".format(eid, idx)))
            for r in worst_20:
                idx = int(r["sample_idx"])
                eid = r["excerpt_id"]
                if idx < n_vis:
                    src = os.path.join(vis_dir, "seg_{}_{}.png".format(eid, idx))
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(cases_worst_dir, "seg_{}_{}.png".format(eid, idx)))

    with open(os.path.join(analysis_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    print("Analysis and cases saved under {}".format(results_dir))
    return analysis_summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="e.g. results/point_segmentation_xxx_0")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--fs", type=int, default=256)
    args = parser.parse_args()
    run_analysis(args.results_dir, num_classes=args.num_classes, fs=args.fs)


if __name__ == "__main__":
    main()
