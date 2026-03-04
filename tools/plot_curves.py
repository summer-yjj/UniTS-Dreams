"""
读取 results/<setting>/metrics.csv，生成训练曲线图到 results/<setting>/curves/*.png
用法: python tools/plot_curves.py --setting <setting>
"""
from __future__ import division
import os
import sys
import argparse
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(metrics_path):
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def plot_class_distribution(results_dir, curves_dir):
    """若存在 class_distribution.json 则绘制 train/val 类别分布（用于检查类别不平衡）"""
    dist_path = os.path.join(results_dir, "class_distribution.json")
    if not os.path.isfile(dist_path):
        return
    try:
        with open(dist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    train = data.get("train")
    val = data.get("val")
    if not train and not val:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, d) in zip(axes, [("train", train), ("val", val)]):
        if not d or "pct" not in d:
            ax.set_title("{}: no data".format(name))
            continue
        pct = d["pct"]
        classes = sorted(pct.keys(), key=int)
        vals = [pct[c] for c in classes]
        labels = ["bg" if c == "0" else "cls{}".format(c) for c in classes]
        bars = ax.bar(labels, vals, color="steelblue", edgecolor="black")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("{} (n={})".format(name, d.get("total", 0)))
        ax.set_ylim(0, 100)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, "{:.1f}%".format(v), ha="center", va="bottom", fontsize=9)
    plt.suptitle("Class distribution (imbalance check)")
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves(results_dir, curves_dir=None):
    if curves_dir is None:
        curves_dir = os.path.join(results_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    plot_class_distribution(results_dir, curves_dir)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        print("No metrics.csv at {}".format(metrics_path))
        return

    rows = load_metrics(metrics_path)
    if not rows:
        print("Empty metrics.csv")
        return

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    train_loss = [float(r.get("train_loss", 0)) for r in rows]
    val_acc = [float(r.get("val_acc", 0)) if r.get("val_acc") else np.nan for r in rows]
    val_macro_f1 = [float(r.get("val_macro_f1", 0)) if r.get("val_macro_f1") else np.nan for r in rows]
    val_spindle_f1 = [float(r.get("val_spindle_f1", 0)) if r.get("val_spindle_f1") else np.nan for r in rows]
    val_event_f1 = [float(r.get("val_event_f1", 0)) if r.get("val_event_f1") else np.nan for r in rows]
    lr = [float(r.get("lr", 0)) if r.get("lr") else np.nan for r in rows]

    # loss 曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, "b-o", label="train_loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # F1 曲线（val point + event）
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, val_macro_f1, "g-s", label="val_macro_f1", markersize=4)
    ax.plot(epochs, val_spindle_f1, "b-o", label="val_spindle_f1", markersize=4)
    ax.plot(epochs, val_event_f1, "r-^", label="val_event_f1", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Validation F1 (point & event)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "f1_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # lr 曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lr, "m-o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "lr.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 验证准确率曲线（单独图，便于分析）
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, val_acc, "c-o", label="val_acc", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "val_acc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 综合摘要图 2x2：loss / val_acc / F1 系列 / lr
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(epochs, train_loss, "b-o", markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, val_acc, "c-o", markersize=4)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation accuracy")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, val_macro_f1, "g-s", label="macro_f1", markersize=4)
    axes[1, 0].plot(epochs, val_spindle_f1, "b-o", label="spindle_f1", markersize=4)
    axes[1, 0].plot(epochs, val_event_f1, "r-^", label="event_f1", markersize=4)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("Validation F1 (point & event)")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, lr, "m-o", markersize=4)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning rate")
    axes[1, 1].set_title("Learning rate schedule")
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 训练损失 vs 验证指标（双轴，便于看过拟合）
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, "b-o", label="train_loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, "c-s", label="val_acc", markersize=4)
    ax2.plot(epochs, val_spindle_f1, "g-^", label="val_spindle_f1", markersize=4)
    ax2.set_ylabel("Val acc / spindle_f1", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    plt.title("Train loss vs validation metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "loss_vs_val.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Curves saved to {}".format(curves_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True, help="e.g. point_segmentation_xxx_UniTS_All_ftM_dm512_el2_test_0")
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()
    results_dir = os.path.join(args.results_root, args.setting)
    if not os.path.isdir(results_dir):
        print("Not a directory: {}".format(results_dir))
        return
    plot_curves(results_dir)
    return


if __name__ == "__main__":
    main()
