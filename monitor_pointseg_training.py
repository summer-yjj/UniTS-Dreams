#!/usr/bin/env python3
"""
监控纺锤波分割训练的关键指标
用于诊断模型坍塌和类别不平衡问题
"""
import json
import sys
from pathlib import Path

def monitor_training(checkpoint_dir=None):
    """Parse training logs and alert on issues."""
    
    if checkpoint_dir is None:
        # 查找最新的checkpoint目录
        checkpoints = list(Path("./checkpoints").glob("point_segmentation_*"))
        if not checkpoints:
            print("❌ No checkpoints found. Run training first.")
            return
        checkpoint_dir = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\n📊 Monitoring: {checkpoint_dir.name}")
    print("=" * 80)
    
    # Try to find results.json
    results_file = checkpoint_dir.parent / f"{checkpoint_dir.name}_results" / "results.json"
    if not results_file.exists():
        results_file = checkpoint_dir.parent / f"{checkpoint_dir.name}_results.json"
    
    if not results_file.exists():
        print(f"⚠️  Results file not found. Looking for .csv instead...")
        csv_file = checkpoint_dir.parent / f"{checkpoint_dir.name}_results.csv"
        if csv_file.exists():
            import csv
            with open(csv_file) as f:
                reader = list(csv.DictReader(f))
                if len(reader) >= 1:
                    last_row = reader[-1]
                    print(f"\nLast epoch ({len(reader)}):")
                    for k, v in list(last_row.items())[:10]:
                        print(f"  {k}: {v}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Key metrics to check
    epochs = results.get("epochs", [])
    if not epochs:
        print("❌ No epoch data found")
        return
    
    print("\n🔍 关键指标检查 (Critical Metrics):\n")
    
    # Check for model collapse
    first_epoch = epochs[0] if epochs else {}
    last_epoch = epochs[-1] if epochs else {}
    
    pred_cls1_first = first_epoch.get("val_pred_cls1", None)
    pred_cls1_last = last_epoch.get("val_pred_cls1", None)
    gt_cls1 = first_epoch.get("val_gt_cls1", None)
    
    print(f"Epoch 1:")
    print(f"  pred_cls1: {pred_cls1_first:.4f} (should decrease from 1.0)")
    print(f"  gt_cls1:   {gt_cls1:.4f} (true positive ratio)")
    
    if pred_cls1_first and pred_cls1_first > 0.9:
        print(f"  ⚠️  WARNING: Model predicting ~100% class 1 in epoch 1!")
    
    print(f"\nEpoch {len(epochs)}:")
    print(f"  pred_cls1: {pred_cls1_last:.4f}")
    
    if pred_cls1_last and pred_cls1_last > 0.95:
        print(f"  🔴 CRITICAL: Model still predicting ~100% class 1!")
        print(f"     → Tversky parameters or class_weight still not effective")
    elif pred_cls1_last and 0.3 < pred_cls1_last < 0.7:
        print(f"  ✅ GOOD: Model learning balanced predictions")
    
    # Print validation metrics over time
    print(f"\n📈 Validation Metrics Trend:")
    print(f"{'Epoch':<8} {'Val_Acc':<10} {'Spindle_F1':<12} {'Pred_Cls1':<12}")
    print("-" * 45)
    
    for i in [0, len(epochs)//4, len(epochs)//2, (3*len(epochs))//4, -1]:
        if i >= len(epochs):
            continue
        ep = epochs[i]
        epoch_num = ep.get("epoch", i+1)
        acc = ep.get("val_acc", 0)
        f1 = ep.get("val_spindle_f1", 0)
        pred = ep.get("val_pred_cls1", 0)
        print(f"{epoch_num:<8} {acc:<10.4f} {f1:<12.4f} {pred:<12.4f}")
    
    # Check if metrics are improving
    spindle_f1_list = [ep.get("val_spindle_f1", 0) for ep in epochs]
    if spindle_f1_list and max(spindle_f1_list) > 0.1:
        print(f"\n✅ Model showing learning signals (F1 > 0.1)")
    else:
        print(f"\n❌ Model not learning (all F1 < 0.1) - Parameters need adjustment")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        monitor_training(sys.argv[1])
    else:
        monitor_training()
