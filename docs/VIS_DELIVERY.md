# Step 6: 训练/测试可视化交付验收

## 1) 修改/新增文件列表

| 类型 | 路径 | 说明 |
|------|------|------|
| 新增 | `docs/VIS_STATUS_REPORT.md` | Step 1 可视化现状报告 |
| 新增 | `docs/VIS_DELIVERY.md` | 本验收说明 |
| 修改 | `exp/exp_pointseg.py` | 训练：创建 results/<setting>、写 metrics.csv、TensorBoard、vali_event_f1；测试：写 per_sample_metrics.csv、pred/gt/probs_cls1.npy、analysis/cases 目录、调用 run_analysis |
| 新增 | `tools/plot_curves.py` | 读 metrics.csv，生成 curves/loss.png、f1_curves.png、lr.png |
| 新增 | `tools/analyze_results.py` | 读 results 与 per_sample、pred、gt、probs_cls1，生成 analysis/*.png、analysis_summary.json，复制 best_20/worst_20 到 cases |
| 修改 | `task.md` | 增补结果路径、TensorBoard 与 plot_curves 命令、单独运行 analyze 命令 |

## 2) 如何运行

- **训练**（会写入 `results/<setting>/metrics.csv`、`results/<setting>/tensorboard/`）：  
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_project_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_project_pointseg_units_baseline \
  --patch_len 16 --stride 16 \
  --train_epochs 2 --batch_size 16
```

- **测试**（会写入 results.json/csv、vis、per_sample_metrics、pred/gt/probs_cls1，并自动调用 analyze_results 生成 analysis 与 cases）：  
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_project_pointseg.yaml \
  --is_training 0 \
  --model UniTS \
  --model_id dreams_project_pointseg_units_baseline \
  --load_ckpt checkpoints/<setting>/best.pth
```

- **生成训练曲线 PNG**（训练结束后）：  
```bash
python tools/plot_curves.py --setting <setting>
```
  其中 `<setting>` 与 run.py 打印的一致，例如：`point_segmentation_dreams_project_pointseg_units_baseline_UniTS_All_ftM_dm512_el2_test_0`。

- **启动 TensorBoard**：  
```bash
tensorboard --logdir=results/<setting>/tensorboard --port=6006
```

- **单独重新生成测试分析图与案例集**：  
```bash
python tools/analyze_results.py --results_dir results/<setting> --num_classes 2 --fs 256
```

## 3) 跑完后 results/<setting>/ 下可见目录与文件

| 路径 | 说明 |
|------|------|
| `results/<setting>/metrics.csv` | 训练时写入；每行：epoch, train_loss, val_acc, val_macro_f1, val_spindle_f1, val_event_f1, lr |
| `results/<setting>/tensorboard/` | 训练时写入；TensorBoard 事件 |
| `results/<setting>/curves/` | 需运行 `plot_curves.py` 后才有 |
| `results/<setting>/curves/loss.png` | 训练 loss 曲线 |
| `results/<setting>/curves/f1_curves.png` | val macro_f1、spindle_f1、event_f1 曲线 |
| `results/<setting>/curves/lr.png` | 学习率曲线 |
| `results/<setting>/results.json` | 测试汇总指标 |
| `results/<setting>/results.csv` | 同上 |
| `results/<setting>/per_sample_metrics.csv` | 每样本 event_f1、mean_boundary_error_ms 等 |
| `results/<setting>/pred_labels.npy` | 测试集逐点预测标签 |
| `results/<setting>/gt_labels.npy` | 测试集逐点真实标签 |
| `results/<setting>/probs_cls1.npy` | 测试集 class 1 概率 [N,T] |
| `results/<setting>/vis/` | seg_*.png、saliency_*.png |
| `results/<setting>/vis_info.json` | n_vis、has_vis_sample_idx_range |
| `results/<setting>/analysis/` | 分析图与摘要 |
| `results/<setting>/analysis/confusion_matrix.png` | 见下表 |
| `results/<setting>/analysis/per_class_metrics.png` | 见下表 |
| `results/<setting>/analysis/boundary_error_hist.png` | 见下表 |
| `results/<setting>/analysis/f1_vs_threshold.png` | 见下表 |
| `results/<setting>/analysis/analysis_summary.json` | 推荐阈值、best_20/worst_20 列表等 |
| `results/<setting>/cases/best_20/` | 从 vis 复制的 event_f1 最高的 20 张图 |
| `results/<setting>/cases/worst_20/` | 从 vis 复制的 event_f1 最低的 20 张图 |

## 4) 每张图回答的工程问题（一句话/图）

| 图 | 回答的问题 |
|----|------------|
| `curves/loss.png` | 训练 loss 是否收敛、是否过拟合。 |
| `curves/f1_curves.png` | 验证集上逐点 F1 与事件 F1 随 epoch 的变化，便于选 epoch 或早停。 |
| `curves/lr.png` | 学习率调度是否符合预期。 |
| `analysis/confusion_matrix.png` | 逐点预测与真实类别之间的混淆情况（多类）。 |
| `analysis/per_class_metrics.png` | 每个事件类的 event F1 是否均衡、哪类最差。 |
| `analysis/boundary_error_hist.png` | 边界误差的分布是否集中、是否存在长尾。 |
| `analysis/f1_vs_threshold.png` | 推理时选用何阈值可使事件 F1 最优，并得到推荐阈值。 |
| `vis/seg_*.png` | 单样本上信号、GT、预测概率/掩码与事件边界是否对齐。 |
| `vis/saliency_*.png` | 模型在时间维上更关注哪些位置（可解释性）。 |
| `cases/best_20/`、`cases/worst_20/` | 哪些样本表现最好/最差，便于做错误分析与改进。 |
