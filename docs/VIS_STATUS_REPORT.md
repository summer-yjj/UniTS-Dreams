# Step 1: 可视化现状报告（point_segmentation）

## 1) 关键词与对应文件路径汇总

| 关键词 | 文件路径 | 说明 |
|--------|----------|------|
| **wandb** | `run.py` (L8, L147-154), `exp/exp_pointseg.py` (L20-22, L170-171), `exp/exp_sup.py`, `exp/exp_pretrain.py`, `run_pretrain.py` | 可选依赖；point_seg 仅 log train_loss, val_acc, val_macro_f1, val_spindle_f1，无 val_loss/val_event_f1/lr |
| **tensorboard / SummaryWriter / mlflow** | 无 | 未使用 |
| **csv / log.csv / metrics.csv** | 无 metrics.csv；`exp/exp_pointseg.py` 写 `results/<setting>/results.csv` | 训练过程无逐 epoch 的 metrics.csv |
| **matplotlib / plt** | `tools/plot_segmentation.py` (L8-10, L41, L74-79, L85-99) | 仅用于分割图、saliency 图 |
| **seaborn** | 无 | 未使用 |
| **results.json / results.csv** | `exp/exp_pointseg.py` (L276-284) | 测试阶段写入 `results/<setting>/results.json`, `results/<setting>/results.csv` |
| **vis/** | `exp/exp_pointseg.py` (L278, L288-297) | 测试阶段写入 `results/<setting>/vis/`，调用 `tools/plot_segmentation.run_visualization` 生成 seg_*.png、saliency_*.png（最多 20 张） |

## 2) 运行输出目录与 setting 命名规则

- **setting 生成**（`run.py` L164-166, L179-184）：
  ```text
  setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
      args.task_name,   # e.g. point_segmentation
      args.model_id,    # e.g. dreams_project_pointseg_units_baseline
      args.model,       # UniTS
      args.data,        # 默认 All
      args.features,   # 默认 M
      args.d_model,    # 512
      args.e_layers,   # 2
      args.des,        # test
      ii               # 0
  )
  ```
  示例：`point_segmentation_dreams_project_pointseg_units_baseline_UniTS_All_ftM_dm512_el2_test_0`

- **实际使用目录**：
  - **checkpoints/**：`checkpoints/<setting>/` → `best.pth`, `finetune_output.log`（exp_pointseg 写日志到 path=checkpoints/<setting>）
  - **logs/**：未使用（无 `logs/<setting>/`）
  - **results/**：`results/<setting>/` → `results.json`, `results.csv`, `vis/*.png`

## 3) 可视化现状总结

### 已有

| 项目 | 说明 |
|------|------|
| 训练曲线 | 仅通过 wandb（若安装）记录 train_loss, val_acc, val_macro_f1, val_spindle_f1；无本地曲线图、无 val_loss/val_event_f1/lr |
| 验证曲线 | 同上，无本地 png |
| vis 图 | 有。测试阶段在 `results/<setting>/vis/` 生成 seg_<excerpt>_<idx>.png、saliency_*.png（最多 20 张），来自第一个 batch |
| results.json / results.csv | 有。测试阶段写入 `results/<setting>/`，含 accuracy, macro_f1, spindle_f1, per_class, overall_macro_event_f1 等 |

### 缺失

| 项目 | 说明 |
|------|------|
| 训练曲线本地图 | 无 TensorBoard 日志；无 metrics.csv；无 loss/F1/lr 的 png |
| 验证 event 指标曲线 | wandb 未记录 val_event_f1；无本地曲线 |
| 混淆矩阵 | 无 |
| 每类指标柱状图 | 无（仅有 results.json 里的 per_class 数字） |
| 边界误差直方图 | 无 |
| F1-阈值曲线与推荐阈值 | 无 |
| 最佳/最差案例集 | 无 best_20 / worst_20 的专门目录与列表 |
| 一键生成 | 无统一脚本；训练曲线、分析图、案例集均需补齐 |

### 能否一键生成

- **当前**：仅“跑 test”可一键得到 results.json/csv + vis（前 20 张）。训练过程无可视化、无定量总结图、无案例集。
- **目标**：按 Step 2–5 补齐后，训练即写 metrics/曲线、测试后一键生成 analysis 图 + cases，并文档化命令。
