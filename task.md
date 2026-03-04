# DREAMS 逐点分割任务（point_segmentation）使用说明

## 1) 数据格式

### 1.1 统一项目格式（推荐，多类可扩展）

最终多类检测与定位使用 `data/dreams_project/` 目录结构：

```text
data/dreams_project/
  raw/                      # 原始 EDF/annotations（可选，未来接入）
  processed/
    pointseg/
      train/
        <excerpt_id>/signal.npy   # float32, [T]
        <excerpt_id>/label.npy    # int64, [T]，0=bg,1=spindle,2=kcomplex,...
      val/...
      test/...
    splits/                 # 可选
      train.txt             # 每行一个 excerpt_id（如 train_00000）
      val.txt
      test.txt
  meta.json                 # fs/label_map/来源/生成时间/优先级等
```

当前脚本会将原有窗口级 npz 中的每个窗口视为一个独立 excerpt，生成逐点标签（窗口内全 0 或全 1）。未来接入多类事件时，只需在 `raw/annotations/` 下放入按类组织的标注文件，并运行多类构建脚本即可重新生成 label.npy。

### 1.2 旧格式（兼容，便于迁移）

Dataset 仍然兼容以下两种 npy 格式，便于从历史数据过渡：

- **格式 A**：单个 `.npy` 文件，形状 `[N, 2]`  
  - 第 0 列：信号值（float）  
  - 第 1 列：逐采样点标签（int，0=background, 1=spindle，可扩展多类）

- **格式 B**：同一目录下两个文件  
  - `signal.npy`：形状 `[N]`，信号  
  - `label.npy`：形状 `[N]`，逐点标签  

若为其它格式，程序会报错并提示当前读到的 shape。

目录约定（可选）：在 `root_path` 下放 `train/`、`val/`、`test/` 子目录，或单文件 `train.npy`、`val.npy`、`test.npy`。也可通过 yaml 的 `split_files` 指定每行一个路径的列表文件。

## 2) 训练命令（可复制）

单卡、不依赖 DDP：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_units_baseline \
  --patch_len 16 \
  --stride 16 \
  --train_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --seg_loss ce_dice \
  --class_weight auto
```

参数说明：
- `--seg_loss`：`ce` | `ce_dice`（默认）| `focal`
- `--class_weight`：`auto`（按训练集频次）| `manual`（需在 yaml/args 中提供 `class_weights`）

## 3) 测试命令（可复制）

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointseg.yaml \
  --is_training 0 \
  --model UniTS \
  --model_id dreams_pointseg_units_baseline \
  --load_ckpt checkpoints/dreams_pointseg_units_baseline_UniTS_All_ftM_dm512_el2_test_0/best.pth
```

若未指定 `--load_ckpt`，默认使用 `checkpoints/<setting>/best.pth`，其中 `<setting>` 由 `task_name`、`model_id`、`model`、`data`、`d_model`、`e_layers`、`des` 等拼接（与训练时一致）。

## 4) 结果输出路径

- **权重**：`checkpoints/<setting>/best.pth`（按 val spindle-F1 保存最优）
- **日志**：`checkpoints/<setting>/finetune_output.log`
- **训练过程**：  
  - `results/<setting>/metrics.csv`：每 epoch 的 train_loss、val_acc、val_macro_f1、val_spindle_f1、val_event_f1、lr  
  - `results/<setting>/tensorboard/`：TensorBoard 事件（训练时写入）  
  - `results/<setting>/curves/`：训练曲线图（需训练后运行 `plot_curves.py` 生成，或见下方命令）  
- **指标与明细**：  
  - `results/<setting>/results.json`：逐点 + 事件级指标与配置摘要  
  - `results/<setting>/results.csv`：同上，表格形式  
  - `results/<setting>/per_sample_metrics.csv`：每个测试样本的 event_f1、mean_boundary_error_ms 等  
- **可视化**：`results/<setting>/vis/`  
  - `seg_<excerpt_id>_<idx>.png`：信号、GT、Pred prob、Pred mask、事件边界  
  - `saliency_<excerpt_id>_<idx>.png`：|grad| vs time 可解释性图  
- **定量总结**：`results/<setting>/analysis/`  
  - `confusion_matrix.png`：逐点混淆矩阵  
  - `per_class_metrics.png`：每类 event F1 条形图  
  - `boundary_error_hist.png`：每样本平均边界误差直方图  
  - `f1_vs_threshold.png`：spindle 类 F1-阈值曲线  
  - `analysis_summary.json`：推荐阈值、best_20/worst_20 样本 id 等  
- **案例集**：`results/<setting>/cases/best_20/`、`results/<setting>/cases/worst_20/`（从 vis 复制的 best/worst 样本图）

测试阶段会自动保存至少 20 个样本图及对应 saliency，并自动运行分析脚本生成 analysis 与 cases。

### 训练过程可视化命令

- **启动 TensorBoard**（训练已写入 `results/<setting>/tensorboard` 后）：  
```bash
tensorboard --logdir=results/<setting>/tensorboard --port=6006
```
  浏览器打开 `http://localhost:6006` 查看 loss、val F1、lr 等曲线。

- **生成训练曲线 PNG**（读取 `results/<setting>/metrics.csv`，输出到 `results/<setting>/curves/`）：  
```bash
python tools/plot_curves.py --setting <setting>
```
  示例：`python tools/plot_curves.py --setting point_segmentation_dreams_project_pointseg_units_baseline_UniTS_All_ftM_dm512_el2_test_0`

### 单独重新生成测试分析图与案例集

若测试时未生成 analysis/cases，可事后运行：  
```bash
python tools/analyze_results.py --results_dir results/<setting> --num_classes 2 --fs 256
```

## 5) stride / patch_len 对边界精度的影响

- `patch_len` 与 `stride` 决定 encoder 的 token 数 L，SegHead 用插值将 L 上采样回 T，因此边界精度不会受 stride 的整数对齐限制。
- 若采用“每 patch 一个 logit 再重复到 patch 范围”的 baseline 方案，边界会按 patch 对齐，精度受 stride 影响；当前实现为插值上采样，文档中会注明若改用 baseline 则存在该限制。

---

**最小端到端复现流程**：  
1）确保已有窗口级 spindle 数据：`data/dreams_units/{train,val,test}.npz`。  
2）运行脚本生成统一项目格式逐点数据：  
```bash
python scripts/convert_npz_spindle_to_pointseg.py
```
3）使用新的 yaml 训练：  
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_project_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_project_pointseg_units_baseline \
  --patch_len 16 \
  --stride 16 \
  --train_epochs 2 \
  --batch_size 16
```
4）使用相同 setting 进行测试并生成结果与可视化，在 `results/<setting>/` 下查看 `results.json`、`results.csv` 和 `vis/*.png`。  
5）未来接入多类标注时，在 `data/dreams_project/raw/annotations/<class_name>/*.txt` 中放置标注文件（每行形如 `excerpt_id start_sec end_sec`），然后运行：  
```bash
python scripts/build_multiclass_labels_from_annotations.py
```
根据需要扩展 `parse_events` 与逐点 label 写回逻辑，即可自动构建多类逐点 label 并重新训练。
