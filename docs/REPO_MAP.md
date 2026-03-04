# UniTS 仓库 Repo Map（DREAMS 逐点分割用）

## 1) 目录结构（根目录下，L3 内）

```
UniTS/
├── run.py                 # 训练/测试主入口（--task_name, --is_training）
├── run_pretrain.py        # 预训练入口
├── prepare_dreams.py      # DREAMS 窗口级数据准备（npz/ts）
├── data_provider/
│   ├── data_factory.py    # data_dict + data_provider(args, config, flag)
│   ├── data_loader.py     # Dataset_ETT_*, Dataset_Custom, UEAloader, *SegLoader, GLUONTSDataset
│   ├── uea.py             # collate_fn（分类 pad）, Normalizer
│   ├── dreams.yaml        # DREAMS 分类配置（UEA/seq_len 256）
│   └── *.yaml             # 各任务 data 配置
├── exp/
│   ├── exp_sup.py         # Exp_All_Task：train/val/test，按 task_name 分支
│   └── exp_pretrain.py    # 预训练 Exp
├── models/
│   ├── UniTS.py           # Model：tokenize, prepare_prompt, backbone, forecast/classification/... , forward(task_name)
│   └── UniTS_zeroshot.py  # zeroshot 用
├── utils/
│   ├── losses.py          # mape/mase/smape, UnifiedMaskRecLoss
│   ├── metrics.py         # metric(pred,true) 回归指标
│   ├── tools.py           # cal_accuracy, adjust_learning_rate 等
│   └── ddp.py             # 分布式 is_main_process 等
├── data/dreams_units/      # train/val/test.npz（现有）
├── dataset/DREAMS/        # .ts 文件（UEA 分类用）
├── checkpoints/           # 权重保存
└── scripts/               # 各任务 shell
```

## 2) 关键 ripgrep 结果摘要

| 搜索内容 | 主要文件 |
|----------|----------|
| data_provider / data_dict / data_provider( | data_provider/data_factory.py |
| Dataset / UEAloader / dreams | data_provider/data_loader.py, data_provider/uea.py, data_provider/dreams.yaml |
| UniTS 模型定义 | models/UniTS.py（class Model） |
| task_name 分支 | run.py（无按 task_name 分支，统一 Exp_All_Task_SUP）；exp/exp_sup.py 内按 task_name 选 train_*/test_*；models/UniTS.py forward 内按 task_name 调用 forecast/classification/... |
| run.py / run_pretrain | run.py, run_pretrain.py |

## 3) Repo Map（文件路径 → 作用，一行）

- **数据入口**
  - `data_provider/data_factory.py` → data_dict 映射 dataset 名到 Dataset 类；data_provider() 根据 config 构建 DataLoader（含 classification/anomaly_detection/其他分支）。
  - `data_provider/data_loader.py` → 定义 Dataset_ETT_hour/minute、Dataset_Custom、UEAloader、PSM/MSL/SMAP/SMD/SWAT SegLoader、GLUONTSDataset。
  - `data_provider/uea.py` → 分类用 collate_fn（pad 到 max_len）、Normalizer、padding_mask。
  - `data_provider/dreams.yaml` → DREAMS 分类任务配置（task_name: classification, data: UEA, root_path: dataset/DREAMS）。

- **训练入口**
  - `run.py` → 解析 args（task_name, is_training, model_id, task_data_config_path 等），init_distributed_mode，固定 seed，创建 Exp = Exp_All_Task_SUP，调用 exp.train(setting) 或 exp.test(setting)。
  - `run_pretrain.py` → 预训练入口，使用 exp_pretrain。

- **模型**
  - `models/UniTS.py` → UniTS Model：tokenize（patch 化）、prepare_prompt、backbone、ForecastHead/CLSHead、forecast/classification/imputation/anomaly_detection；forward(x_enc, ..., task_name) 按 task_name 分发。

- **Loss / 指标 / 工具**
  - `utils/losses.py` → mape_loss, mase_loss, smape_loss, UnifiedMaskRecLoss（预训练）。
  - `utils/metrics.py` → metric(pred, true) 回归 MAE/MSE/RMSE/MAPE/MSPE。
  - `utils/tools.py` → cal_accuracy(y_pred, y_true)、adjust_learning_rate、NativeScaler 等。

- **实验逻辑**
  - `exp/exp_sup.py` → Exp_All_Task：_build_model、_get_data（data_provider）、_select_criterion（按 task_name 选 CE/MSE）、train_one_epoch（按 task_name 调用 train_long_term_forecast / train_classification / …）、test（按 task_name 调用 test_classification / test_long_term_forecast / …）。

以上为“DREAMS 微观事件逐点检测与定位”增量实现所依赖的 Repo Map。
