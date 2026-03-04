# UniTS → DREAMS 纺锤波逐采样点检测 改造计划（仅计划，无代码修改）

## 1) 现有 data_provider 数据流

### 1.1 数据从哪里读

- **UEA 分类（当前 DREAMS 使用）**
  - `data_provider/data_factory.py`：`config['data'] == 'UEA'` 时使用 `UEAloader`。
  - `UEAloader`（`data_provider/data_loader.py`）：`root_path` 来自 config（如 `dataset/DREAMS`），通过 `load_all` → `load_single` 用 `sktime.datasets.load_from_tsfile_to_dataframe` 读取 `.ts` 文件（如 `DREAMS_TRAIN.ts`）。
  - `.ts` 来源：`scripts/npz_to_ts.py` 从 `data/dreams_units/train.npz`（由 `prepare_dreams.py` 生成）写出。当前 `prepare_dreams.py` 做的是**窗口级**二分类：每个窗口 256 点，label 为「窗口内是否含纺锤波」0/1。

- **其他任务简要**
  - **Forecast**：`Dataset_ETT_*` / `Dataset_Custom` 等从 CSV 的 `root_path` + `data_path` 读入，按 border 切 train/val/test。
  - **Imputation**：同 forecast 类 Data，`__getitem__` 返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)。
  - **Anomaly**：`PSMSegLoader` 等从 `root_path` 下 npy/csv 读 train/test 与 `test_label`；按 `win_size` 滑动窗口取段。

### 1.2 __getitem__ 返回什么、label 形状

- **UEA（当前 DREAMS）**
  - `__getitem__(ind)` 返回：`(feature_tensor, label_tensor)`。
  - `feature_tensor`：`(seq_len, feat_dim)`，即 `(T, C)`，已 instance_norm，来自 `feature_df.loc[self.all_IDs[ind]].values`。
  - `label_tensor`：`(num_labels,)`，即 `(1,)` 或标量等价，**整型类别索引**（0/1 窗口级标签）。

- **Collate（UEA）**  
  `data_provider/uea.py` 的 `collate_fn`：
  - 输入：list of `(X, y)`，X 为 `(seq_len, feat_dim)`，y 为 `(num_labels,)`。
  - 输出：
    - `X`: `(batch_size, padded_length, feat_dim)`，即 `(B, T, C)`；
    - `targets`: `(batch_size, num_labels)`，即 `(B, 1)` 或 `(B,)`；
    - `padding_masks`: `(batch_size, padded_length)`，bool，1 表示有效时间步。

- **Forecast（ETT/Custom）**
  - `__getitem__` 返回：`(seq_x, seq_y, seq_x_mark, seq_y_mark)`。
  - `seq_x`: `(seq_len, C)`；`seq_y`: `(label_len+pred_len, C)`。无单独「label」；监督信号是未来序列。

- **Anomaly（PSM/MSL/SMD/SWAT 等）**
  - `__getitem__` 返回：`(data, label)`。
  - `data`: `(win_size, num_features)`，即 `(T, D)`；
  - `label`: `(win_size,)`，**逐点** 0/1（当前实现里 train/val 用占位 `test_labels[0:win_size]`，仅 test 用真实逐点标签）。

---

## 2) UniTS 模型 forward 当前输出 shape（训练/评估）

- 约定：**输入 `x_enc` 为 (B, T, D)**。`tokenize` 里 `x.mean(1, keepdim=True)` 即对时间维 dim=1 求均。

- **long_term_forecast / short_term_forecast**  
  - 训练：`dec_out` = `forecast(...)` → **[B, pred_len, D]**；exp 里取 `outputs[:, -pred_len:, f_dim:]` 与 `batch_y[:, -pred_len:, f_dim:]` 做 MSE。  
  - 评估：同 shape **[B, pred_len, D]**，与 `batch_y` 对齐算 MSE/MAE。

- **classification**  
  - 训练：`dec_out` = `classification(...)` → ** [B, N]**（N=num_class），与 `label` (B,) 做 CE。  
  - 评估：**[B, N]**，softmax 后 argmax 得 (B,) 与 label 算 accuracy。

- **imputation**  
  - 训练/评估：`dec_out` = `imputation(...)` → **[B, L, D]**（L=seq_len），在 `mask==0` 位置与 `batch_x` 算 MSE。

- **anomaly_detection**  
  - 训练/评估：`dec_out` = `anomaly_detection(...)` → ** [B, L, D]**（L=seq_len），为**重建**；exp 里用 MSE(batch_x, outputs) 得到 (B, L) 的 score，再在 test 上阈值得到逐点二值 pred，与 gt 做 F-score 等。  
  - 即：模型输出是重建序列，**不是**逐点 logits；逐点决策在 exp 里用重建误差做。

---

## 3) 需要改动的文件（按优先级）

1. **P0 - 数据与任务定义**
   - `prepare_dreams.py`  
     改为产出**逐采样点**标签：每个样本为固定长度 T 的 EEG 段，label 为 `(T,)` 的 0/1，并写入 npz（或兼容格式），且需与后续 DataLoader 约定一致（见下）。
   - `data_provider/dreams.yaml`  
     新增或改为「pointwise_binary_segmentation」之类 task_name，并约定 seq_len、enc_in=1、无 pred_len/num_class，仅 DREAMS 用。
   - `data_provider/data_factory.py`  
     为 DREAMS 逐点任务分支：选择「DREAMS」或新 data 类型，构建专用 DataLoader（见下），返回 (X, label_pointwise, padding_mask) 等。
   - 新增 `data_provider/dreams_pointwise.py`（或扩展现有 DREAMS 加载逻辑）  
     实现从 npz/ts 读入「固定 T 的序列 + (T,) 的 0/1 label」；`__getitem__` 返回 (x, y_pointwise)；x 形状 (T, 1) 或 (1, T) 需与 4) 统一。

2. **P0 - 模型**
   - `models/UniTS.py`  
     - 在 `forward` 中增加分支 `task_name == 'pointwise_binary_segmentation'`（或你定的名字），调用新方法 `pointwise_segmentation(x_enc, x_mark_enc, task_id)`。  
     - 实现 `pointwise_segmentation`：复用现有 tokenize → prepare_prompt（可用与 anomaly 类似的 prompt，无 cls/category）→ backbone；**不再**用 `ForecastHead`/`CLSHead`，改为新增 **SegmentationHead**：从 backbone 的 token 序列得到**与输入时间步一一对应的 logits**，输出 shape 见 4)。  
     - `__init__` 里对该 task 注册 prompt_tokens/mask_tokens（若沿用现有设计）、以及 `cls_nums` 或等价配置（如 seq_len、是否 padding）供 head 知道原长 T。
   - 新增 head：在 `models/UniTS.py`（或单独模块再导入）中实现 **SegmentationHead**，输入 backbone 输出 `(B, n_vars, num_tokens, d_model)`，输出 (B, T) 或 (B, T, 2)，T 为原始时间步数，需处理 token 数 → T 的对齐（见 5)。

3. **P0 - 训练与评估**
   - `exp/exp_sup.py`  
     - `_select_criterion`：对 pointwise_binary_segmentation 选 BCE 或 CE（若用 [B,T,2]）。  
     - `get_task_data_config_list` / 配置解析：支持新 task_name。  
     - `train_one_epoch` 分支：调用 `train_pointwise_segmentation(model, sample, criterion, config, task_id)`。  
     - 实现 `train_pointwise_segmentation`：从 this_batch 解出 batch_x、label (B, T)，前向得到 logits (B, T) 或 (B, T, 2)，在有效位置（可选 padding_mask）上算 loss。  
     - `test` 分支：调用 `test_pointwise_segmentation(...)`。  
     - 实现 `test_pointwise_segmentation`：逐 batch 前向，收集 logits 与 label，在有效时间步上算 accuracy / F1 / IoU 等，并支持 `gather_tensors_from_all_gpus`。
   - `run.py`  
     若需为新任务加命令行参数（如 `--task_name` 默认或 `--loss pointwise_bce`）可在此加；非必须。

4. **P1 - 配置与脚本**
   - `data_provider/dreams.yaml`  
     已列在 P0；确保 `seq_len`、`enc_in`、`root_path`、数据格式与 `dreams_pointwise` 一致。
   - `scripts/npz_to_ts.py`（可选）  
     若继续用 .ts 作为中间格式，需支持「每行一个样本 + 逐点标签」或改为仅从 npz 读；若全用 npz 可不再写 .ts，视数据管线而定。

5. **P2 - 工具与评估**
   - `utils/metrics.py`  
     若有现成 `metric` 只做回归，可新增 pointwise 分类指标（如 binary_f1, iou）或单独函数供 exp_sup 调用。
   - `utils/losses.py`  
     若有 BCE/CE 封装则复用，否则在 exp 里用 `F.binary_cross_entropy_with_logits` / `CrossEntropyLoss` 即可。

6. **不改或仅极小改**
   - `models/UniTS_zeroshot.py`：若不做 zeroshot DREAMS，可暂不改；若做则需同步 task 分支与 head。
   - `exp/exp_pretrain.py`：预训练任务可暂不包含 pointwise；若包含需加数据接口与 loss。
   - `utils/dataloader.py`（BalancedDataLoaderIterator）：无需改，只要新 DataLoader 返回的 batch 结构被 exp 正确解包即可。

---

## 4) 逐点检测的目标 I/O 定义

- **输入**
  - 统一为 ** [B, T, 1]**（B=batch_size, T=序列长度, 1=单通道 EEG）。  
  - 理由：与现有 UniTS 约定一致（tokenize 中时间维为 dim=1），且与 UEA collate `(B, padded_length, feat_dim)`、anomaly 的 `(B, win_size, D)` 一致，便于复用 data_factory 与 tokenize。

- **输出 logits**
  - 方案 A：** [B, T]**，每个时间点一个标量 logit（二类用 sigmoid）。  
  - 方案 B：** [B, T, 2]**，每个时间点 2 维 logit（二类用 softmax + CE）。  
  - 建议采用 ** [B, T]**，loss 用 `F.binary_cross_entropy_with_logits`，实现简单且与 label [B, T] 一一对应。

- **label**
  - ** [B, T]**，dtype 可 long 或 float；值为 0 或 1（非纺锤 / 纺锤）。  
  - 若存在 padding（变长序列），用 padding_mask (B, T) 在 loss 和评估时屏蔽，不参与梯度与指标。

---

## 5) patch_len / stride 对时间分辨率的影响与边界对齐

- **当前机制**
  - `tokenize` 中：`x` 为 (B, T, D)，先 `x.permute(0, 2, 1)` 成 (B, D, T)，再对 T 做 `unfold(size=patch_len, step=stride)` → 得到 **num_tokens = (T - patch_len) // stride + 1**（若 T 先被 pad 到可整除 patch_len）。  
  - backbone 输出为 (B, n_vars, num_tokens, d_model)。ForecastHead 用 `fold(kernel_size=(patch_len,1), stride=(stride,1))` 从 token 序列还原到时间维，得到长度为 **(num_tokens - 1) * stride + patch_len**，与 pred_len 对齐。

- **对逐点检测的影响**
  - 若 patch_len=16、stride=8：1 个 token 对应 16 个时间点，stride 8 则 token 间重叠，**时间分辨率是 patch 级别**，不是逐点。  
  - 若 patch_len=1、stride=1：num_tokens=T，**时间分辨率与采样点一致**，可直接做逐点 logits；但 token 序列变长，计算量与显存大增。

- **推荐策略**
  - **方案 A（保持 patch_len/stride > 1）**  
    - backbone 输出 num_tokens 个 token。SegmentationHead：每个 token 预测 1 个 logit（或 2 维），再通过**插值/上采样**从 num_tokens 恢复到 T。  
    - 对齐方式：将 token 索引 i 映射到时间区间 [i*stride, i*stride+patch_len)，对该区间内的时间点可重复同一 logit，或线性插值；若 T 与 (num_tokens-1)*stride+patch_len 不一致，对输出做插值到 T（如 `interpolate(..., size=T)`），再与 label [B, T] 对齐。  
  - **方案 B（patch_len=1, stride=1）**  
    - 仅当 T 较小或可接受显存时使用；无需插值，head 直接输出 (B, T) 或 (B, T, 2)。

- **边界与 padding**
  - tokenize 里已有：若 `T % patch_len != 0`，会在右侧 pad 到整除，并返回 `padding` 长度。  
  - 在 SegmentationHead 和 loss 中：输出先还原到「未 pad 的序列长度」或直接还原到 T（若 T 固定），只对前 T 个时间点与 label 计算 loss；或对 padding 位置用 mask 屏蔽，避免 pad 位置参与 loss 和指标。  
  - 若使用插值还原到 T：建议统一「有效长度」为 config 的 seq_len（即 T），pad 部分不参与 loss。

---

## 6) 小结

- **数据**：从「窗口级 0/1」改为「固定长度 T 的序列 + [T] 的 0/1 标签」；新 DataLoader 返回 (X, y_pointwise) 或 (X, y_pointwise, padding_mask)，X 为 [B, T, 1]。  
- **模型**：新增 task 分支与 SegmentationHead，forward 输出 [B, T]（或 [B, T, 2]），与 label [B, T] 对齐。  
- **训练/评估**：exp_sup 中新增 pointwise 的 train/test 与 criterion，指标为 F1/IoU/accuracy 等。  
- **分辨率与边界**：patch_len/stride>1 时用 token→时间插值/上采样到 T 并做边界与 padding 的 mask；或 patch_len=stride=1 直接逐点。

本计划不包含任何代码 diff，仅作改造路线与接口约定。
