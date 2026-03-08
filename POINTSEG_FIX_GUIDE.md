# DREAMS纺锤波分割问题诊断与解决方案

## 📊 问题诊断

### 核心问题：模型竟然在第1个epoch就坍塌了
```
Epoch 1: pred_cls1 1.000000 (预测全部为正类)
         gt_cls1 0.038529    (实际正类仅3.85%)
→ 完全忽视了class weights！
```

### 根本原因分析

#### 1️⃣ **Tversky参数反向（最严重）**
当前设置：`alpha=0.6, beta=0.4`
```
Tversky_IoU = TP / (TP + α·FN + β·FP + smooth)
当 α > β 时：更严惩"漏检" → 模型倾向过度预测正类 ❌
```

**对于稀有类别检测（class 1占3.85%）：**
- 应该 **减少对漏检的惩罚**（α应该小）
- 应该 **增加对误报的惩罚**（β应该大）

#### 2️⃣ **class_weight实现未充分考虑不平衡**
```python
# 当前实现（seg_losses.py line 95）：
weights = median_freq / counts
# 对于极度不平衡（41.18:1），weights = [1.0, 41.18]
# 但仍不足以阻止模型全部预测为1 ❌
```

#### 3️⃣ **seg_pos_weight=2.0未被应用**
搜索整个codebase，这个参数没有被实际使用在损失计算中。

#### 4️⃣ **Tversky不考虑样本权重**
不像CE损失，Tversky损失内部没有应用class_weight的地方（_tversky_loss函数）。

---

## ✅ 解决方案

### 方案1：参数调整（快速试）
**修改脚本参数**（见run_pointseg_fixed.sh）：

```bash
--seg_loss focal_tversky      # 从tversky改为focal_tversky
--tversky_alpha 0.2           # 从0.6改为0.2 ⬇️ 降低漏检惩罚
--tversky_beta 0.8            # 从0.4改为0.8 ⬆️ 增加误报惩罚
--focal_tversky_gamma 2.5     # 新增：使用focal weighting
--pointseg_pos_window_weight 3.0  # 从1.5改为3.0 ⬆️
--learning_rate 0.0002        # 从0.0001改为0.0002 ⬆️
--train_epochs 50             # 增加epoch数 - 早期可能需要更多迭代恢复
```

**期望效果**：
- Focal机制会加重错误分类样本的权重
- 减小α、增大β会让模型学会保守预测
- 更高的学习率帮助跳出坏的局部最优

---

### 方案2：代码级修复（根本解决）
需要修改 `utils/seg_losses.py` 中的 `_tversky_loss` 函数，使其支持class_weight：

```python
# 修改后应该这样：
def _tversky_loss(logits, y, num_classes, alpha=0.7, beta=0.3, 
                  smooth=1e-5, include_background=False, gamma=None, 
                  class_weights=None):  # ← 新增参数
    """Multi-class Tversky loss with optional class weighting."""
    probs = F.softmax(logits, dim=1)
    y_onehot = F.one_hot(y.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 2, 1).float()
    class_ids = range(num_classes) if include_background else range(1, num_classes)
    losses = []
    
    for c in class_ids:
        p = probs[:, c, :]
        t = y_onehot[:, c, :]
        tp = (p * t).sum()
        fn = ((1.0 - p) * t).sum()
        fp = (p * (1.0 - t)).sum()
        tv = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        
        loss_c = 1.0 - tv
        if gamma is not None and gamma > 0:
            focal_weight = (1.0 - tv) ** gamma
            loss_c = focal_weight * loss_c
        
        # 应用class weight
        if class_weights is not None:
            loss_c = loss_c * class_weights[c]
        
        losses.append(loss_c)
    
    if not losses:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return torch.stack(losses).mean()
```

然后在 `compute_seg_loss` 中调用时传入weights：
```python
if seg_loss == "focal_tversky":
    alpha = float(cfg.get("tversky_alpha", 0.7))
    beta = float(cfg.get("tversky_beta", 0.3))
    gamma = float(cfg.get("focal_tversky_gamma", 2.0))
    include_bg = bool(cfg.get("tversky_include_background", False))
    return _tversky_loss(logits, y, num_classes, alpha=alpha, beta=beta, 
                         gamma=gamma, include_background=include_bg, 
                         class_weights=weights)  # ← 传入权重
```

---

### 方案3：额外的改进方向

#### A. 使用更强的数据增强
```python
--pointseg_weighted_sampling 2.0  # 更强的重采样
```

#### B. 尝试不同的损失函数组合
```bash
--seg_loss ce           # 重新尝试纯CE + Focal
--focal_gamma 3.0       # 更强的Focal
```

或：
```bash
--seg_loss ce_dice      # CE + Dice的组合
```

#### C. 检查数据质量
```python
# 添加一些诊断代码到训练脚本
print(f"Class distribution: {np.bincount(y_train)} / total={len(y_train)}")
print(f"Batch negative/positive ratio: {(batch_y==0).sum()} / {(batch_y==1).sum()}")
```

---

## 🚀 建议执行顺序

### 第1步：快速尝试（15分钟）
```bash
bash run_pointseg_fixed.sh
# 观察第1-5个epoch的结果
# 如果pred_cls1逐渐降低（而不是直接=1.0），说明有效！
```

### 第2步：如果仍然不好（需要代码修改）
- 修改 `utils/seg_losses.py` 中的 `_tversky_loss` 和 `compute_seg_loss`
- 重新运行脚本

### 第3步：精细调参
根据第1-5个epoch的趋势调整：
- 如果pred_cls1开始下降但过度→增加beta
- 如果pred_cls1仍然≈1.0 →增加focal_gamma或学习率

---

## 📋 检查清单

在运行之前：
- [ ] 确认pretrained checkpoint路径正确
- [ ] 确认CUDA可用
- [ ] 备份现有checkpoint（以防万一）

运行中观察：
- [ ] Epoch 1-5 pred_cls1是否从1.0下降？（关键指标！）
- [ ] train_loss是否真的在下降（不是虚假稳定）？
- [ ] validation metrics是否有改善？

---

## 📞 如果还是不行

可能的次要原因：
1. **数据本身问题**：annotation质量差？极少数样本？
2. **预训练模型不匹配**：pretrain用的数据和DREAMS差异太大？
3. **模型容量**：d_model=128可能太小？尝试256

检查数据：
```python
python -c "
import yaml
with open('data_provider/dreams_pointwise_pointseg.yaml') as f:
    cfg = yaml.safe_load(f)
print('Data path:', cfg)
# 验证数据文件是否存在且不为空
"
```
