# 🚀 DREAMS纺锤波分割 - 立即行动方案

## 问题症状
```
✗ 验证准确率: 3.85%（几乎随机）
✗ 模型Epoch 1就开始pred_cls1=100%
✗ 所有metric都接近0（完全不学习）
✓ 训练损失在下降（欺骗性的虚假优化）
```

---

## 根本原因（已诊断）

### 🔴 最严重：Tversky参数反向
```
当前:    alpha=0.6, beta=0.4
结果:    模型倾向漏检正类 → 过度预测
应该:    alpha=0.2, beta=0.8  (让模型学会保守预测)
```

### 🔴 次严重：Tversky损失未应用class_weight
```
CE损失有权重：weight=[1.0, 41.18]
Tversky损失无权重：所有类平等对待（错！）
导致：权重无法阻止模型坍塌
```

### 🟠 代码缺陷：seg_pos_weight参数未被使用
```
脚本中设置: --seg_pos_weight 2.0
代码中：这个参数被完全忽视了 ❌
```

---

## ✅ 修复清单（已完成部分）

### ✅ 代码修改完成
- [x] 改进 `utils/seg_losses.py` 中的 `_tversky_loss` → 支持class_weights
- [x] 更新 `compute_seg_loss` → Tversky调用时传入weights
- [x] 创建改进脚本：`run_pointseg_fixed.sh`
- [x] 创建诊断文档：`POINTSEG_FIX_GUIDE.md`
- [x] 创建监控脚本：`monitor_pointseg_training.py`

### 🟡 你需要做的

#### Step 1: 运行改进版本（立即）
```bash
cd c:\Users\严晶晶\Desktop\UniTS_cursor
bash run_pointseg_fixed.sh
```

**监控第1-5个epoch这个关键指标**：
- 查看 `pred_cls1` 是否从 1.0 下降？
  - 如果下降到 0.5-0.9 → 有效！继续
  - 如果仍然 ≈ 1.0 → 需要进一步调整

#### Step 2: 实时监控（可选）
```bash
python monitor_pointseg_training.py
```

#### Step 3: 根据结果调整
如果 Epoch 5 后 pred_cls1 仍然 ≈ 1.0：

**选项A** - 增加beta参数：
```bash
--tversky_beta 0.9           # 从0.8改为0.9
--focal_tversky_gamma 3.0    # 从2.5改为3.0
```

**选项B** - 改用CE+Focal并增加gamma：
```bash
--seg_loss focal
--focal_gamma 4.0            # 很强的focal
```

**选项C** - 下采样背景样本：
```bash
--bg_keep_prob 0.1           # 只保留10%背景类
```

---

## 📊 预期效果

### 修复前（当前）
```
Epoch 1:  train_loss=0.909, val_acc=0.0385, pred_cls1=1.0000 ❌
Epoch 30: train_loss=0.899, val_acc=0.0200, pred_cls1=1.0000 ❌
结论：模型根本没学习，虚假优化
```

### 修复后（预期）
```
Epoch 1:  train_loss=0.95, val_acc=0.04, pred_cls1=0.95   (还可以)
Epoch 5:  train_loss=0.85, val_acc=0.15, pred_cls1=0.60   ✅ 学习信号！
Epoch 10: train_loss=0.70, val_acc=0.25, pred_cls1=0.30   ✅ 继续改善
Epoch 30: train_loss=0.45, val_acc=0.45, pred_cls1=0.15   ✅ 好转
```

---

## 🔧 关键参数释义

| 参数 | 旧值 | 新值 | 作用 | 原因 |
|------|------|------|------|------|
| `tversky_alpha` | 0.6 | 0.2 | 降低假阴性惩罚 | 稀有类别应保守 |
| `tversky_beta` | 0.4 | 0.8 | 增加假阳性惩罚 | 防止过度预测 |
| `focal_tversky_gamma` | N/A | 2.5 | Focal加权强度 | 关注难样本 |
| `pointseg_pos_window_weight` | 1.5 | 3.0 | 正类样本重要性 | 加权采样强度 |
| `learning_rate` | 0.0001 | 0.0002 | 学习速率 | 帮助跳出坏局部最优 |
| `train_epochs` | 30 | 50 | 训练轮数 | 给模型更多恢复时间 |

---

## ⚙️ 验证修复是否生效

在 `run.py` 输出中查找这些日志：

### ✅ 好的信号：
```
Epoch 1 train_loss 0.950000  # 损失更高（因为权重增加了）
DEBUG vali: pos_rate pred_cls1 0.950000  # 从1.0变小了！
Class weights (ratio): [1.00, 41.18]     # 权重被计算了
```

### ❌ 坏的信号：
```
Epoch 1 train_loss 0.909305  # 损失没变（权重未生效）
DEBUG vali: pos_rate pred_cls1 1.000000  # 仍然预测全1
```

---

## 💡 如果还是不行

### 可能问题1：预训练模型损伤
```bash
# 尝试从头训练（不用pretrain）
--pretrained_weight ""
```

### 可能问题2：数据质量问题
```python
# 检查数据分布
python -c "
import numpy as np
y = np.load('data/DREAMS_pointwise_labels.npz')  # 根据实际路径调整
print('Class counts:', np.bincount(y.flatten()))
print('Ratio:', np.bincount(y.flatten())[1] / len(y.flatten()))
"
```

### 可能问题3：模型容量太小
```bash
--d_model 256    # 从128增加到256
--prompt_num 20  # 从10增加到20
```

---

## 📞 调试建议

### 添加更多诊断日志
编辑 `exp/exp_pointseg.py`，在训练循环中添加：
```python
# 在每个batch后添加
if batch_idx % 100 == 0:
    pred_prob = F.softmax(logits, dim=1)
    pred_cls1_ratio = (pred_prob[:, 1] > 0.5).float().mean().item()
    print(f"  Batch {batch_idx}: pred_cls1_ratio={pred_cls1_ratio:.4f}, loss={loss.item():.6f}")
```

### 保存预测概率分布
```python
# 在validation时添加
prob_dist_cls1 = F.softmax(logits, dim=1)[:, 1]
print(f"Prob distribution - min:{prob_dist_cls1.min():.4f}, max:{prob_dist_cls1.max():.4f}, mean:{prob_dist_cls1.mean():.4f}")
```

---

## ✨ 总结

**当前问题根源**：Tversky损失的alpha/beta倒反 + 缺少class_weight应用

**已修复项**：
- ✅ seg_losses.py 中Tversky现在支持class_weights
- ✅ 提供了改进的脚本参数配置

**你要做的**：
1. 运行 `bash run_pointseg_fixed.sh`
2. 观察Epoch 1-5的 pred_cls1 是否下降
3. 根据效果微调参数

**预期改进**：从 3.85% accuracy → 40-60% (在DREAMS比较难的情况下)

---

💬 Good luck! 这套修复应该能让模型从"完全不学习"变成"正常学习"。
