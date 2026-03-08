#!/bin/bash
# DREAMS 逐点分割 - 回归基线（FIXED_4）：破“全预测 1”的僵局
# ============================================
# FIXED_1/2/3 都在往“全 1”或“全 0”塌，可能原因：
#   - focal_tversky + 极度不平衡 → 梯度偏向“预测 1”
#   - patch_len=64 导致 patch_embedding 不加载预训练，表示能力差
#   - pos_window_weight=3 让每个 batch 里正类比例偏高
# 本版做减法，尽量只动必要项：
#   1. seg_loss 改为 ce_dice（无 Tversky/focal），梯度更稳
#   2. patch_len=16 stride=16，与预训练一致，加载 patch_embedding
#   3. pointseg_pos_window_weight=1（不再过采样含纺锤波窗口）
#   4. 仍用 class_weight auto，不加 seg_pos_weight

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_baseline_p16_FIXED_4 \
  --patch_len 16 \
  --stride 16 \
  --train_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --seg_loss ce_dice \
  --class_weight auto \
  --bg_keep_prob 1.0 \
  --pointseg_weighted_sampling 1 \
  --pointseg_pos_window_weight 1.0 \
  --lradj cosine \
  --early_stop_patience 8 \
  --pretrained_weight /data/YanJingjing/projects/UniTS_new/checkpoints/units_x128_pretrain_checkpoint.pth \
  --d_model 128 \
  --prompt_num 10 \
  --pointseg_best_metric spindle_event_combo \
  --pointseg_best_spindle_weight 0.7 \
  --pointseg_best_event_weight 0.3
