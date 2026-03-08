#!/bin/bash
# DREAMS 逐点分割 - 方案A（修订）：无预训练 + Focal，避免全预测 cls1
# ============================================
# 策略：
#   1. 禁用预训练，从头学习 DREAMS 特征
#   2. Focal loss 应对不平衡；bg_keep_prob=0.5 保留足够背景信号，避免塌缩成全 1
#   3. pos_window_weight=2 适度偏向含纺锤波窗口，不极端过采样
#   4. seg_pos_weight=2 适度强化正类，配合 class_weight auto
#   5. 学习率 0.0005 无预训练时更稳

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_SchemeA_nopretrain_focal \
  --patch_len 64 \
  --stride 32 \
  --train_epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --seg_loss focal \
  --focal_gamma 3.5 \
  --class_weight auto \
  --seg_pos_weight 2.0 \
  --bg_keep_prob 0.5 \
  --pointseg_weighted_sampling 1 \
  --pointseg_pos_window_weight 2.0 \
  --lradj cosine \
  --d_model 128 \
  --prompt_num 10 \
  --pointseg_best_metric spindle_event_combo \
  --pointseg_best_spindle_weight 0.7 \
  --pointseg_best_event_weight 0.3
