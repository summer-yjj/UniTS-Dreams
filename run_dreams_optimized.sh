#!/bin/bash
# DREAMS 逐点分割 - 防塌缩 + 稳 F1 版（基于 run_dreams 结果优化）
# ============================================
# 现象：best 在 epoch 2，之后 pred_cls1 一路掉到接近 0，spindle_f1 归零，train loss 在 11–12 暴增
# 调整：
#   1. Tversky 改为 alpha=0.5 beta=0.5，避免 beta=0.8 把模型推向“几乎不预测 1”
#   2. 加 seg_pos_weight=2，强化正类，减缓塌缩
#   3. lr 0.0002->0.0001，减缓后期塌缩、给更稳的收敛
#   4. early_stop_patience=5，best 已常在 2–5 epoch，早停省时

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_11-16hz_p64_s32_FIXED_2 \
  --patch_len 64 \
  --stride 32 \
  --train_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --seg_loss focal_tversky \
  --tversky_alpha 0.5 \
  --tversky_beta 0.5 \
  --focal_tversky_gamma 2.5 \
  --class_weight auto \
  --seg_pos_weight 2.0 \
  --bg_keep_prob 1.0 \
  --pointseg_weighted_sampling 1 \
  --pointseg_pos_window_weight 3.0 \
  --lradj cosine \
  --early_stop_patience 5 \
  --pretrained_weight /data/YanJingjing/projects/UniTS_new/checkpoints/units_x128_pretrain_checkpoint.pth \
  --d_model 128 \
  --prompt_num 10 \
  --pointseg_best_metric spindle_event_combo \
  --pointseg_best_spindle_weight 0.7 \
  --pointseg_best_event_weight 0.3
