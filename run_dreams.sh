#!/bin/bash
# DREAMS 逐点分割修复版本：针对极度不平衡类别
# ============================================
# 修改说明：
# 1. Tversky参数反向：alpha=0.2(低FN惩罚), beta=0.8(高FP惩罚) - 适应稀有类别
# 2. 使用focal_tversky增加梯度信号
# 3. 增加class_weight权重和focal_gamma
# 4. 调整学习率和warmup策略

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_11-16hz_p64_s32_FIXED_1 \
  --patch_len 64 \
  --stride 32 \
  --train_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.0002 \
  --seg_loss focal_tversky \
  --tversky_alpha 0.2 \
  --tversky_beta 0.8 \
  --focal_tversky_gamma 2.5 \
  --class_weight auto \
  --bg_keep_prob 1.0 \
  --pointseg_weighted_sampling 1 \
  --pointseg_pos_window_weight 3.0 \
  --lradj cosine \
  --pretrained_weight /data/YanJingjing/projects/UniTS_new/checkpoints/units_x128_pretrain_checkpoint.pth \
  --d_model 128 \
  --prompt_num 10 \
  --pointseg_best_metric spindle_event_combo \
  --pointseg_best_spindle_weight 0.7 \
  --pointseg_best_event_weight 0.3
