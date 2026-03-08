#!/bin/bash
# DREAMS 逐点分割：11–16 Hz 带通滤波 + patch_len=64 stride=32
# 数据加载处已对 EEG 做 11–16 Hz 滤波（见 dreams_pointwise_pointseg.yaml）；
# 64 点 ≈ 0.25s @256Hz，可包含 3–4 个纺锤波周期；stride 32 约 50% 重叠，减轻纺锤波被从中间切断。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_11-16hz_p64_s32_ep30 \
  --patch_len 64 \
  --stride 32 \
  --train_epochs 30 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --seg_loss tversky \
  --tversky_alpha 0.6 \
  --tversky_beta 0.4 \
  --class_weight auto \
  --seg_pos_weight 2.0 \
  --bg_keep_prob 1.0 \
  --pointseg_weighted_sampling 1 \
  --pointseg_pos_window_weight 1.5 \
  --lradj cosine \
  --pretrained_weight /data/YanJingjing/projects/UniTS_new/checkpoints/units_x128_pretrain_checkpoint.pth \
  --d_model 128 \
  --prompt_num 10 \
  --pointseg_best_metric spindle_event_combo \
  --pointseg_best_spindle_weight 0.7 \
  --pointseg_best_event_weight 0.3
