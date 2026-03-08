#!/bin/bash
# DREAMS 逐点分割 - 折中版：避免全 0（FIXED_1）与全 1（FIXED_2）两种塌缩
# ============================================
# FIXED_1：beta=0.8 过重 → 后期塌缩成全 0，spindle_f1 归零
# FIXED_2：seg_pos_weight=2 与 auto 叠加 → 从 epoch1 就全预测 1，acc 崩
# 本版：
#   1. Tversky 保持 0.5/0.5（平衡），不加重正类导致全 1
#   2. 不加 seg_pos_weight（仅用 class_weight auto，约 11x），避免 FIXED_2 式塌缩
#   3. lr=0.0001、early_stop_patience=5 保留

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py \
  --task_name point_segmentation \
  --task_data_config_path data_provider/dreams_pointwise_pointseg.yaml \
  --is_training 1 \
  --model UniTS \
  --model_id dreams_pointseg_11-16hz_p64_s32_FIXED_3 \
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
