#!/usr/bin/env bash
set -euo pipefail

# ---- config (edit if you like) ----
LOG_DIR="${LOG_DIR:-./logs}"
PREP_LOG="$LOG_DIR/prepare_motor_$(date +%Y%m%d_%H%M%S).log"
TRAIN_LOG="$LOG_DIR/train_mamba_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "[$(date)] starting both jobs..."
echo "Logs:"
echo "  prepare -> $PREP_LOG"
echo "  train   -> $TRAIN_LOG"
echo

# ---- job 1: data prep (background) ----
# python prepare_motor/prepare_motor.py \
#   --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp \
#   --athena_path " " \
#   --num_bins 8 \
#   --loss_type tpp \
#   --num_threads 100 \
#   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader
#   >"$PREP_LOG" 2>&1 &

# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#   --num_processes 2 \
#   --mixed_precision bf16 \
#   --gpu_ids "0,2" \
#   pretrain_motor.py \
#   --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp \
#   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp/output_mamba \
#   --model mamba \
#   --loss_type tpp 

python pretrain_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp/output_mamba \
  --model mamba \
  --loss_type tpp \
  --n_layers 12

#  825, 8, 6100] torch.Size([825, 8, 6100])