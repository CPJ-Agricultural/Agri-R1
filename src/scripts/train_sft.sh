#!/bin/bash
# ============================================================================
# SFT Training for Agricultural VQA
# Dataset: ~845k samples, 1 epoch
# Hardware: 4x GPUs with DeepSpeed Zero-3
# ============================================================================

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate agri-r1

export CUDA_VISIBLE_DEVICES=0,1,2,3

# WandB Configuration
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="agri-r1"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"
export WANDB_NAME="sft_agri_1epoch"

# Suppress warnings
export PYTHONWARNINGS="ignore::UserWarning"

# DeepSpeed Configuration
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Python Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# ============================================================================
# Training Configuration
# ============================================================================
PER_DEVICE_BS=24
GRAD_ACCUM=2
NUM_GPUS=4
EFFECTIVE_BS=$((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))

MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="/path/to/sft_dataset"
OUTPUT_DIR="/path/to/outputs/sft_agri"
DS_CONFIG="${REPO_ROOT}/src/r1-v/configs/ds_zero3.json"

echo "Starting SFT training..."
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Batch size: ${PER_DEVICE_BS} × ${GRAD_ACCUM} × ${NUM_GPUS} = ${EFFECTIVE_BS}"
echo "Output: ${OUTPUT_DIR}"

# TODO: Need to implement agri_sft.py training script
# This will use standard HuggingFace Trainer with agricultural VQA data

python "${REPO_ROOT}/src/r1-v/src/open_r1/sft.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_PATH}" \
  --dataset_train_split train \
  --dataset_test_split validation \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size ${PER_DEVICE_BS} \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --num_train_epochs 1 \
  --output_dir "${OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --bf16 true \
  --logging_first_step true \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 4 \
  --gradient_checkpointing true \
  --deepspeed "${DS_CONFIG}" \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --max_grad_norm 1.0 \
  --report_to wandb \
  --seed 42

echo "Training completed!"
