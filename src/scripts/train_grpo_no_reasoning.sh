#!/bin/bash
# ============================================================================
# GRPO Training without REASONING for Agricultural VQA
# Dataset: 200k samples, 3 epochs
# Hardware: 4x A800 80GB GPUs
# Note: This version trains without chain-of-thought reasoning
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
export WANDB_NAME="grpo_no_reasoning_200k_3epochs"

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
export PYTHONPATH="${REPO_ROOT}/src/r1-v/src:${PYTHONPATH}"

# ============================================================================
# Training Configuration
# ============================================================================
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="/path/to/training_data/format1/train"
OUTPUT_DIR="/path/to/outputs/grpo_no_reasoning_200k_3epochs"
DS_CONFIG="${REPO_ROOT}/src/r1-v/configs/ds_zero3.json"

echo "Starting GRPO (No REASONING) training..."
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"

deepspeed --num_gpus=4 "${REPO_ROOT}/src/r1-v/src/open_r1/grpo_no_reasoning.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_PATH}" \
  --dataset_train_split train \
  --learning_rate 8e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.15 \
  --per_device_train_batch_size 10 \
  --gradient_accumulation_steps 4 \
  --num_generations 3 \
  --temperature 0.7 \
  --num_train_epochs 3 \
  --max_steps 3750 \
  --output_dir "${OUTPUT_DIR}" \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --bf16 true \
  --logging_first_step true \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 300 \
  --gradient_checkpointing true \
  --deepspeed "${DS_CONFIG}" \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --max_pixels 147456 \
  --max_grad_norm 0.3 \
  --report_to wandb \
  --seed 42

echo "Training completed!"
