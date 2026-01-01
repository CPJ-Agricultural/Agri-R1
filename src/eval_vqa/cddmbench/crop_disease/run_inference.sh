#!/bin/bash
################################################################################
# 4-GPU parallel inference script - Only infer first 4 models
################################################################################
#
# Inference models:
#   - checkpoint-900 (GPU 0)
#   - checkpoint-1200-inference (GPU 1)
#   - checkpoint-1500-inference (GPU 2)
#   - checkpoint-1800-inference (GPU 3)
#
# Estimated time: 25-40 minutes
#
################################################################################

set -e

# ============================================================================
# Configuration section
# ============================================================================

# Project paths
PROJECT_ROOT="/root/autodl-tmp/grpo_format1_200k_3epochs"
SCRIPT_DIR="${PROJECT_ROOT}/inference_experiment_backup"
MODEL_BASE_DIR="${PROJECT_ROOT}/batch_test_RECOMMEND_1_bs192"
DATA_DIR="${PROJECT_ROOT}/datasets"

# Dataset
INPUT_JSON="${DATA_DIR}/01disease_diagnosisv1.json"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${PROJECT_ROOT}/batch_inference_4models_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}"

# Model list (only 4 models)
MODELS=(
    "checkpoint-900"
    "checkpoint-1200-inference"
    "checkpoint-1500-inference"
    "checkpoint-1800-inference"
)

# Inference parameters
NUM_SAMPLES=0  # 0 = all samples
BATCH_SIZE=32  # Batch size (optimized: 32)
MAX_NEW_TOKENS=320  # Maximum generation tokens (optimized: 320)
NUM_WORKERS=2  # DataLoader worker threads

# Environment configuration
export PYTHONWARNINGS="ignore::UserWarning"

# ============================================================================
# Display configuration information
# ============================================================================

echo "================================================================================"
echo "4-GPU parallel inference - Only first 4 models"
echo "================================================================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output directory: ${OUTPUT_BASE}"
echo "================================================================================"
echo "Inference models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo "================================================================================"
echo "Optimization parameters:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max New Tokens: ${MAX_NEW_TOKENS}"
echo "  Parallel strategy: 4 models run simultaneously (1 GPU each)"
echo "================================================================================"
echo ""

# Activate conda environment
echo ">> Activating conda environment..."
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate agri-sft

# Check GPU
echo ""
echo ">> GPU status check:"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Available GPU count: ${GPU_COUNT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# Check dataset
if [ ! -f "${INPUT_JSON}" ]; then
    echo "❌ Error: Dataset file does not exist: ${INPUT_JSON}"
    exit 1
fi

TOTAL_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${INPUT_JSON}'))))")
echo ">> Dataset information:"
echo "   Path: ${INPUT_JSON}"
echo "   Total samples: ${TOTAL_SAMPLES}"
if [ ${NUM_SAMPLES} -eq 0 ]; then
    NUM_SAMPLES=${TOTAL_SAMPLES}
    echo "   Will process: All ${NUM_SAMPLES} samples"
else
    echo "   Will process: First ${NUM_SAMPLES} samples"
fi
echo "   Batch size: ${BATCH_SIZE}"
echo "   Estimated batches/model: $((NUM_SAMPLES / BATCH_SIZE + 1))"
echo ""

# Check inference script
if [ ! -f "${SCRIPT_DIR}/run_inference_fast_batch_optimized.py" ]; then
    echo "❌ Error: Optimized batch inference script does not exist: ${SCRIPT_DIR}/run_inference_fast_batch_optimized.py"
    exit 1
fi

echo "================================================================================"
echo "Starting inference..."
echo "================================================================================"
echo ""

# ============================================================================
# Inference function
# ============================================================================

run_inference() {
    local MODEL_NAME=$1
    local GPU_ID=$2

    echo "[GPU ${GPU_ID}] Starting: ${MODEL_NAME}"

    # Path configuration
    MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    INFERENCE_OUTPUT="${OUTPUT_DIR}/inference_results.json"
    EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
    LOG_FILE="${OUTPUT_DIR}/inference.log"

    # Record start time
    START_TIME=$(date +%s)

    # Set GPU
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    cd "${SCRIPT_DIR}"

    # Execute optimized batch inference
    echo "[GPU ${GPU_ID}] ${MODEL_NAME}: Inferencing (Batch Size=${BATCH_SIZE}, Max Tokens=${MAX_NEW_TOKENS})..." | tee -a "${LOG_FILE}"

    if python3 run_inference_fast_batch_optimized.py \
        --model_path "${MODEL_PATH}" \
        --input_json "${INPUT_JSON}" \
        --output_json "${INFERENCE_OUTPUT}" \
        --num_samples ${NUM_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --num_workers ${NUM_WORKERS} \
        >> "${LOG_FILE}" 2>&1; then

        INFERENCE_END=$(date +%s)
        INFERENCE_TIME=$((INFERENCE_END - START_TIME))

        echo "[GPU ${GPU_ID}] ${MODEL_NAME}: Inference completed (${INFERENCE_TIME}s)" | tee -a "${LOG_FILE}"

        # Execute evaluation
        echo "[GPU ${GPU_ID}] ${MODEL_NAME}: Evaluating..." | tee -a "${LOG_FILE}"

        if python3 evaluate_with_fuzzy_matching.py \
            --results "${INFERENCE_OUTPUT}" \
            --threshold 0.6 \
            --output "${EVALUATION_OUTPUT}" \
            >> "${LOG_FILE}" 2>&1; then

            EVAL_END=$(date +%s)
            TOTAL_TIME=$((EVAL_END - START_TIME))

            # Extract results
            CROP_ACC=$(python3 -c "import json; d=json.load(open('${EVALUATION_OUTPUT}')); print(f\"{d['crop_accuracy']:.2%}\")")
            DISEASE_ACC=$(python3 -c "import json; d=json.load(open('${EVALUATION_OUTPUT}')); print(f\"{d['disease_accuracy']:.2%}\")")

            echo "[GPU ${GPU_ID}] ${MODEL_NAME}: ✓ Completed!" | tee -a "${LOG_FILE}"
            echo "[GPU ${GPU_ID}] ${MODEL_NAME}:   Crop accuracy: ${CROP_ACC}" | tee -a "${LOG_FILE}"
            echo "[GPU ${GPU_ID}] ${MODEL_NAME}:   Disease accuracy: ${DISEASE_ACC}" | tee -a "${LOG_FILE}"
            echo "[GPU ${GPU_ID}] ${MODEL_NAME}:   Total time: ${TOTAL_TIME}s ($((TOTAL_TIME/60))min)" | tee -a "${LOG_FILE}"

            return 0
        else
            echo "[GPU ${GPU_ID}] ${MODEL_NAME}: ❌ Evaluation failed" | tee -a "${LOG_FILE}"
            return 1
        fi
    else
        echo "[GPU ${GPU_ID}] ${MODEL_NAME}: ❌ Inference failed" | tee -a "${LOG_FILE}"
        return 1
    fi
}

# ============================================================================
# 4 models parallel inference
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4 models parallel inference (GPU 0-3, Batch Size=${BATCH_SIZE}, Max Tokens=${MAX_NEW_TOKENS})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

# Start 4 parallel tasks
PIDS=()
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    GPU_ID=$i

    echo "Starting: ${MODEL_NAME} on GPU ${GPU_ID} (Batch Size=${BATCH_SIZE}, Max Tokens=${MAX_NEW_TOKENS})"

    (
        run_inference "${MODEL_NAME}" "${GPU_ID}"
    ) &

    PIDS+=($!)

    # Brief delay to avoid simultaneous loading
    sleep 5
done

echo ""
echo "Waiting for 4 models to complete..."
echo "Process IDs: ${PIDS[@]}"
echo ""
echo "💡 Tip: You can run 'watch -n 1 nvidia-smi' in another terminal to monitor GPU utilization"
echo ""

# Wait for all processes to complete
SUCCESS_COUNT=0
FAILED_COUNT=0

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    MODEL_NAME="${MODELS[$i]}"

    if wait $PID; then
        echo "✓ ${MODEL_NAME} completed"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ ${MODEL_NAME} failed"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All completed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Duration: $((TOTAL_DURATION/60))min (${TOTAL_DURATION}s)"
echo "Success: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# Generate summary report
# ============================================================================

SUMMARY_FILE="${OUTPUT_BASE}/experiment_summary.txt"

cat > "${SUMMARY_FILE}" <<EOF
4 Models Parallel Inference Experiment Report (Optimized)
==================================================

Experiment Configuration
--------------------------------------------------
Start time: $(date '+%Y-%m-%d %H:%M:%S' -d @${START_TIME})
End time: $(date '+%Y-%m-%d %H:%M:%S' -d @${END_TIME})
Dataset: ${INPUT_JSON}
Samples: ${NUM_SAMPLES}
Batch Size: ${BATCH_SIZE}
Max New Tokens: ${MAX_NEW_TOKENS}
Parallel strategy: 4 models run simultaneously
Inference script: run_inference_fast_batch_optimized.py

Time Statistics
--------------------------------------------------
Total duration: $((TOTAL_DURATION/60))min (${TOTAL_DURATION}s)

Result Statistics
--------------------------------------------------
Total models: 4
Success: ${SUCCESS_COUNT}
Failed: ${FAILED_COUNT}

==================================================
Detailed Results for Each Model
==================================================

EOF

# Collect results for each model
for MODEL_NAME in "${MODELS[@]}"; do
    EVAL_FILE="${OUTPUT_BASE}/${MODEL_NAME}/evaluation_results.json"

    if [ -f "${EVAL_FILE}" ]; then
        CROP_ACC=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['crop_accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        DISEASE_ACC=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['disease_accuracy']:.2%}\")" 2>/dev/null || echo "N/A")

        echo "[${MODEL_NAME}]" >> "${SUMMARY_FILE}"
        echo "  Status: Success" >> "${SUMMARY_FILE}"
        echo "  Crop identification accuracy: ${CROP_ACC}" >> "${SUMMARY_FILE}"
        echo "  Disease identification accuracy: ${DISEASE_ACC}" >> "${SUMMARY_FILE}"
        echo "  Output directory: ${OUTPUT_BASE}/${MODEL_NAME}" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    else
        echo "[${MODEL_NAME}]" >> "${SUMMARY_FILE}"
        echo "  Status: Failed" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
done

echo ""
echo "================================================================================"
echo "🎉 All tasks completed!"
echo "================================================================================"
echo "Output directory: ${OUTPUT_BASE}"
echo ""
echo "View summary report:"
echo "  cat ${SUMMARY_FILE}"
echo "================================================================================"
echo ""

# Display summary report
cat "${SUMMARY_FILE}"
