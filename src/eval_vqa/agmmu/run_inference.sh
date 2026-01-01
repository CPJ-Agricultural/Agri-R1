#!/bin/bash
################################################################################
# Main script for generalization experiments
# Run inference and evaluation on AGMMU and MIRAGE datasets with two models
################################################################################

set -e  # Exit immediately on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration paths
BASE_DIR="/root/autodl-tmp/grpo_format1_200k_3epochs"
AGRI_LLM_DIR="${BASE_DIR}/AGRI-LLM"
MODEL_DIR="${BASE_DIR}/batch_test_RECOMMEND_1_bs192"

# Model paths
SFT_MODEL="${MODEL_DIR}/sft_inference_model"
CKPT1800_MODEL="${MODEL_DIR}/checkpoint-1800-inference"

# Dataset paths
AGMMU_DATA="${AGRI_LLM_DIR}/agmmu_dataset/validation_set.json"
AGMMU_IMAGES="${AGRI_LLM_DIR}/agmmu_dataset"
MIRAGE_DATA="${AGRI_LLM_DIR}/mirage_dataset/MMST_Standard_1000.json"
MIRAGE_IMAGES="${AGRI_LLM_DIR}/mirage_dataset/MMST_Standard_1000_images"

# Results directory
RESULTS_DIR="${AGRI_LLM_DIR}/generalization_results"
mkdir -p "${RESULTS_DIR}"

# Log file
LOG_FILE="${RESULTS_DIR}/experiment_log_$(date +%Y%m%d_%H%M%S).txt"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_section() {
    echo -e "\n${GREEN}========================================${NC}" | tee -a "${LOG_FILE}"
    echo -e "${GREEN}$1${NC}" | tee -a "${LOG_FILE}"
    echo -e "${GREEN}========================================${NC}\n" | tee -a "${LOG_FILE}"
}

# Check models and datasets
check_prerequisites() {
    log_section "Checking Prerequisites"

    if [ ! -d "${SFT_MODEL}" ]; then
        log_error "SFT model does not exist: ${SFT_MODEL}"
        exit 1
    fi
    log_info "✓ SFT model exists"

    if [ ! -d "${CKPT1800_MODEL}" ]; then
        log_error "Checkpoint-1800 model does not exist: ${CKPT1800_MODEL}"
        exit 1
    fi
    log_info "✓ Checkpoint-1800 model exists"

    if [ ! -f "${AGMMU_DATA}" ]; then
        log_error "AGMMU dataset does not exist: ${AGMMU_DATA}"
        exit 1
    fi
    log_info "✓ AGMMU dataset exists"

    if [ ! -f "${MIRAGE_DATA}" ]; then
        log_error "MIRAGE dataset does not exist: ${MIRAGE_DATA}"
        exit 1
    fi
    log_info "✓ MIRAGE dataset exists"

    log_info "All prerequisites check passed!"
}

# Run AGMMU inference
run_agmmu_inference() {
    local model_path=$1
    local model_name=$2
    local output_file="${RESULTS_DIR}/${model_name}_agmmu_predictions.json"

    log_section "Running AGMMU Inference - ${model_name}"

    cd "${AGRI_LLM_DIR}"

    python run_inference_agmmu.py \
        --model_path "${model_path}" \
        --data_path "${AGMMU_DATA}" \
        --image_dir "${AGMMU_IMAGES}" \
        --output "${output_file}" \
        --batch_size 16 \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_info "✓ AGMMU inference completed: ${output_file}"
    else
        log_error "✗ AGMMU inference failed"
        return 1
    fi
}

# Run MIRAGE inference
run_mirage_inference() {
    local model_path=$1
    local model_name=$2
    local output_file="${RESULTS_DIR}/${model_name}_mirage_predictions.json"

    log_section "Running MIRAGE Inference - ${model_name}"

    cd "${AGRI_LLM_DIR}"

    python run_inference_mirage.py \
        --model_path "${model_path}" \
        --data_path "${MIRAGE_DATA}" \
        --image_dir "${MIRAGE_IMAGES}" \
        --output "${output_file}" \
        --batch_size 12 \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_info "✓ MIRAGE inference completed: ${output_file}"
    else
        log_error "✗ MIRAGE inference failed"
        return 1
    fi
}

# Evaluate AGMMU results
evaluate_agmmu() {
    local model_name=$1
    local pred_file="${RESULTS_DIR}/${model_name}_agmmu_predictions.json"
    local eval_file="${RESULTS_DIR}/${model_name}_agmmu_evaluation.json"

    log_section "Evaluating AGMMU Results - ${model_name}"

    cd "${AGRI_LLM_DIR}"

    python evaluate_agmmu.py \
        --data_path "${AGMMU_DATA}" \
        --image_dir "${AGMMU_IMAGES}" \
        --predictions "${pred_file}" \
        --output "${eval_file}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_info "✓ AGMMU evaluation completed: ${eval_file}"
    else
        log_error "✗ AGMMU evaluation failed"
        return 1
    fi
}

# Evaluate MIRAGE results
evaluate_mirage() {
    local model_name=$1
    local pred_file="${RESULTS_DIR}/${model_name}_mirage_predictions.json"
    local eval_file="${RESULTS_DIR}/${model_name}_mirage_evaluation.json"

    log_section "Evaluating MIRAGE Results - ${model_name}"

    cd "${AGRI_LLM_DIR}"

    python evaluate_with_llm_judge.py \
        --data_path "${MIRAGE_DATA}" \
        --predictions "${pred_file}" \
        --output "${eval_file}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_info "✓ MIRAGE evaluation completed: ${eval_file}"
    else
        log_error "✗ MIRAGE evaluation failed"
        return 1
    fi
}

# Generate comparison report
generate_comparison_report() {
    log_section "Generating Generalization Capability Comparison Report"

    cd "${AGRI_LLM_DIR}"

    python generate_comparison_report.py \
        --results_dir "${RESULTS_DIR}" \
        --output "${RESULTS_DIR}/generalization_comparison_report.md" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_info "✓ Comparison report generated successfully"
    else
        log_warning "Comparison report generation failed (may need to create script manually)"
    fi
}

# Main function
main() {
    log_section "Generalization Experiment Started"
    log_info "Start time: $(date)"
    log_info "Log file: ${LOG_FILE}"

    # Check prerequisites
    check_prerequisites

    # Experiment 1: SFT Model + AGMMU
    run_agmmu_inference "${SFT_MODEL}" "sft_model"
    evaluate_agmmu "sft_model"

    # Experiment 2: SFT Model + MIRAGE
    run_mirage_inference "${SFT_MODEL}" "sft_model"
    evaluate_mirage "sft_model"

    # Experiment 3: Checkpoint-1800 + AGMMU
    run_agmmu_inference "${CKPT1800_MODEL}" "checkpoint1800"
    evaluate_agmmu "checkpoint1800"

    # Experiment 4: Checkpoint-1800 + MIRAGE
    run_mirage_inference "${CKPT1800_MODEL}" "checkpoint1800"
    evaluate_mirage "checkpoint1800"

    # Generate comparison report
    generate_comparison_report

    log_section "Generalization Experiment Completed"
    log_info "End time: $(date)"
    log_info "All results saved in: ${RESULTS_DIR}"

    # Display results summary
    echo ""
    echo "=========================================="
    echo "Experiment Results Summary"
    echo "=========================================="
    echo "SFT Model:"
    echo "  - AGMMU predictions: ${RESULTS_DIR}/sft_model_agmmu_predictions.json"
    echo "  - AGMMU evaluation: ${RESULTS_DIR}/sft_model_agmmu_evaluation.json"
    echo "  - MIRAGE predictions: ${RESULTS_DIR}/sft_model_mirage_predictions.json"
    echo "  - MIRAGE evaluation: ${RESULTS_DIR}/sft_model_mirage_evaluation.json"
    echo ""
    echo "Checkpoint-1800 Model:"
    echo "  - AGMMU predictions: ${RESULTS_DIR}/checkpoint1800_agmmu_predictions.json"
    echo "  - AGMMU evaluation: ${RESULTS_DIR}/checkpoint1800_agmmu_evaluation.json"
    echo "  - MIRAGE predictions: ${RESULTS_DIR}/checkpoint1800_mirage_predictions.json"
    echo "  - MIRAGE evaluation: ${RESULTS_DIR}/checkpoint1800_mirage_evaluation.json"
    echo ""
    echo "Comparison report: ${RESULTS_DIR}/generalization_comparison_report.md"
    echo "=========================================="
}

# Run main function
main "$@"
