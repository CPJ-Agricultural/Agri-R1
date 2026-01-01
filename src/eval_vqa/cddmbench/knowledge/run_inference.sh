#!/bin/bash
# Knowledge Inference Script - For agricultural disease knowledge QA tasks (Zero-shot version)

# Configuration parameters
MODEL_PATH="/root/autodl-tmp/grpo_format1_200k_3epochs/batch_test_RECOMMEND_1_bs192/checkpoint-1800-inference"
DATA_PATH="/root/autodl-tmp/grpo_format1_200k_3epochs/datasets/disease_knowledge.json"
OUTPUT_DIR="/root/autodl-tmp/grpo_format1_200k_3epochs/checkpoint1800_zeroshot_inference_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Knowledge inference script (Zero-shot version)
echo "================================"
echo "Knowledge Inference (Agricultural Disease Knowledge QA - Zero-shot)"
echo "================================"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Features:"
echo "  1. Zero-shot mode (no examples provided, direct inference)"
echo "  2. Optimized System Prompt (emphasizes professional knowledge answers)"
echo "  3. Single answer format, no caption"
echo "  4. Batch size set to 4 (knowledge answers are longer)"
echo "  5. Max tokens set to 512 (ensures complete knowledge answers)"
echo "  6. Robust CUDA error recovery mechanism"
echo "  7. Automatic single-sample retry for failed samples"
echo "  8. Direct inference generation, no API calls required"
echo "================================"

cd /root/autodl-tmp/grpo_format1_200k_3epochs

# Use Knowledge Zero-shot parameters
python3 run_inference_knowledge_zeroshot.py \
    --model_path "$MODEL_PATH" \
    --input_json "$DATA_PATH" \
    --output_json "$OUTPUT_DIR/inference_results.json" \
    --batch_size 4 \
    --max_new_tokens 512 \
    --max_image_size 896 \
    2>&1 | tee "$OUTPUT_DIR/inference.log"

echo ""
echo "================================"
echo "Inference completed!"
echo "================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Run evaluation (statistics on success rate)
echo "================================"
echo "Starting evaluation statistics..."
echo "================================"

python3 << EOF
import json
import sys

# Read inference results
with open('$OUTPUT_DIR/inference_results.json', 'r') as f:
    results = json.load(f)

total_samples = len(results)
valid_samples = 0
empty_solutions = 0
failed_solutions = 0

for item in results:
    solution = item.get('solution', '')

    if not solution:
        empty_solutions += 1
        continue

    if 'CUDA error' in solution or 'Failed' in solution or 'failed' in solution:
        failed_solutions += 1
        continue

    valid_samples += 1

print("\n" + "="*80)
print("Knowledge Inference Statistics")
print("="*80)
print(f"Total samples: {total_samples}")
print(f"Successful inference: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
print(f"Empty results: {empty_solutions} ({empty_solutions/total_samples*100:.1f}%)")
print(f"Failed samples: {failed_solutions} ({failed_solutions/total_samples*100:.1f}%)")
print("="*80)

# Show first 3 samples as examples
print("\nSample Examples (First 3 successful samples):")
print("="*80)
count = 0
for item in results:
    solution = item.get('solution', '')
    if solution and 'CUDA error' not in solution and 'Failed' not in solution and 'failed' not in solution:
        print(f"\nSample {count+1}:")
        print(f"Question: {item.get('problem', '')[:100]}...")
        print(f"Answer length: {len(solution)} characters")
        print(f"Answer preview: {solution[:200]}...")
        count += 1
        if count >= 3:
            break
print("="*80)

EOF

echo ""
echo "All done!"
