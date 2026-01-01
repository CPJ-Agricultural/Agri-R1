#!/usr/bin/env python3
# coding: utf-8
"""
Zero-shot Inference Script - Only output answer tags (using optimized prompts)
Optimization objectives:
1.  only output <answer>...</answer> format
2. Use optimized prompts from five-shot script
3. Zero-shot inference without examples
4. Optimize batch size to fully utilize VRAM
"""
import os
import json
import torch
import argparse
import re
from PIL import Image
from collections import OrderedDict
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

# ========== Optimized System Prompt (from fiveshot script) ==========
SYSTEM_PROMPT = """You are an agricultural visual question answering assistant. Based on the provided crop image, you provide professional and precise answers to the user's questions.

## Skills
1. Identify crop type from image accurately
2. Identify disease/pest type from image with scientific precision
3. Understand user's question thoroughly
4. Analyze image comprehensively based on the question
5. Combine analysis to provide professional answers

## Rules
1. Base answers solely on image and question
2. Prioritize scientific accuracy in all responses
3. Never return empty answers
4. Answer MUST include BOTH plant type and disease type
5. Focus on PEST/DISEASE identification as primary task
6. Answers should be scientifically accurate and detailed
7. Use standardized disease/pest names
8. Describe key visual symptoms when identifying diseases

## Output Format
Return your answer in this format:
<answer>Your detailed answer including both crop type and disease/pest identification with key symptoms</answer>

## Important Notes
- Always identify the plant species first
- Then identify the disease or pest issue
- Include observable symptoms in your identification
- Be specific about disease names (e.g., "Late Blight" not just "blight")
- Maintain professional and scientific terminology"""


class InferenceDataset(Dataset):
    """Inference dataset"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def validate_and_fix_format(text):
    """
    Validate and fix output format
    Ensure output conforms to <answer>...</answer> format
    """
    original_text = text.strip()

    # Check if already in correct format
    answer_pattern = r'<answer>(.*?)</answer>'
    has_answer = re.search(answer_pattern, original_text, re.DOTALL)

    # If format is completely correct, return directly
    if has_answer:
        return original_text

    # Attempt to fix format
    fixed_text = original_text

    # Remove common error prefixes
    prefixes_to_remove = [
        'ANSWER:', 'Answer:', 'NEW ANSWER:', '<think>', '</think>'
    ]
    for prefix in prefixes_to_remove:
        if prefix in fixed_text:
            fixed_text = fixed_text.replace(prefix, '').strip()

    # Attempt to extract answer part
    answer_content = fixed_text

    # If text is too long, truncate to first 500 characters
    if len(answer_content) > 500:
        answer_content = answer_content[:500]

    # Build correct format
    formatted_output = f"<answer>{answer_content}</answer>"

    return formatted_output


def collate_fn(batch, processor, device):
    """
    Batch collate function
    Process a batch of samples into model inputs
    """
    images = []
    texts = []
    metadata = []

    for entry in batch:
        image_path = entry.get("image", "")
        question = entry.get("question", "")

        # Check image path
        if not os.path.exists(image_path):
            # Skip if image does not exist
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Build messages (zero-shot)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            # Apply chat template
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            images.append(image)
            texts.append(text)
            metadata.append(entry)

        except Exception as e:
            print(f"Warning: Failed to process sample {image_path}: {e}")
            continue

    if len(images) == 0:
        return None, None, None

    # Process inputs
    try:
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        return inputs, metadata, len(images)
    except Exception as e:
        print(f"Warning: Batch processing failed: {e}")
        return None, None, None


def generate_batch(model, processor, inputs, device, batch_size, max_new_tokens=128):
    """
    Batch generation
    Optimized parameters to ensure output completeness and format correctness
    """
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                do_sample=False,  # Use greedy decoding, faster
                num_beams=1,  # Don't use beam search
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Only decode newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        responses = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Validate and fix format
        fixed_responses = []
        for i, response in enumerate(responses):
            response = response.strip()
            fixed_response = validate_and_fix_format(response)
            fixed_responses.append(fixed_response)

        return fixed_responses

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return format-correct error message
        error_msg = f"<answer>Generation error occurred.</answer>"
        return [error_msg] * batch_size


def validate_output_quality(results):
    """
    Validate output quality
    Statistics on format correctness and completeness
    """
    stats = {
        'total': len(results),
        'with_answer_tag': 0,
        'complete_format': 0,
    }

    for result in results:
        solution = result.get('solution', '')

        has_answer = '<answer>' in solution and '</answer>' in solution

        if has_answer:
            stats['with_answer_tag'] += 1
            stats['complete_format'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--input_json', type=str, required=True, help='Input dataset JSON')
    parser.add_argument('--output_json', type=str, required=True, help='Output results JSON')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process (0=all)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch processing size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker threads')
    parser.add_argument('--max_new_tokens', type=int, default=120, help='Maximum generation tokens')
    args = parser.parse_args()

    print("="*80)
    print("Zero-shot Inference Script - Only output answer tags (optimized prompt version)")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Input data: {args.input_json}")
    print(f"Output path: {args.output_json}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max generation length: {args.max_new_tokens}")
    print(f"Worker threads: {args.num_workers}")
    print(f"Prompt strategy: Zero-shot + Optimized System Prompt (from fiveshot script)")
    print("="*80)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    Total VRAM: {total_mem:.2f} GB")

    # Load model
    print(f"\nLoading model...")
    start_load_time = time.time()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    # Set left padding - correct setting for decoder-only architecture
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    load_time = time.time() - start_load_time
    print(f"✓ Model loaded successfully! Time: {load_time:.2f}s")

    # Check VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\nVRAM usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    # Read data
    print(f"\nReading dataset...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limit sample count
    if args.num_samples > 0:
        data = data[:args.num_samples]
    total = len(data)
    print(f"✓ Will process {total} data samples")

    # Create dataset and DataLoader
    dataset = InferenceDataset(data)

    # Start inference
    results = []
    start_time = time.time()
    successful_count = 0
    failed_count = 0
    processed_count = 0

    print(f"\nStarting batch inference...")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Expected batches: {(total + args.batch_size - 1) // args.batch_size}\n")

    # Manual batch processing (avoid DataLoader to prevent multiprocessing issues)
    batch_count = 0
    for batch_start in tqdm(range(0, total, args.batch_size), desc="Inference progress"):
        batch_end = min(batch_start + args.batch_size, total)
        batch = dataset.data[batch_start:batch_end]

        # Collate batch
        inputs, metadata, valid_count = collate_fn(batch, processor, device)

        if inputs is None or valid_count == 0:
            # Handle failed samples
            for entry in batch:
                new_entry = OrderedDict()
                new_entry["question_id"] = entry.get("question_id", f"test_conv_{processed_count+1:04d}")
                new_entry["problem"] = entry.get("question", "")
                new_entry["image"] = entry.get("image", "")
                new_entry["answer"] = entry.get("answer", "")
                new_entry["solution"] = "<answer>Unable to process image.</answer>"
                results.append(new_entry)
                failed_count += 1
                processed_count += 1
            continue

        # Generate responses
        solutions = generate_batch(model, processor, inputs, device, valid_count, args.max_new_tokens)

        # Build results
        for idx, (meta, solution) in enumerate(zip(metadata, solutions)):
            processed_count += 1

            # Check if successful (format correct)
            if '<answer>' in solution:
                successful_count += 1
            else:
                failed_count += 1

            # Build result entry
            new_entry = OrderedDict()
            new_entry["question_id"] = meta.get("question_id", f"test_conv_{processed_count:04d}")
            new_entry["problem"] = meta.get("question", "")
            new_entry["image"] = meta.get("image", "")
            new_entry["answer"] = meta.get("answer", "")
            new_entry["solution"] = solution
            results.append(new_entry)

        batch_count += 1

        # Print progress every 10 batches
        if batch_count % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / batch_count
            remaining_batches = ((total + args.batch_size - 1) // args.batch_size) - batch_count
            eta_seconds = avg_time_per_batch * remaining_batches

            # Check GPU utilization
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                util_pct = (allocated / total_mem) * 100

                print(f"\nProgress: [{processed_count}/{total}] (Batch {batch_count})")
                print(f"  Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"  Average speed: {avg_time_per_batch:.2f} s/batch ({avg_time_per_batch/args.batch_size:.2f} s/sample)")
                print(f"  Estimated remaining: {eta_seconds/60:.1f} minutes")
                print(f"  Format correct: {successful_count}, Format incorrect: {failed_count}")
                print(f"  GPU VRAM usage: {allocated:.2f}/{total_mem:.2f} GB ({util_pct:.1f}%)\n")

    # Final statistics
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"Average speed: {total_time/total:.2f} s/sample")
    print(f"Throughput: {total/(total_time/60):.1f} samples/minute")
    print(f"Format correct: {successful_count}/{total} ({successful_count/total*100:.1f}%)")
    print(f"Format incorrect: {failed_count}/{total}")

    # Validate output quality
    print(f"\n{'='*80}")
    print("Output Quality Validation")
    print(f"{'='*80}")
    quality_stats = validate_output_quality(results)
    print(f"Total samples: {quality_stats['total']}")
    print(f"Contains <answer> tag: {quality_stats['with_answer_tag']} ({quality_stats['with_answer_tag']/quality_stats['total']*100:.1f}%)")
    print(f"Complete format: {quality_stats['complete_format']} ({quality_stats['complete_format']/quality_stats['total']*100:.1f}%)")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nGPU VRAM:")
        print(f"  Current usage: {allocated:.2f} GB ({allocated/total_mem*100:.1f}%)")
        print(f"  Peak usage: {peak_mem:.2f} GB ({peak_mem/total_mem*100:.1f}%)")

    print(f"{'='*80}")

    # Save results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {args.output_json}")

    # Print warning if format correctness is below 80%
    format_correctness = successful_count / total * 100
    if format_correctness < 80:
        print(f"\n⚠️ Warning: Format correctness ({format_correctness:.1f}%) is below 80%, recommend checking model output")
    else:
        print(f"\n✓ Format correctness achieved {format_correctness:.1f}%")


if __name__ == "__main__":
    main()
