#!/usr/bin/env python3
# coding: utf-8
"""
Knowledge Inference Script - For agricultural disease knowledge QA tasks (Zero-shot + REASONING version)
Features:
1. Single answer format with chain-of-thought reasoning
2. For disease control, symptoms, pathogens and other knowledge QA
3. Uses Zero-shot mode (no examples provided)
4. Optimized System Prompt emphasizing professional knowledge answers with step-by-step reasoning
"""
import os
import json
import torch
import argparse
from PIL import Image
from collections import OrderedDict
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import time
import traceback
import gc

# ========== System Prompt for Knowledge Tasks (Aligned with zero-shot, with think tags) ==========
SYSTEM_PROMPT = """You are an agricultural disease knowledge expert. Based on the provided crop disease image, you provide comprehensive, scientifically accurate answers to questions about disease control, symptoms, pathogens, environmental conditions, and other agricultural knowledge.

## Skills
1. Deep understanding of plant pathology and disease management
2. Knowledge of disease symptoms and diagnostic features
3. Expertise in control measures and prevention strategies
4. Understanding of pathogen biology and disease cycles
5. Knowledge of environmental factors affecting disease development
6. Ability to provide detailed, educational responses

## Rules
1. Provide comprehensive, detailed answers based on scientific knowledge
2. Include specific details such as:
   - Control methods with exact dosages and application timing
   - Detailed symptom descriptions with visual characteristics
   - Pathogen identification with scientific names
   - Environmental conditions with specific parameters
   - Cultural practices and management strategies
3. Answer format should be professional and educational
4. Include practical, actionable information
5. Use standardized terminology and measurements
6. Structure answers clearly with numbered points when listing multiple items

## Output Format
You MUST return your answer in this format with step-by-step reasoning:
<think>Step 1: [Your reasoning for step 1]
Step 2: [Your reasoning for step 2]
Step 3: [Your reasoning for step 3]
Step 4: [Your reasoning for step 4]</think>
<answer>Your comprehensive, detailed answer with specific information about disease control/symptoms/pathogens/environmental conditions</answer>

## Important Notes
- Answers should be thorough and informative
- Include both preventive and curative measures when discussing control
- Provide scientific names alongside common names
- Specify exact concentrations, temperatures, and other measurements
- Organize information logically (e.g., by category, priority, or sequence)
- Provide step-by-step reasoning in <think> tags before your final answer"""

# ========== Zero-shot Mode (No Examples) ==========
# Note: This script uses zero-shot prompting - all examples have been removed
# The model should follow the format instructions in SYSTEM_PROMPT without examples

# ========== Utility Functions ==========

def check_and_resize_image(image_path, max_size=896):
    """Check and resize image dimensions"""
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        print(f"Image loading failed {image_path}: {e}")
        return None


def process_single_sample(entry, processor, model, device, max_new_tokens, max_image_size=896):
    """Process single sample (Zero-shot + Reasoning mode)"""
    image_path = entry.get("image", "")
    question = entry.get("question", "")

    if not os.path.exists(image_path):
        return "Failed: Image not found"

    try:
        # Clean cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load and check image
        image = check_and_resize_image(image_path, max_size=max_image_size)
        if image is None:
            return "Failed: Image loading error"

        # Build messages (Zero-shot + Reasoning - no examples)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Question: {question}"}
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        response = processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Clean up
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    except Exception as e:
        print(f"Single sample processing failed: {e}")
        # Force cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except:
                pass
        gc.collect()
        return f"Failed: {str(e)}"


def collate_fn_safe(batch, processor, device, max_size=896):
    """Safer batch collate function (Zero-shot + Reasoning version)"""
    images = []
    texts = []
    metadata = []

    for entry in batch:
        image_path = entry.get("image", "")
        question = entry.get("question", "")

        if not os.path.exists(image_path):
            continue

        try:
            # Load and check image
            image = check_and_resize_image(image_path, max_size)
            if image is None:
                continue

            # Build messages (Zero-shot + Reasoning - no examples)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"Question: {question}"}
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
        print(f"Warning: batch processing failed: {e}")
        return None, None, None


def generate_batch_safe(model, processor, inputs, device, batch_size, max_new_tokens):
    """Safer batch generation"""
    try:
        # Clean GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
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

        # Clean up
        del generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [r.strip() for r in responses]

    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "index" in error_msg.lower() or "assert" in error_msg.lower():
            print(f"\nCUDA error detected: {error_msg[:150]}...")
            # CUDA error recovery
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            gc.collect()
            return None  # Return None to indicate retry needed
        else:
            raise
    except Exception as e:
        print(f"Generation error: {e}")
        traceback.print_exc()
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--input_json', type=str, required=True, help='Input dataset JSON')
    parser.add_argument('--output_json', type=str, required=True, help='Output results JSON')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process (0=all)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (Zero-shot+Reasoning recommended 1-2)')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Maximum generation tokens (Reasoning needs longer)')
    parser.add_argument('--max_image_size', type=int, default=896, help='Maximum image size')
    args = parser.parse_args()

    print("="*80)
    print("Knowledge Zero-shot + Reasoning Optimized Inference Script")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Input data: {args.input_json}")
    print(f"Output path: {args.output_json}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max generation tokens: {args.max_new_tokens}")
    print(f"Max image size: {args.max_image_size}")
    print(f"Optimization focus: Agricultural disease control and management measures")
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
            print(f"    Total memory: {total_mem:.2f} GB")

    # Load model
    print(f"\nLoading model...")
    start_load_time = time.time()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    # Set padding_side='left'
    processor.tokenizer.padding_side = 'left'
    print(f"✓ Tokenizer padding_side set to: {processor.tokenizer.padding_side}")

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

    # Check GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\nGPU memory usage:")
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
    print(f"✓ Will process {total} samples")

    # Start inference
    results = []
    start_time = time.time()
    successful_count = 0
    failed_count = 0
    retry_count = 0
    processed_count = 0

    print(f"\nStarting batch inference...")
    print(f"Strategy: Optimized Zero-shot + Reasoning + Try batch first, retry single sample on failure")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Estimated batches: {(total + args.batch_size - 1) // args.batch_size}\n")

    # Batch processing
    batch_count = 0
    for batch_start in tqdm(range(0, total, args.batch_size), desc="Inference progress"):
        batch_end = min(batch_start + args.batch_size, total)
        batch = data[batch_start:batch_end]

        # Try batch processing
        inputs, metadata, valid_count = collate_fn_safe(batch, processor, device, args.max_image_size)

        # If collate fails, go directly to single sample mode
        if inputs is None or valid_count == 0:
            for entry in batch:
                solution = process_single_sample(entry, processor, model, device, args.max_new_tokens, args.max_image_size)
                retry_count += 1

                new_entry = OrderedDict()
                new_entry["question_id"] = entry.get("question_id", f"test_knowledge_conv_{processed_count+1:04d}")
                new_entry["problem"] = entry.get("question", "")
                new_entry["image"] = entry.get("image", "")
                new_entry["answer"] = entry.get("answer", "")
                new_entry["solution"] = solution
                results.append(new_entry)

                if "Failed" not in solution and "failed" not in solution:
                    successful_count += 1
                else:
                    failed_count += 1
                processed_count += 1
            continue

        # Try batch generation
        solutions = generate_batch_safe(model, processor, inputs, device, valid_count, args.max_new_tokens)

        # If batch generation fails, switch to single sample retry
        if solutions is None:
            print(f"  Batch {batch_count} failed, switching to single sample mode retry...")
            for entry in metadata:
                solution = process_single_sample(entry, processor, model, device, args.max_new_tokens, args.max_image_size)
                retry_count += 1

                new_entry = OrderedDict()
                new_entry["question_id"] = entry.get("question_id", f"test_knowledge_conv_{processed_count+1:04d}")
                new_entry["problem"] = entry.get("question", "")
                new_entry["image"] = entry.get("image", "")
                new_entry["answer"] = entry.get("answer", "")
                new_entry["solution"] = solution
                results.append(new_entry)

                if "Failed" not in solution and "failed" not in solution:
                    successful_count += 1
                else:
                    failed_count += 1
                processed_count += 1

            # Clean up inputs
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        # Batch successful, save results
        for idx, (meta, solution) in enumerate(zip(metadata, solutions)):
            processed_count += 1

            if "Failed" not in solution and "failed" not in solution:
                successful_count += 1
            else:
                failed_count += 1

            new_entry = OrderedDict()
            new_entry["question_id"] = meta.get("question_id", f"test_knowledge_conv_{processed_count:04d}")
            new_entry["problem"] = meta.get("question", "")
            new_entry["image"] = meta.get("image", "")
            new_entry["answer"] = meta.get("answer", "")
            new_entry["solution"] = solution
            results.append(new_entry)

        # Clean up
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        batch_count += 1

        # Print progress every 2 batches
        if batch_count % 2 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / batch_count
            remaining_batches = ((total + args.batch_size - 1) // args.batch_size) - batch_count
            eta_seconds = avg_time_per_batch * remaining_batches

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                util_pct = (allocated / total_mem) * 100

                print(f"\nProgress: [{processed_count}/{total}] (Batch {batch_count})")
                print(f"  Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"  Estimated remaining: {eta_seconds/60:.1f} minutes")
                print(f"  Success: {successful_count}, Failed: {failed_count}, Retry: {retry_count}")
                print(f"  Success rate: {successful_count/processed_count*100:.1f}%")
                print(f"  GPU memory: {allocated:.2f}/{total_mem:.2f} GB ({util_pct:.1f}%)\n")

    # Final statistics
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Inference complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"Average speed: {total_time/total:.2f} s/sample")
    print(f"Throughput: {total/(total_time/60):.1f} samples/minute")
    print(f"Success: {successful_count}/{total} ({successful_count/total*100:.1f}%)")
    print(f"Failed: {failed_count}/{total} ({failed_count/total*100:.1f}%)")
    print(f"Retry count: {retry_count}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nGPU memory:")
        print(f"  Current usage: {allocated:.2f} GB ({allocated/total_mem*100:.1f}%)")
        print(f"  Peak usage: {peak_mem:.2f} GB ({peak_mem/total_mem*100:.1f}%)")

    print(f"{'='*80}")

    # Save results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
