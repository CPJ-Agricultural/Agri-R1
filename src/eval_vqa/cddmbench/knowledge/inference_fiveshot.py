#!/usr/bin/env python3
# coding: utf-8
"""
Five-shot Inference Script for Knowledge QA - Using 5 examples to enhance reasoning
Features:
1. Five carefully designed knowledge QA examples
2. Optimized system prompt for agricultural disease knowledge
3. Robust CUDA error recovery mechanism
4. JSON format output
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

# ========== Optimized System Prompt ==========
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
Return your answer in JSON format:
{"answer": "Your comprehensive answer with specific details"}

## Important Notes
- Answers should be thorough and informative
- Include both preventive and curative measures when discussing control
- Provide scientific names alongside common names
- Specify exact concentrations, temperatures, and other measurements
- Organize information logically (e.g., by category, priority, or sequence)"""

# ========== Five-shot Examples ==========
EXAMPLES = [
    {
        "question": "How do you control this disease?",
        "answer": '{"answer": "For Tomato Late Blight control: 1) Cultural practices - Remove and destroy infected plants immediately, ensure 2-3 year crop rotation with non-solanaceous crops, maintain proper plant spacing (45-60cm) for air circulation, avoid overhead irrigation. 2) Chemical control - Apply fungicides preventively before disease onset: Mancozeb (2-3 g/L) or Chlorothalonil (2 g/L) every 7-10 days; For active infections use systemic fungicides: Metalaxyl+Mancozeb (2.5 g/L) or Dimethomorph (1 ml/L) every 5-7 days. 3) Resistant varieties - Plant resistant cultivars like Iron Lady or Mountain Magic. 4) Environmental management - Reduce leaf wetness duration, avoid working with plants when wet, maintain optimal greenhouse conditions (20-25°C, <85% RH)."}'
    },
    {
        "question": "What are the symptoms of this disease?",
        "answer": '{"answer": "Tomato Leaf Mold (Fulvia fulva) symptoms: 1) Early stage - Pale green to yellowish spots on upper leaf surface (2-5mm diameter), fuzzy olive-green to brown mold growth on lower leaf surface. 2) Progressive symptoms - Spots coalesce into larger irregular patches, leaves turn yellow and curl upward at margins, older leaves affected first with disease progressing upward. 3) Advanced stage - Extensive defoliation from bottom to top, fruit infection with dark leathery lesions near stem end, reduced photosynthesis leading to stunted growth and yield loss. 4) Environmental indicators - Symptoms intensify under high humidity (>85% RH) and moderate temperatures (20-25°C), spore masses visible as velvety coating on leaf undersides."}'
    },
    {
        "question": "What pathogen causes this disease?",
        "answer": '{"answer": "This disease is caused by Alternaria solani (Tomato Early Blight). Pathogen characteristics: 1) Classification - Fungus, Ascomycota phylum, order Pleosporales, family Pleosporaceae. 2) Morphology - Produces multicellular dark brown conidia (100-300 μm long) with transverse and longitudinal septa forming characteristic beaked club shape, arranged in chains. 3) Lifecycle - Overwinters as mycelium in infected plant debris and on seed surfaces; produces conidia under favorable conditions (24-29°C, >90% RH); spores disseminated by wind, rain splash, and mechanical contact; germinates within 35-45 minutes under optimal moisture. 4) Host range - Primarily affects Solanaceous crops (tomato, potato, eggplant); has multiple races with varying virulence. 5) Survival - Can survive 1-2 years in soil on crop residue; also seedborne, making seed treatment important."}'
    },
    {
        "question": "Under what environmental conditions does this disease develop?",
        "answer": '{"answer": "Wheat Leaf Rust (Puccinia triticina) environmental requirements: 1) Temperature - Optimal: 15-22°C for urediniospore germination and infection; Minimum: 2°C; Maximum: 35°C (spores killed above 40°C). Infection can occur within 6-8 hours at optimal temperature. 2) Moisture - Requires free moisture or high relative humidity (>95% RH) for 4-6 hours for successful infection; Dew formation critical for spore germination; Light intermittent rain promotes spore dispersal. 3) Light - Disease severity increases under moderate light intensity; Dense canopy creates favorable microclimate. 4) Host factors - Susceptible growth stages: tillering to heading most vulnerable; Nitrogen-rich plants more susceptible; Water-stressed plants show increased susceptibility. 5) Geographic/seasonal patterns - Prevalent in regions with cool, moist springs (10-25°C); Continuous wheat production areas with overlapping crop stages; Disease spreads rapidly under favorable conditions (epidemic can develop within 2-3 weeks)."}'
    },
    {
        "question": "What are the best management practices for this disease?",
        "answer": '{"answer": "Integrated Disease Management for Rice Blast (Magnaporthe oryzae): 1) Resistant varieties - Plant resistant cultivars appropriate for local race populations; rotate varieties every 2-3 years to avoid pathogen adaptation. 2) Cultural practices - Use certified disease-free seeds; Apply balanced fertilization (avoid excessive nitrogen >120 kg/ha); Maintain optimal water management with intermittent irrigation; Ensure proper plant spacing (20×15 cm); Remove and destroy crop residues after harvest; Practice 1-2 year crop rotation. 3) Chemical control - Seed treatment: Tricyclazole (0.75 g/kg seed) or Carbendazim (2 g/kg); Foliar sprays: Apply at critical stages (tillering, panicle initiation, heading): Tricyclazole (0.6 g/L), Azoxystrobin (1 ml/L), or Isoprothiolane (1.5 ml/L) at 10-12 day intervals. 4) Biological control - Apply Pseudomonas fluorescens (10 g/L) as seedling dip and foliar spray. 5) Monitoring - Scout fields weekly during susceptible stages; Use disease forecasting models based on weather data; Apply fungicides preventively when conditions favor disease development (high humidity, moderate temperature 25-28°C, prolonged leaf wetness)."}'
    }
]


def check_and_resize_image(image_path, max_size=896):
    """Check and resize image to avoid CUDA issues"""
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


def process_single_sample(entry, processor, model, device, max_new_tokens):
    """Process single sample (Five-shot mode)"""
    image_path = entry.get("image", "")
    question = entry.get("question", "")

    if not os.path.exists(image_path):
        return "Failed: Image not found"

    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load and check image
        image = check_and_resize_image(image_path, max_size=896)
        if image is None:
            return "Failed: Image loading error"

        # Build messages (Five-shot: system + 5 examples + current question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add 5 examples
        for i, example in enumerate(EXAMPLES):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": example['question']}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example['answer']
            })

        # Add current question
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        })

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

        # Cleanup
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--input_json', type=str, required=True, help='Input dataset JSON')
    parser.add_argument('--output_json', type=str, required=True, help='Output results JSON')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process (0=all)')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max generation tokens (knowledge answers are longer)')
    parser.add_argument('--max_image_size', type=int, default=896, help='Max image size')
    args = parser.parse_args()

    print("="*80)
    print("Knowledge Five-shot Inference Script (Agricultural Disease Knowledge QA)")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Input data: {args.input_json}")
    print(f"Output path: {args.output_json}")
    print(f"Max generation tokens: {args.max_new_tokens}")
    print(f"Max image size: {args.max_image_size}")
    print(f"Prompt: Five-shot with optimized Knowledge System Prompt")
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
        print(f"\nGPU memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    # Read data
    print(f"\nReading dataset...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limit samples
    if args.num_samples > 0:
        data = data[:args.num_samples]
    total = len(data)
    print(f"✓ Will process {total} samples")

    # Start inference (single-sample mode for knowledge QA due to longer answers)
    results = []
    start_time = time.time()
    successful_count = 0
    failed_count = 0

    print(f"\nStarting inference (single-sample mode)...")
    print(f"Strategy: Five-shot with knowledge examples")
    print(f"Max New Tokens: {args.max_new_tokens}\n")

    for idx, entry in enumerate(tqdm(data, desc="Inference Progress")):
        solution = process_single_sample(entry, processor, model, device, args.max_new_tokens)

        new_entry = OrderedDict()
        new_entry["question_id"] = entry.get("question_id", f"test_knowledge_fiveshot_{idx+1:04d}")
        new_entry["problem"] = entry.get("question", "")
        new_entry["image"] = entry.get("image", "")
        new_entry["answer"] = entry.get("answer", "")
        new_entry["solution"] = solution
        results.append(new_entry)

        if "Failed" not in solution and "failed" not in solution:
            successful_count += 1
        else:
            failed_count += 1

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (idx + 1)
            remaining_time = avg_time * (total - idx - 1)

            print(f"\nProgress: [{idx+1}/{total}]")
            print(f"  Elapsed: {elapsed_time/60:.1f} min")
            print(f"  Remaining: {remaining_time/60:.1f} min")
            print(f"  Success: {successful_count}, Failed: {failed_count}")
            print(f"  Success rate: {successful_count/(idx+1)*100:.1f}%\n")

    # Final statistics
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"Average speed: {total_time/total:.2f} seconds/sample")
    print(f"Throughput: {total/(total_time/60):.1f} samples/minute")
    print(f"Success: {successful_count}/{total} ({successful_count/total*100:.1f}%)")
    print(f"Failed: {failed_count}/{total} ({failed_count/total*100:.1f}%)")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nGPU memory:")
        print(f"  Current: {allocated:.2f} GB ({allocated/total_mem*100:.1f}%)")
        print(f"  Peak: {peak_mem:.2f} GB ({peak_mem/total_mem*100:.1f}%)")

    print(f"{'='*80}")

    # Save results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
