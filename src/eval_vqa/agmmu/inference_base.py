#!/usr/bin/env python3
"""
AGMMU Inference Script - For generalization experiments
Supports MCQ inference, adapted for Qwen-2.5-VL model
"""
import os
import json
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


def load_model(model_path, device="cuda"):
    """Load model and processor"""
    print(f"[INFO] Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    # Fix padding side warning - decoder-only models require left padding
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    print(f"[INFO] Model loaded successfully on {device}")
    return model, processor


def batch_inference(model, processor, batch_data, device, max_new_tokens=256):
    """Batch inference"""
    batch_images = []
    batch_texts = []
    valid_indices = []

    for idx, item in enumerate(batch_data):
        try:
            image_path = item['image_path']
            question = item['question']

            if not os.path.exists(image_path):
                print(f"  Warning: Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")

            # Build conversation messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            batch_images.append(image)
            batch_texts.append(text)
            valid_indices.append(idx)

        except Exception as e:
            print(f"  Error processing item {idx}: {str(e)}")
            continue

    if len(batch_images) == 0:
        return []

    try:
        # Process inputs
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

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
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        responses = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return [(valid_indices[i], responses[i].strip()) for i in range(len(responses))]

    except Exception as e:
        print(f"  Batch inference failed: {str(e)}")
        return []


def run_inference_agmmu(model_path, data_path, image_dir, output_path, batch_size=4, max_samples=None):
    """Run AGMMU inference"""
    # Load data
    print(f"\n[INFO] Loading dataset from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"[INFO] Total samples: {len(data)}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(model_path, device)

    # Prepare batch data
    predictions = []

    for i in tqdm(range(0, len(data), batch_size), desc="Inference"):
        batch = data[i:i+batch_size]

        # Prepare batch
        batch_items = []
        for item in batch:
            # Process image paths (Windows path conversion)
            image_paths = [img.replace('\\', '/') for img in item['images']]
            full_path = os.path.join(image_dir, image_paths[0])

            batch_items.append({
                'id': item['id'],
                'image_path': full_path,
                'question': item['question'],
                'qtype': item['qtype']
            })

        # Batch inference
        batch_results = batch_inference(model, processor, batch_items, device)

        # Collect results
        for idx, response in batch_results:
            item = batch_items[idx]
            predictions.append({
                'id': item['id'],
                'prediction': response,
                'qtype': item['qtype']
            })

    # Save results
    print(f"\n[INFO] Saving predictions to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Completed! Total predictions: {len(predictions)}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="AGMMU Inference for Generalization Experiment")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to AGMMU validation JSON")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for predictions")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")

    args = parser.parse_args()

    print("="*80)
    print("AGMMU Generalization Inference")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    run_inference_agmmu(
        model_path=args.model_path,
        data_path=args.data_path,
        image_dir=args.image_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    """
    Usage:
    python run_inference_agmmu.py \
        --model_path ../batch_test_RECOMMEND_1_bs192/sft_inference_model \
        --data_path agmmu_dataset/validation_set.json \
        --image_dir agmmu_dataset \
        --output results/sft_model_agmmu_predictions.json \
        --batch_size 4
    """
    main()
