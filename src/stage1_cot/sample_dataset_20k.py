#!/usr/bin/env python3
# coding: utf-8
"""
Dataset Sampling Script - Extract 20k samples from large dataset
Features:
- Random sampling with seed for reproducibility
- Stratified sampling option to maintain class distribution
- Validates image paths before sampling
- Outputs JSON format compatible with training pipeline
"""

import json
import random
import os
import argparse
from collections import defaultdict
from tqdm import tqdm


def load_dataset(input_path):
    """
    Load dataset from JSON file

    Args:
        input_path: Path to input JSON file

    Returns:
        List of dataset entries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_path}")
    return data


def validate_images(data, verbose=True):
    """
    Validate that image paths exist

    Args:
        data: List of dataset entries
        verbose: Print validation progress

    Returns:
        Tuple of (valid_data, invalid_count)
    """
    valid_data = []
    invalid_count = 0

    iterator = tqdm(data, desc="Validating images") if verbose else data

    for entry in iterator:
        image_path = entry.get('image', '')

        if os.path.exists(image_path):
            valid_data.append(entry)
        else:
            invalid_count += 1
            if verbose and invalid_count <= 10:
                print(f"Warning: Image not found: {image_path}")

    if verbose:
        print(f"Valid entries: {len(valid_data)}")
        print(f"Invalid entries: {invalid_count}")

    return valid_data, invalid_count


def random_sample(data, sample_size, seed=42):
    """
    Perform random sampling

    Args:
        data: List of dataset entries
        sample_size: Number of samples to extract
        seed: Random seed for reproducibility

    Returns:
        List of sampled entries
    """
    random.seed(seed)

    if sample_size >= len(data):
        print(f"Warning: Requested sample size ({sample_size}) >= dataset size ({len(data)})")
        return data

    sampled = random.sample(data, sample_size)
    print(f"Randomly sampled {len(sampled)} entries")

    return sampled


def stratified_sample(data, sample_size, class_key='answer', seed=42):
    """
    Perform stratified sampling to maintain class distribution

    Args:
        data: List of dataset entries
        sample_size: Number of samples to extract
        class_key: Key to use for stratification (default: 'answer')
        seed: Random seed for reproducibility

    Returns:
        List of sampled entries
    """
    random.seed(seed)

    # Group by class
    class_groups = defaultdict(list)
    for entry in data:
        class_label = entry.get(class_key, 'unknown')
        # For agricultural dataset, extract disease name from answer
        # e.g., "Tomato Early Blight" -> "early blight"
        if isinstance(class_label, str):
            # Simple heuristic: take last 2-3 words as disease name
            words = class_label.lower().split()
            if len(words) >= 2:
                disease = ' '.join(words[-2:])
            else:
                disease = words[-1] if words else 'unknown'
            class_groups[disease].append(entry)
        else:
            class_groups['unknown'].append(entry)

    print(f"Found {len(class_groups)} classes")

    # Calculate samples per class
    total_entries = len(data)
    sampled = []

    for disease, entries in class_groups.items():
        # Proportional sampling
        class_ratio = len(entries) / total_entries
        class_sample_size = int(sample_size * class_ratio)

        # Ensure at least 1 sample per class if class has data
        if class_sample_size == 0 and len(entries) > 0:
            class_sample_size = 1

        # Sample from this class
        if class_sample_size >= len(entries):
            class_samples = entries
        else:
            class_samples = random.sample(entries, class_sample_size)

        sampled.extend(class_samples)
        print(f"  {disease}: {len(entries)} total, sampled {len(class_samples)}")

    # If we're short of target, randomly sample more
    if len(sampled) < sample_size:
        remaining_needed = sample_size - len(sampled)
        remaining_data = [e for e in data if e not in sampled]
        if remaining_data:
            additional = random.sample(remaining_data, min(remaining_needed, len(remaining_data)))
            sampled.extend(additional)

    print(f"Stratified sampling complete: {len(sampled)} entries")

    return sampled


def save_dataset(data, output_path):
    """
    Save sampled dataset to JSON file

    Args:
        data: List of dataset entries
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Sample 20k entries from dataset')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--sample_size', type=int, default=20000,
                       help='Number of samples to extract (default: 20000)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified sampling to maintain class distribution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate image paths before sampling')
    parser.add_argument('--class_key', type=str, default='answer',
                       help='Key to use for stratification (default: answer)')

    args = parser.parse_args()

    print("="*80)
    print("Dataset Sampling Script")
    print("="*80)

    # Load dataset
    data = load_dataset(args.input)

    # Validate images if requested
    if args.validate:
        data, invalid_count = validate_images(data)
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid image paths found and removed")

    # Perform sampling
    if args.stratified:
        print(f"\nPerforming stratified sampling...")
        sampled = stratified_sample(data, args.sample_size, args.class_key, args.seed)
    else:
        print(f"\nPerforming random sampling...")
        sampled = random_sample(data, args.sample_size, args.seed)

    # Save results
    save_dataset(sampled, args.output)

    print("="*80)
    print("Sampling complete!")
    print("="*80)


if __name__ == "__main__":
    main()
