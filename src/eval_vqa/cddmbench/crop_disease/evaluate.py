#!/usr/bin/env python3
"""
Evaluate model inference results - Using complete fuzzy matching vocabulary
Based on vocabulary and evaluation logic from training script grpo_agri_vqa_v3.py
"""
import json
import re
from collections import defaultdict

# ===== Plant synonym dictionary (from grpo_agri_vqa_v3.py) =====
PLANT_VARIATIONS = {
    "tomato": [
        "tomato", "tomato plant", "tomatoes", "solanum lycopersicum",
        "lycopersicon esculentum", "nightshade", "tomato leaf", "tomato crop"
    ],
    "potato": [
        "potato", "potato plant", "potatoes", "solanum tuberosum",
        "white potato", "irish potato", "potato tuber"
    ],
    "corn": [
        "corn", "corn plant", "maize", "zea mays",
        "corn crop", "sweet corn", "field corn", "corn leaf"
    ],
    "apple": [
        "apple", "apple tree", "malus domestica",
        "apple crop", "apple leaf", "apple plant"
    ],
    "grape": [
        "grape", "grapevine", "vitis vinifera", "grape plant",
        "grape leaf", "vineyard"
    ],
    "wheat": [
        "wheat", "wheat plant", "triticum aestivum", "wheat crop"
    ],
    "rice": [
        "rice", "rice plant", "oryza sativa", "rice crop"
    ],
    "soybean": [
        "soybean", "soy plant", "glycine max", "soya bean"
    ],
    "bell pepper": [
        "bell pepper", "pepper plant", "capsicum annuum", "sweet pepper", "pepper"
    ],
    "cherry": [
        "cherry", "cherry tree", "prunus avium", "sweet cherry"
    ],
    "peach": [
        "peach", "peach tree", "prunus persica", "peach plant"
    ],
    "strawberry": [
        "strawberry", "strawberry plant", "fragaria"
    ],
    "blueberry": [
        "blueberry", "blueberry plant", "vaccinium"
    ],
    "raspberry": [
        "raspberry", "raspberry plant", "rubus"
    ],
    "pumpkin": [
        "pumpkin", "pumpkin plant", "cucurbita"
    ],
    "orange": [
        "orange", "orange tree", "citrus sinensis", "orange plant"
    ],
}

# ===== Disease synonym dictionary (from grpo_agri_vqa_v3.py) =====
DISEASE_VARIATIONS = {
    "late blight": [
        "late blight", "phytophthora infestans", "phytophthora",
        "oomycete disease", "late blight disease"
    ],
    "early blight": [
        "early blight", "alternaria solani", "alternaria",
        "target spot", "early blight disease", "alternaria leaf spot"
    ],
    "powdery mildew": [
        "powdery mildew", "erysiphales", "white powdery coating",
        "mildew", "powdery mildew fungus"
    ],
    "septoria leaf spot": [
        "septoria leaf spot", "septoria", "leaf spot disease"
    ],
    "mosaic virus": [
        "mosaic virus", "viral mosaic", "mosaic disease",
        "virus infection", "viral disease"
    ],
    "leaf mold": [
        "leaf mold", "fulvia fulva", "tomato leaf mold",
        "fungal leaf mold"
    ],
    "bacterial spot": [
        "bacterial spot", "bacterial disease", "bacterial leaf spot",
        "bacteria infection"
    ],
    "yellow leaf curl virus": [
        "yellow leaf curl virus", "ylcv", "leaf curl virus",
        "yellow leaf curl", "viral leaf curl"
    ],
    "spider mites": [
        "spider mites", "mite damage", "mite infestation"
    ],
    "target spot": [
        "target spot", "corynespora cassiicola", "concentric lesions"
    ],
    "leaf rust": [
        "leaf rust", "rust disease", "rust fungus"
    ],
    "common rust": [
        "common rust", "corn rust", "puccinia sorghi"
    ],
    "northern leaf blight": [
        "northern leaf blight", "turcicum leaf blight", "leaf blight"
    ],
    "gray leaf spot": [
        "gray leaf spot", "grey leaf spot", "cercospora"
    ],
    "grey spot": [
        "grey spot", "gray spot", "cercospora"
    ],
    "leaf scorch": [
        "leaf scorch", "marginal leaf burn", "leaf tip burn"
    ],
    "healthy": [
        "healthy", "no disease", "disease-free", "normal plant",
        "no symptoms", "healthy plant", "not diseased"
    ],
    "black rot": [
        "black rot", "rot disease", "rotting", "fungal rot"
    ],
    "apple scab": [
        "apple scab", "scab disease", "venturia inaequalis"
    ],
    "scab": [
        "scab", "scab disease"
    ],
    "alternaria blotch": [
        "alternaria blotch", "alternaria", "blotch disease"
    ],
    "leaf blight": [
        "leaf blight", "blight disease", "blight"
    ],
    "brown spot": [
        "brown spot", "brown leaf spot", "spot disease"
    ],
    "frog eye leaf spot": [
        "frog eye leaf spot", "frog eye", "leaf spot"
    ],
    "esca": [
        "esca", "esca disease", "black measles"
    ],
    "stem rust": [
        "stem rust", "rust disease", "puccinia"
    ],
    "stripe rust": [
        "stripe rust", "yellow rust", "puccinia striiformis"
    ],
    "loose smut": [
        "loose smut", "smut disease", "ustilago"
    ],
    "blast": [
        "blast", "rice blast", "magnaporthe"
    ],
    "tungro": [
        "tungro", "rice tungro", "viral disease"
    ],
    "bacterial leaf blight": [
        "bacterial leaf blight", "bacterial blight", "blight"
    ],
    "citrus greening": [
        "citrus greening", "huanglongbing", "hlb", "greening disease"
    ],
    "leaf spot": [
        "leaf spot", "spot disease"
    ],
    "cedar apple rust": [
        "cedar apple rust", "rust disease"
    ],
    "root rot": [
        "root rot", "rot disease"
    ],
    "leaf smut": [
        "leaf smut", "smut disease"
    ],
}


def fuzzy_match_score_v3(text, term_list):
    """
    Improved fuzzy matching - 5-tier scoring system
    Returns: 0.0-1.0 score
    """
    if not text or not term_list:
        return 0.0

    text_lower = text.lower()

    # Tier 1: Exact match (1.0)
    for term in term_list:
        if term.lower() in text_lower:
            return 1.0

    # Tier 2: High quality match (0.85)
    # Multi-word terms allow missing 1 word
    for term in term_list:
        words = term.lower().split()
        if len(words) > 1:
            matches = sum(1 for word in words if word in text_lower and len(word) > 3)
            if matches >= len(words) - 1:
                return 0.85

    # Tier 3: Partial match (0.7)
    # Keyword stem matching
    for term in term_list:
        key_words = [w for w in term.lower().split() if len(w) > 4]
        if key_words:
            for kw in key_words:
                if any(kw[:6] in word or word[:6] in kw
                       for word in text_lower.split() if len(word) > 4):
                    return 0.7

    # Tier 4: Keyword match (0.5)
    for term in term_list:
        core_words = [w for w in term.lower().split() if len(w) > 3]
        if any(cw in text_lower for cw in core_words):
            return 0.5

    # Tier 5: Related word match (0.25)
    for term in term_list:
        # Simple stem matching
        for word in term.lower().split():
            if len(word) > 4:
                for text_word in text_lower.split():
                    if len(text_word) > 4 and (word[:4] in text_word or text_word[:4] in word):
                        return 0.25

    return 0.0


def extract_answer_from_solution(solution):
    """Extract <answer> section from solution"""
    answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return solution  # If no <answer> tag, use entire solution


def parse_image_path(image_path):
    """Parse crop and disease from image path"""
    try:
        # Path format: .../Crop,Disease/plant_xxxxx.jpg
        folder_name = image_path.split('/')[-2]
        crop, disease = folder_name.split(',')
        return crop.strip(), disease.strip()
    except:
        return None, None


def evaluate_sample(solution, crop_name, disease_name, threshold=0.6):
    """
    Evaluate single sample
    Returns: (crop_correct, disease_correct, crop_score, disease_score)
    """
    # Extract answer section
    answer_text = extract_answer_from_solution(solution)

    # Normalize crop and disease names to lowercase
    crop_name_lower = crop_name.lower() if crop_name else ""
    disease_name_lower = disease_name.lower() if disease_name else ""

    # Crop matching
    crop_score = 0.0
    crop_correct = 0
    if crop_name_lower in PLANT_VARIATIONS:
        variations = PLANT_VARIATIONS[crop_name_lower]
        crop_score = fuzzy_match_score_v3(answer_text, variations)
        crop_correct = 1 if crop_score >= threshold else 0

    # Disease matching
    disease_score = 0.0
    disease_correct = 0
    if disease_name_lower in DISEASE_VARIATIONS:
        variations = DISEASE_VARIATIONS[disease_name_lower]
        disease_score = fuzzy_match_score_v3(answer_text, variations)
        disease_correct = 1 if disease_score >= threshold else 0

    return crop_correct, disease_correct, crop_score, disease_score


def evaluate_results(results_file, threshold=0.6):
    """
    Evaluate inference results file
    """
    print("="*80)
    print("Evaluating Model Inference Results")
    print("="*80)
    print(f"Results file: {results_file}")
    print(f"Matching threshold: {threshold}")
    print("="*80)

    # Read results file
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nTotal samples: {len(data)}")

    # Statistics variables
    total_samples = 0
    crop_correct_count = 0
    disease_correct_count = 0

    crop_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    disease_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Evaluate each sample
    print("\nStarting evaluation...")
    for i, entry in enumerate(data):
        image_path = entry.get('image', '')
        solution = entry.get('solution', '')

        # Parse crop and disease
        crop_name, disease_name = parse_image_path(image_path)

        if not crop_name or not disease_name:
            continue

        # Evaluate
        crop_correct, disease_correct, crop_score, disease_score = evaluate_sample(
            solution, crop_name, disease_name, threshold
        )

        # Statistics
        total_samples += 1
        crop_correct_count += crop_correct
        disease_correct_count += disease_correct

        crop_stats[crop_name]['correct'] += crop_correct
        crop_stats[crop_name]['total'] += 1

        disease_stats[disease_name]['correct'] += disease_correct
        disease_stats[disease_name]['total'] += 1

        # Print detailed info for first 5 samples
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"  Image: {image_path}")
            print(f"  Expected: Crop={crop_name}, Disease={disease_name}")
            print(f"  Solution: {solution[:200]}...")
            print(f"  Crop Score: {crop_score:.2f} ({'✓' if crop_correct else '✗'})")
            print(f"  Disease Score: {disease_score:.2f} ({'✓' if disease_correct else '✗'})")

    # Calculate accuracy
    crop_accuracy = crop_correct_count / total_samples if total_samples > 0 else 0
    disease_accuracy = disease_correct_count / total_samples if total_samples > 0 else 0

    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Total samples: {total_samples}")
    print(f"Crop identification accuracy: {crop_accuracy:.2%} ({crop_correct_count}/{total_samples})")
    print(f"Disease identification accuracy: {disease_accuracy:.2%} ({disease_correct_count}/{total_samples})")
    print("="*80)

    # Accuracy by crop type
    print("\nCrop identification accuracy by type:")
    for crop, stats in sorted(crop_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {crop:20s}: {acc:>6.2%} ({stats['correct']:>3d}/{stats['total']:>3d})")

    # Disease accuracy (Top 15)
    print("\nDisease identification accuracy by type (Top 15):")
    disease_items = sorted(disease_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:15]
    for disease, stats in disease_items:
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {disease:30s}: {acc:>6.2%} ({stats['correct']:>3d}/{stats['total']:>3d})")

    return {
        'total_samples': total_samples,
        'crop_accuracy': crop_accuracy,
        'disease_accuracy': disease_accuracy,
        'crop_stats': dict(crop_stats),
        'disease_stats': dict(disease_stats),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True,
                        help='Inference results JSON file path')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Fuzzy matching threshold (default 0.6)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output evaluation results JSON file path')

    args = parser.parse_args()

    # Evaluate
    results = evaluate_results(args.results, args.threshold)

    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
