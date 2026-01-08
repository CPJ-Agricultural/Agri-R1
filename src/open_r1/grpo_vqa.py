# GRPO Agricultural VQA Training - Data-driven Reward Function Design
# Based on deep dataset analysis, optimized reward structure for high score convergence
# Core improvements: Answer-dominant (67%) + Dynamic evaluation (Diagnosis vs Treatment) + Reasoning logic quality (17%)

import os
import re
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer.dynamic_callbacks import (
    DynamicBatchSizeCallback,
    EarlyStepCacheClearCallback,
    GradientMonitoringCallback,
    KLDivergenceMonitoringCallback,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# ===== Plant Synonym Library =====
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
}

# ===== Disease Synonym Library =====
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
    "leaf scorch": [
        "leaf scorch", "marginal leaf burn", "leaf tip burn"
    ],
    "healthy": [
        "healthy", "no disease", "disease-free", "normal plant",
        "no symptoms", "healthy plant"
    ],
    "black rot": [
        "black rot", "rot disease", "rotting", "fungal rot"
    ],
    "apple scab": [
        "apple scab", "scab disease", "venturia inaequalis"
    ],
    "alternaria blotch": [
        "alternaria blotch", "alternaria", "blotch disease"
    ],
    "leaf blight": [
        "leaf blight", "blight disease", "blight"
    ],
}

PLANT_KEYWORDS = list(PLANT_VARIATIONS.keys())
DISEASE_KEYWORDS = list(DISEASE_VARIATIONS.keys())

# ===== Treatment Keywords Library =====
TREATMENT_KEYWORDS = {
    'pesticides': [
        'fungicide', 'copper', 'chlorothalonil', 'mancozeb', 'metalaxyl',
        'mefenoxam', 'azoxystrobin', 'propiconazole', 'tebuconazole',
        'captan', 'thiophanate', 'benomyl', 'carbendazim',
        'bordeaux mixture', 'pesticide', 'wettable powder'
    ],
    'cultural_practices': [
        'rotation', 'crop rotation', 'air circulation', 'circulation',
        'spacing', 'debris', 'resistant varieties', 'resistant',
        'watering', 'overhead watering', 'drainage', 'mulching',
        'prune', 'pruning', 'remove infected', 'sanitation'
    ],
    'application_methods': [
        'spray', 'application', 'apply', 'protective gear',
        'label instructions', 'dosage', 'rate', 'dilution',
        'times solution', 'wettable powder', 'emulsion'
    ],
    'application_timing': [
        'timing', 'early stage', 'first sign', 'early growing season',
        'before disease', 'onset', 'every 7', 'every 10', 'every 14',
        'days', 'repeat', 'consecutive'
    ]
}

# Precompile regex patterns for performance
RE_THINK = re.compile(r'<think>(.*?)</think>', re.DOTALL)
RE_ANSWER = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
RE_STEP = re.compile(r'step\s*\d+[:\.]', re.IGNORECASE)

SYSTEM_PROMPT = """You are a plant disease management expert. Carefully analyze the given image and question, following these guidelines:

## Core Requirements:
1. Output must be in English and structured into explicit steps labeled "Step 1: … Step 2: …"
2. For PREVENTION/CONTROL questions: Focus ONLY on method reasoning - DO NOT re-diagnose
3. For IDENTIFICATION questions: Focus on visual evidence and diagnostic reasoning
4. Keep reasoning concise and practical

## Question Type Guidelines:

### FOR PREVENTION/CONTROL METHODS QUESTIONS:
- Step 1: Analyze disease characteristics that influence control strategies
- Step 2: Recommend cultural/preventive practices based on disease biology
- Step 3: Outline chemical control timing and selection
- Step 4: Integrate application methods and safety precautions

### FOR DISEASE IDENTIFICATION QUESTIONS:
- Step 1: Plant identification based on morphological features
- Step 2: Symptom observation and description
- Step 3: Disease pattern analysis
- Step 4: Preliminary diagnosis with confidence level

## CRITICAL RULES:
- If question asks about CONTROL/MANAGEMENT/PREVENTION/TREATMENT/METHODS: Use PREVENTION/CONTROL guideline
- If question asks about IDENTIFICATION/WHAT/NAME/DISEASE: Use IDENTIFICATION guideline
- NEVER mix guidelines - choose one based on question type

## Output Format:
<think>Step 1: ... Step 2: ... Step 3: ... Step 4: ...</think>
<answer>Your final answer here</answer>"""


# ===== Improved Fuzzy Matching V3: 5-tier more lenient =====
def fuzzy_match_score_v3(text, term_list):
    """
    Improved fuzzy matching - more lenient, adapted to low keyword density characteristics of dataset
    Returns: 0.0-1.0 score (5 tiers)
    """
    text_lower = text.lower()

    # Tier 1: Exact match (1.0)
    for term in term_list:
        if term.lower() in text_lower:
            return 1.0

    # Tier 2: High quality match (0.85) - narrower spacing
    # Multi-word terms allow missing 1 word
    for term in term_list:
        words = term.lower().split()
        if len(words) > 1:
            matches = sum(1 for word in words if word in text_lower and len(word) > 3)
            if matches >= len(words) - 1:
                return 0.85  # Increased from 0.8 to 0.85

    # Tier 3: Partial match (0.7) - narrower spacing
    # Keyword stem matching
    for term in term_list:
        key_words = [w for w in term.lower().split() if len(w) > 4]
        if key_words:
            for kw in key_words:
                if any(kw[:6] in word or word[:6] in kw
                       for word in text_lower.split() if len(word) > 4):
                    return 0.7  # Increased from 0.6 to 0.7

    # Tier 4: Keyword match (0.5) - narrower spacing
    for term in term_list:
        core_words = [w for w in term.lower().split() if len(w) > 3]
        if any(cw in text_lower for cw in core_words):
            return 0.5  # Increased from 0.35 to 0.5

    # Tier 5: Weak relevance (0.25) - narrower spacing
    related_terms = {
        'blight': ['disease', 'infection', 'lesion'],
        'rot': ['decay', 'rotting', 'damaged'],
        'spot': ['lesion', 'mark', 'blotch'],
    }
    for term in term_list:
        for key, synonyms in related_terms.items():
            if key in term.lower() and any(syn in text_lower for syn in synonyms):
                return 0.25  # Increased from 0.15 to 0.25

    return 0.0


def extract_plant_disease(answer_text):
    """Extract plant and disease from answer text"""
    text_lower = answer_text.lower()

    # Extract plant name
    plant = None
    for p in PLANT_KEYWORDS:
        if p in text_lower:
            plant = p.title()
            break

    # Extract disease name (prioritize longest match)
    disease = None
    max_len = 0
    for d in DISEASE_KEYWORDS:
        pattern = r'\b' + re.escape(d) + r'\b'
        if re.search(pattern, text_lower) and len(d) > max_len:
            disease = d.title()
            max_len = len(d)

    return plant, disease


def is_treatment_question(problem_text):
    """Determine if it's a treatment question - expanded keywords"""
    problem_lower = problem_text.lower()
    treatment_indicators = [
        'control', 'prevent', 'treatment', 'treat', 'manage',
        'cure', 'remedy', 'how to', 'what should', 'best way',
        'prevention', 'management', 'methods', 'measures',
        'combat', 'eliminate', 'reduce', 'stop'
    ]
    return any(word in problem_lower for word in treatment_indicators)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "answer_keyword", "reasoning"],
        metadata={"help": "List of reward functions. Possible values: 'format', 'answer_keyword', 'reasoning'"},
    )
    max_pixels: Optional[int] = field(
        default=147456,  # 384×384 correct resolution
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def format_reward(completions, **kwargs):
    """
    Format quality evaluation: [0, 0.5]
    Evaluate structure integrity and content quality
    """
    rewards = []
    for completion in completions:
        score = 0.0
        content = completion[0]["content"]

        # 1. Basic structure (0.15) - must have think and answer
        if RE_THINK.search(content) and RE_ANSWER.search(content):
            score += 0.15
        else:
            # Missing basic structure, directly return 0
            rewards.append(0.0)
            continue

        # 2. Step structure integrity (0.15)
        think_match = RE_THINK.search(content)
        if think_match:
            think_text = think_match.group(1)
            step_count = len(re.findall(r'Step \d+:', think_text))
            if step_count >= 4:
                score += 0.15
            elif step_count >= 3:
                score += 0.12
            elif step_count >= 2:
                score += 0.08
            else:
                score += 0.03  # At least has 1 step

        # 3. Step content quality (0.10) - each Step should have substantial content
        if think_match:
            steps = re.findall(r'Step \d+:\s*(.+?)(?=Step \d+:|$)', think_text, re.DOTALL)
            quality_steps = 0
            for step_content in steps:
                # Each Step should have at least 30 characters of substantial content
                if len(step_content.strip()) >= 30:
                    quality_steps += 1

            if quality_steps >= 4:
                score += 0.10
            elif quality_steps >= 3:
                score += 0.08
            elif quality_steps >= 2:
                score += 0.05

        # 4. Think reasonable length (0.05)
        # Dataset average 487 characters, set reasonable range 150-800
        if think_match:
            think_len = len(think_match.group(1))
            if 150 <= think_len <= 800:
                score += 0.05
            elif 100 <= think_len <= 1000:
                score += 0.03
            elif think_len >= 80:
                score += 0.01

        # 5. Answer quality (0.05)
        answer_match = RE_ANSWER.search(content)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_len = len(answer_text)
            # Dataset average 69 characters, set reasonable range 15-200
            if 15 <= answer_len <= 200:
                score += 0.05
            elif 10 <= answer_len <= 300:
                score += 0.03
            elif answer_len >= 5:
                score += 0.01

        rewards.append(min(0.5, score))

    return rewards


def answer_keyword_reward(completions, solution, **kwargs):
    """
    Answer keyword matching - main component: [0, 2.0]

    Dynamic evaluation:
    - Diagnosis questions (99.6%): plant(0.8) + disease(1.2)
    - Treatment questions (0.4%): treatment keywords(2.0)

    Accurately determine question type through problem text for optimal evaluation effect
    """
    rewards = []

    # Get problem field from kwargs (trainer passes all dataset fields)
    problems = kwargs.get('problem', [None] * len(completions))

    for completion, sol, prob in zip(completions, solution, problems):
        score = 0.0
        content = completion[0]["content"]

        # Extract generated answer and reference answer
        gen_answer_match = RE_ANSWER.search(content)
        ref_answer_match = RE_ANSWER.search(sol)

        if not (gen_answer_match and ref_answer_match):
            rewards.append(0.0)
            continue

        gen_answer = gen_answer_match.group(1).strip()
        ref_answer = ref_answer_match.group(1).strip()

        # Determine question type: use problem field for accurate judgment (most accurate method)
        if prob and is_treatment_question(prob):
            # [Treatment question] Evaluate treatment keywords in Answer
            score = evaluate_treatment_in_answer(gen_answer, ref_answer)
        else:
            # [Diagnosis question] Evaluate plant + disease in Answer
            # Default to diagnosis question (99.6% of samples are diagnosis questions)
            score = evaluate_diagnosis_in_answer(gen_answer, ref_answer)

        rewards.append(min(2.0, score))

    return rewards


def evaluate_diagnosis_in_answer(gen_answer, ref_answer):
    """
    Evaluate diagnosis answer: plant + disease
    Returns: [0, 2.0]
    """
    score = 0.0
    gen_lower = gen_answer.lower()
    ref_lower = ref_answer.lower()

    # Extract key information from reference answer
    ref_plant, ref_disease = extract_plant_disease(ref_answer)

    # 1. Plant matching (0.8)
    if ref_plant and ref_plant.lower() in PLANT_VARIATIONS:
        variations = PLANT_VARIATIONS[ref_plant.lower()]
        plant_score = fuzzy_match_score_v3(gen_lower, variations)
        score += 0.8 * plant_score

    # 2. Disease/condition matching (1.2)
    if ref_disease:
        if ref_disease.lower() == "healthy":
            # Special handling for healthy status
            if any(w in gen_lower for w in ['healthy', 'no disease', 'disease-free']):
                score += 1.2
        elif ref_disease.lower() in DISEASE_VARIATIONS:
            variations = DISEASE_VARIATIONS[ref_disease.lower()]
            disease_score = fuzzy_match_score_v3(gen_lower, variations)
            score += 1.2 * disease_score

    return score


def evaluate_treatment_in_answer(gen_answer, ref_answer):
    """
    Evaluate treatment answer: treatment keywords
    Returns: [0, 2.0]

    Scoring criteria:
    - Pesticide keywords: 0.6
    - Cultural practices: 0.5
    - Application methods: 0.5
    - Application timing: 0.4
    """
    score = 0.0
    gen_lower = gen_answer.lower()

    # 1. Pesticides (0.6)
    pesticide_count = sum(1 for kw in TREATMENT_KEYWORDS['pesticides'] if kw in gen_lower)
    if pesticide_count >= 3:
        score += 0.6
    elif pesticide_count >= 2:
        score += 0.45
    elif pesticide_count >= 1:
        score += 0.3

    # 2. Cultural practices (0.5)
    cultural_count = sum(1 for kw in TREATMENT_KEYWORDS['cultural_practices'] if kw in gen_lower)
    if cultural_count >= 3:
        score += 0.5
    elif cultural_count >= 2:
        score += 0.35
    elif cultural_count >= 1:
        score += 0.2

    # 3. Application methods (0.5)
    method_count = sum(1 for kw in TREATMENT_KEYWORDS['application_methods'] if kw in gen_lower)
    if method_count >= 3:
        score += 0.5
    elif method_count >= 2:
        score += 0.35
    elif method_count >= 1:
        score += 0.2

    # 4. Application timing (0.4)
    timing_count = sum(1 for kw in TREATMENT_KEYWORDS['application_timing'] if kw in gen_lower)
    if timing_count >= 3:
        score += 0.4
    elif timing_count >= 2:
        score += 0.3
    elif timing_count >= 1:
        score += 0.15

    return score


def reasoning_reward_v2(completions, **kwargs):
    """
    Reasoning quality evaluation - focus on logical coherence: [0, 0.5]
    """
    rewards = []

    for completion in completions:
        score = 0.0
        content = completion[0]["content"]

        think_match = RE_THINK.search(content)
        if not think_match:
            rewards.append(0.0)
            continue

        think_text = think_match.group(1).strip()

        # 1. Logical coherence (0.25)
        logic_score = evaluate_logical_coherence(think_text)
        score += 0.25 * logic_score

        # 2. Professionalism (0.15)
        prof_score = evaluate_professionalism_context(think_text)
        score += 0.15 * prof_score

        # 3. Completeness (0.10)
        completeness_score = evaluate_reasoning_completeness(think_text)
        score += 0.10 * completeness_score

        rewards.append(min(0.5, score))

    return rewards


def evaluate_logical_coherence(text):
    """
    Evaluate logical coherence [0, 1.0]
    Lower threshold for easier high scores
    """
    score = 0.0
    text_lower = text.lower()

    # Check causal relationship usage - lower requirements
    causal_patterns = [
        r'observe.*?(?:because|since|due to)',
        r'symptom.*?indicat',
        r'(?:because|since).*?therefore',
        r'(?:show|display).*?(?:indicate|suggest)',  # Added
        r'(?:based on|from).*?(?:conclude|determine)',  # Added
    ]

    for pattern in causal_patterns:
        if re.search(pattern, text_lower, re.DOTALL):
            score += 0.5  # Increased from 0.4 to 0.5
            break

    # Check step continuity - lower difficulty
    steps = re.findall(r'step\s*\d+[:\.]?\s*([^.]*(?:\.[^.]*){0,2})', text_lower, re.IGNORECASE)
    if len(steps) >= 2:
        for i in range(1, len(steps)):
            prev_words = set(steps[i-1].split()[:20])
            curr_words = set(steps[i].split()[:15])
            prev_words = {w for w in prev_words if len(w) > 3}  # Reduced from 4 to 3
            if prev_words & curr_words:
                score += 0.4  # Increased from 0.3 to 0.4
                break

    return min(1.0, score)


def evaluate_professionalism_context(text):
    """
    Evaluate professional term context usage [0, 1.0]
    Expanded patterns, lower threshold
    """
    score = 0.0
    text_lower = text.lower()

    professional_contexts = [
        (r'(?:pathogen|fungus|bacteria|virus)\s+(?:cause|infect|spread|attack)', 0.35),  # Increased
        (r'symptom.*?(?:include|show|indicate|appear|present)', 0.35),  # Increased + expanded
        (r'(?:diagnosis|identify).*?based on', 0.30),
        (r'(?:lesion|spot|blight).*?(?:circular|brown|yellow|dark|light)', 0.25),  # Increased + expanded
        (r'(?:leaf|plant|crop).*?(?:damage|affect|infect|disease)', 0.20),  # Added
        (r'(?:control|treat|prevent).*?(?:spray|apply|use|fungicide)', 0.20),  # Added
    ]

    for pattern, weight in professional_contexts:
        if re.search(pattern, text_lower):
            score += weight

    return min(1.0, score)


def evaluate_reasoning_completeness(text):
    """
    Evaluate reasoning completeness [0, 1.0]
    Check for complete reasoning chain: observation → analysis → conclusion
    Expanded based on dataset high-frequency words
    """
    score = 0.0
    text_lower = text.lower()

    # Observation phase - expanded with dataset high-frequency words
    has_observation = any(word in text_lower for word in
                         ['observe', 'see', 'visible', 'appear', 'show', 'display',
                          'symptoms', 'lesions', 'spots', 'brown', 'yellowing',  # Dataset high-frequency
                          'affected', 'leaf', 'leaves'])  # Dataset high-frequency

    # Analysis phase - expanded with dataset high-frequency words
    has_analysis = any(word in text_lower for word in
                      ['analyze', 'assess', 'examine', 'indicate', 'suggest', 'pattern',
                       'diagnosis', 'infection', 'disease', 'caused',  # Dataset high-frequency
                       'bacterial', 'fungal'])  # Dataset high-frequency

    # Conclusion phase - expanded with dataset high-frequency words
    has_conclusion = any(word in text_lower for word in
                        ['therefore', 'conclude', 'diagnosis', 'identify', 'determine',
                         'control', 'prevent', 'fungicides', 'apply',  # Dataset high-frequency
                         'preventive', 'spray'])  # Dataset high-frequency

    # Increased weight
    if has_observation:
        score += 0.40  # Increased from 0.35
    if has_analysis:
        score += 0.35  # Maintained
    if has_conclusion:
        score += 0.35  # Increased from 0.30

    return min(1.0, score)


reward_funcs_registry = {
    "format": format_reward,
    "answer_keyword": answer_keyword_reward,
    "reasoning": reasoning_reward_v2,
}


class PromptDatasetWrapper:
    def __init__(self, hf_dataset, system_prompt):
        self.dataset = hf_dataset
        self.system_prompt = system_prompt

        # Cache validated image paths at initialization
        self._image_path_cache = {}
        print("Validating and caching image paths...")
        for idx in range(len(hf_dataset)):
            img_path = hf_dataset[idx]["image"]
            self._image_path_cache[idx] = self._validate_image_path(img_path)
        print(f"✓ Cached {len(self._image_path_cache)} image paths")

    def _validate_image_path(self, img_path):
        """Validate image path once during initialization"""
        if "/agri-r1/src/data/" in img_path:
            img_path = img_path.replace("/agri-r1/src/data/", "/agri-r1/data/")
        elif "/src/data/" in img_path:
            img_path = img_path.replace("/src/data/", "/data/")

        if not os.path.exists(img_path):
            alternatives = [
                img_path.replace("/root/autodl-tmp/agri-r1/", "/root/autodl-tmp/agri/src/r1-v/agri-r1/"),
                os.path.join("/root/autodl-tmp/agri/src/r1-v/agri-r1/data", os.path.basename(img_path)),
                img_path.replace("/agri-r1/data/", "/agri-r1/src/data/"),
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    img_path = alt
                    break
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        return img_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Use cached validated path
        example["image"] = self._image_path_cache[idx]

        example["prompt"] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Question: {example['problem']}\n\nPlease analyze the image and provide your answer in the required format."},
                ],
            },
        ]
        return example


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    print("\n" + "="*80)
    print("GRPO Agricultural VQA Training - V3 Data-Driven Reward Design")
    print("="*80)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output: {training_args.output_dir}")
    print(f"Batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} × {training_args.world_size} GPUs")
    print(f"GRPO generations: {training_args.num_generations}")
    print(f"Reward components:")
    print(f"  - Format:         [0, 0.5]  (17%) - Structure evaluation")
    print(f"  - Answer Keyword: [0, 2.0]  (67%) - Dynamic (Diagnosis/Treatment)")
    print(f"  - Reasoning:      [0, 0.5]  (17%) - Logic quality")
    print(f"  TOTAL:            [0, 3.0]")
    print(f"DeepSpeed: {training_args.deepspeed}")
    print("="*80)
    print()

    dataset_dict = load_from_disk(script_args.dataset_name)

    print("\n" + "="*80)
    print("Dataset Validation")
    print("="*80)
    required_fields = ["image", "problem", "solution"]

    # Handle single Dataset or DatasetDict
    if hasattr(dataset_dict, 'column_names'):
        # Is Dataset
        train_dataset_raw = dataset_dict
    else:
        # Is DatasetDict
        train_dataset_raw = dataset_dict[script_args.dataset_train_split]

    assert all(field in train_dataset_raw.column_names for field in required_fields), f"Missing required fields: {required_fields}"
    print(f"✓ Required fields present: {required_fields}")
    print(f"✓ Dataset size: {len(train_dataset_raw)} samples")
    print(f"✓ First sample image path: {train_dataset_raw[0]['image']}")
    print("="*80)
    print()

    wrapped_dataset = PromptDatasetWrapper(train_dataset_raw, SYSTEM_PROMPT)

    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = {}
    if "torch_dtype" not in training_args.model_init_kwargs:
        import torch
        training_args.model_init_kwargs["torch_dtype"] = torch.bfloat16

    # ============================================================================
    # Initialize Training Callbacks
    # ============================================================================
    print("\n" + "="*80)
    print("Initializing Training Callbacks")
    print("="*80)

    # Dynamic batch size: smaller batch during warmup to avoid initial memory pressure
    # For Format3: Step 0-5 use grad_accum=2, Step 6+ use grad_accum=4
    # For Format1: Step 0-10 use grad_accum=2, Step 11+ use grad_accum=4
    initial_grad_accum = int(os.getenv("INITIAL_GRAD_ACCUM", "2"))
    final_grad_accum = training_args.gradient_accumulation_steps
    warmup_steps = int(os.getenv("WARMUP_STEPS", "5"))

    callbacks = [
        DynamicBatchSizeCallback(
            initial_grad_accum_steps=initial_grad_accum,
            final_grad_accum_steps=final_grad_accum,
            warmup_steps=warmup_steps
        ),
        EarlyStepCacheClearCallback(
            clear_until_step=warmup_steps + 3  # Clear cache a bit longer than warmup
        ),
        GradientMonitoringCallback(
            grad_norm_threshold=20.0  # Warn if gradient norm exceeds 20
        ),
        KLDivergenceMonitoringCallback(
            check_after_steps=3  # Check KL after 3 steps
        ),
    ]

    print(f"✓ Dynamic Batch Size: grad_accum {initial_grad_accum}→{final_grad_accum} at step {warmup_steps}")
    print(f"✓ Early Cache Clear: steps 0-{warmup_steps + 3}")
    print(f"✓ Gradient Monitoring: threshold={20.0}")
    print(f"✓ KL Divergence Monitoring: enabled")
    print("="*80 + "\n")

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=wrapped_dataset,
        callbacks=callbacks,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    final_save_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    print(f"\n✓ Final model saved to: {final_save_path}")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
