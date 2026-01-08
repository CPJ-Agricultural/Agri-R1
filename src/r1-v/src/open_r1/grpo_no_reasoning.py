# GRPO Agricultural VQA Training - No REASONING Version
# Data-driven reward function design without chain-of-thought reasoning
# Core: Answer-dominant evaluation (100%) - Dynamic assessment (Diagnosis vs Treatment)

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
RE_ANSWER = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

SYSTEM_PROMPT = """You are a plant disease management expert. Carefully analyze the given image and question, and provide a direct, professional answer.

## Core Requirements:
1. Output must be in English
2. For PREVENTION/CONTROL questions: Provide actionable treatment recommendations
3. For IDENTIFICATION questions: Provide clear plant and disease identification
4. Keep answers concise and professional

## Question Type Guidelines:

### FOR PREVENTION/CONTROL METHODS QUESTIONS:
Provide comprehensive control strategies including:
- Cultural practices (rotation, spacing, sanitation)
- Chemical control options (fungicides, timing, application methods)
- Preventive measures

### FOR DISEASE IDENTIFICATION QUESTIONS:
Provide clear identification including:
- Plant type
- Disease/condition name
- Key diagnostic symptoms

## Output Format:
<answer>Your direct, professional answer here</answer>

## Important Notes:
- Always identify the plant species first (for identification questions)
- Be specific about disease names (e.g., "Late Blight" not just "blight")
- Include observable symptoms when identifying diseases
- For control questions, provide practical, actionable advice"""


# ===== Improved Fuzzy Matching V3: 5-tier system =====
def fuzzy_match_score_v3(text, term_list):
    """
    Improved fuzzy matching - more lenient, adapted to low keyword density in dataset
    Returns: 0.0-1.0 score (5 tiers)
    """
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

    # Tier 5: Weak relevance (0.25)
    related_terms = {
        'blight': ['disease', 'infection', 'lesion'],
        'rot': ['decay', 'rotting', 'damaged'],
        'spot': ['lesion', 'mark', 'blotch'],
    }
    for term in term_list:
        for key, synonyms in related_terms.items():
            if key in term.lower() and any(syn in text_lower for syn in synonyms):
                return 0.25

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

    # Extract disease name (prioritize longest disease name)
    disease = None
    max_len = 0
    for d in DISEASE_KEYWORDS:
        pattern = r'\b' + re.escape(d) + r'\b'
        if re.search(pattern, text_lower) and len(d) > max_len:
            disease = d.title()
            max_len = len(d)

    return plant, disease


def is_treatment_question(problem_text):
    """Determine if question is about treatment - expanded keywords"""
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
        default_factory=lambda: ["format", "answer_keyword"],
        metadata={"help": "List of reward functions. Possible values: 'format', 'answer_keyword'"},
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
    Format quality assessment: [0, 1.0]
    Evaluate structure integrity and content quality (no REASONING version)
    """
    rewards = []
    for completion in completions:
        score = 0.0
        content = completion[0]["content"]

        # 1. Basic structure (0.4) - must have answer tags
        if RE_ANSWER.search(content):
            score += 0.4
        else:
            # Missing basic structure, return 0
            rewards.append(0.0)
            continue

        # 2. Answer quality (0.6)
        answer_match = RE_ANSWER.search(content)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_len = len(answer_text)

            # Evaluate answer length and quality
            # Optimal range: 30-300 characters for direct answers
            if 30 <= answer_len <= 300:
                score += 0.6
            elif 20 <= answer_len <= 400:
                score += 0.45
            elif 15 <= answer_len <= 500:
                score += 0.3
            elif answer_len >= 10:
                score += 0.15

        rewards.append(min(1.0, score))

    return rewards


def answer_keyword_reward(completions, solution, **kwargs):
    """
    Answer keyword matching - main component: [0, 2.0]

    Dynamic evaluation:
    - Diagnosis questions (99.6%): plant(0.8) + disease(1.2)
    - Treatment questions (0.4%): treatment keywords(2.0)

    Accurately determine question type through problem text for optimal evaluation
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

        # Determine question type: use problem field for accurate judgment
        if prob and is_treatment_question(prob):
            # [Treatment question] Evaluate treatment keywords in answer
            score = evaluate_treatment_in_answer(gen_answer, ref_answer)
        else:
            # [Diagnosis question] Evaluate plant + disease in answer
            # Default to diagnosis (99.6% of samples are diagnosis questions)
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


reward_funcs_registry = {
    "format": format_reward,
    "answer_keyword": answer_keyword_reward,
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
    print("GRPO Agricultural VQA Training - No REASONING Version")
    print("="*80)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output: {training_args.output_dir}")
    print(f"Batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} × {training_args.world_size} GPUs")
    print(f"GRPO generations: {training_args.num_generations}")
    print(f"Reward components:")
    print(f"  - Format:         [0, 1.0]  (33%) - Structure evaluation (answer only)")
    print(f"  - Answer Keyword: [0, 2.0]  (67%) - Dynamic (Diagnosis/Treatment)")
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
