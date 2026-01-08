# Prompts and Evaluation Guide

This document provides comprehensive details on the prompts and evaluation methods used in the Agri-R1 project, covering Reasoning (Chain-of-Thought) generation, Reasoning enhancement, and GRPO (Group Relative Policy Optimization) training.

---

## Table of Contents

1. [Stage 1: Reasoning Data Generation](#stage-1-reasoning-data-generation)
2. [Stage 1.5: Reasoning Enhancement](#stage-15-reasoning-enhancement)
3. [Stage 2: GRPO Training](#stage-2-grpo-training)
4. [Evaluation Metrics](#evaluation-metrics)

---

## Stage 1: Reasoning Data Generation

### System Prompt

```
You are a plant disease management expert. Generate clear, step-by-step reasoning for agricultural questions.

Core Requirements:
1. Output must be in English and structured into 3-4 explicit steps labeled "Step 1: … Step 2: …"
2. For PREVENTION/CONTROL questions: Focus ONLY on method reasoning - DO NOT re-diagnose
3. For IDENTIFICATION questions: Focus on visual evidence and diagnostic reasoning
4. Keep reasoning concise and practical

Question Type Guidelines:

FOR PREVENTION/CONTROL METHODS QUESTIONS:
- Step 1: Analyze disease characteristics that influence control strategies
- Step 2: Recommend cultural/preventive practices based on disease biology
- Step 3: Outline chemical control timing and selection
- Step 4: Integrate application methods and safety precautions

FOR DISEASE IDENTIFICATION QUESTIONS:
- Step 1: Plant identification based on morphological features
- Step 2: Symptom observation and description
- Step 3: Disease pattern analysis
- Step 4: Preliminary diagnosis with confidence level

CRITICAL RULES:
- If question asks about CONTROL/MANAGEMENT/PREVENTION/TREATMENT/METHODS: Use PREVENTION/CONTROL guideline
- If question asks about IDENTIFICATION/WHAT/NAME: Use IDENTIFICATION guideline
- NEVER mix guidelines - choose one based on question type

Output Format:
{"think": "Step 1: … Step 2: … Step 3: …"}

Output only the JSON string.
```

### Few-Shot Examples

#### Prevention/Control Example

**Question:** What effective prevention and control methods can be applied to Apple Brown Spot?

**Answer:**
```
(1) Prune appropriately, improve drainage, and enhance ventilation and light penetration in the orchard.
(2) In fall and winter, clean up fallen leaves and diseased branches and leaves in the orchard, and bury or burn them deeply.
(3) Apply protective fungicides starting in mid-May, every 15 days, for a total of 3-4 applications. Common fungicides include Bordeaux mixture (1:2:200), 30% Captan 500x solution, 77% Kocide 800x solution, 70% Thiophanate-methyl 800x solution, 70% Mancozeb 500x solution, and 75% Chlorothalonil 800x solution.
```

**Think (Reasoning):**
```
Step 1: Based on fungal disease characteristics, recommend pruning for improved air circulation to reduce humidity.
Step 2: Advise sanitation measures in fall/winter to eliminate overwintering fungal spores in plant debris.
Step 3: Schedule fungicide applications from mid-May to target primary infection periods during the growing season.
Step 4: Select appropriate fungicide combinations and note application precautions for comprehensive disease management.
```

#### Identification Example

**Question:** What is the content of this picture?

**Answer:** This image shows an apple leaf affected by Alternaria Blotch.

**Think (Reasoning):**
```
Step 1: Identify plant - leaf ovate with serrated margin and pinnate venation, consistent with apple leaf morphology.
Step 2: Describe symptoms - multiple circular brown lesions with yellowish halos scattered across leaf surface.
Step 3: Assess distribution - lesions measure approximately 2-5 mm in diameter and cover about 20% of visible leaf area.
Step 4: Preliminary diagnosis - Alternaria Blotch based on characteristic lesion appearance; confidence: medium.
```

#### Verification Example

**Question:** Is this leaf from a pear tree?

**Answer:** No, this is an apple leaf.

**Think (Reasoning):**
```
Step 1: Analyze leaf shape - ovate form with pointed tip differs from pear's broader shape.
Step 2: Examine margins - fine serrations present, unlike pear's finer serration pattern.
Step 3: Study venation - pinnate pattern with 45-degree branching typical of apple.
Step 4: Confirm identification - all morphological features consistent with apple leaf; confidence: high.
```

### Question Type Detection

The system automatically detects question types using keyword matching:

**Prevention/Control Keywords:**
- control, prevention, management, treatment, methods, method
- how to, what measures, how can, ways to, strategies
- protect, avoid, reduce, eliminate, combat, fight

**Identification Keywords:**
- what is, identify, name, diagnose, what disease
- what kind, what type, what are the symptoms

**Default:** If ambiguous, defaults to prevention/control based on dataset characteristics.

### API Configuration

```python
API_BASE = "your-api-endpoint"
API_KEY = "your-api-key"
MODEL = "deepseek-ai/DeepSeek-VL2"  # or "gpt-4o", "claude-3.5"

# Generation parameters
temperature = 0.3
max_tokens = 1024
```

### Processing Pipeline

1. **Image Preprocessing**: Resize to 384×384 pixels
2. **Dataset Sampling**: Sample training data (e.g., 20k samples)
3. **Question Type Detection**: Classify as identification or prevention/control
4. **Prompt Construction**: Build prompt with appropriate few-shot examples
5. **API Call**: Generate Reasoning using VLM (GPT-4o/Claude/DeepSeek)
6. **JSON Parsing**: Extract and validate `{"think": "..."}` format
7. **Quality Filtering**: Ensure 3-4 steps with substantial content

---

## Stage 1.5: Reasoning Enhancement

### Evaluation System Prompt

```
You are an expert evaluator for agricultural image captions. Please evaluate the quality of the image caption based on the following criteria:

Evaluation Criteria:
1. Accuracy: Correct identification of plant and disease/pest (or confirmation of health)
2. Completeness: Inclusion of all key elements (plant type, disease type, symptoms, severity)
3. Detail: Specific description of symptoms (location, shape, color, extent, quantity, etc.)
4. Relevance: Information is relevant for agricultural diagnosis and treatment
5. Clarity: Clear, concise, and professional language (80-120 words)

Rating Scale:
- 1-3: Poor (vague, inaccurate, or missing key information)
- 4-6: Fair (some useful information but incomplete or partially inaccurate)
- 7-8: Good (clear, mostly accurate, and relevant)
- 9-10: Excellent (precise, accurate, and highly relevant for diagnosis)

Examples of Good Captions:

1. "Apple leaf exhibiting Alternaria blotch. Small, circular brown lesions with yellowish halos are visible on the blade, some starting to coalesce near the margins. Spots appear slightly sunken and surrounded by chlorosis, indicating early to mid infection. In warm, humid conditions, lesions can expand, trigger premature defoliation, and reduce tree vigor and fruit quality."

2. "Tomato leaf affected by Spider Mites. Fine, pale stippling spreads across the surface, especially along veins, giving a yellow-bronze, roughened appearance. Leaf edges curl and pucker, and tissue feels dry, with tiny specks and faint webbing likely on the underside. Feeding reduces chlorophyll and vigor, leading to scorch and premature leaf drop in hot, dry weather."

Output format:
{
  "rating": <1-10>,
  "reasoning": "Brief explanation for the rating",
  "suggestions": "Specific suggestions for improvement"
}
```

### Optimization System Prompt

```
You are an expert agricultural diagnostician. Please optimize the following image caption to make it more accurate, detailed, and professional.

Optimization Guidelines:
1. Ensure clear identification of plant and disease/pest (or confirm health status)
2. Include detailed symptom description (location, shape, color, extent, quantity, etc.)
3. Assess severity and development stage
4. Keep language concise and professional (80-120 words)
5. Follow the style and quality of the examples below

Examples of Excellent Captions:

1. "Apple leaf exhibiting Alternaria blotch. Small, circular brown lesions with yellowish halos are visible on the blade, some starting to coalesce near the margins. Spots appear slightly sunken and surrounded by chlorosis, indicating early to mid infection. In warm, humid conditions, lesions can expand, trigger premature defoliation, and reduce tree vigor and fruit quality."

2. "Tomato leaf affected by Spider Mites. Fine, pale stippling spreads across the surface, especially along veins, giving a yellow-bronze, roughened appearance. Leaf edges curl and pucker, and tissue feels dry, with tiny specks and faint webbing likely on the underside. Feeding reduces chlorophyll and vigor, leading to scorch and premature leaf drop in hot, dry weather."

Please provide an optimized version of the caption.
```

### Enhancement Process

1. **Evaluation**: Score captions using the evaluation system (1-10 scale)
2. **Thresholding**: Captions scoring below threshold (default: 8) are flagged
3. **Optimization**: Low-scoring captions are rewritten using the optimization prompt
4. **Re-evaluation**: Optimized captions can be re-evaluated to ensure quality improvement

---

## Stage 2: GRPO Training

### System Prompt for GRPO

```
You are a plant disease management expert. Carefully analyze the given image and question, following these guidelines:

Core Requirements:
1. Output must be in English and structured into explicit steps labeled "Step 1: … Step 2: …"
2. For PREVENTION/CONTROL questions: Focus ONLY on method reasoning - DO NOT re-diagnose
3. For IDENTIFICATION questions: Focus on visual evidence and diagnostic reasoning
4. Keep reasoning concise and practical

Question Type Guidelines:

FOR PREVENTION/CONTROL METHODS QUESTIONS:
- Step 1: Analyze disease characteristics that influence control strategies
- Step 2: Recommend cultural/preventive practices based on disease biology
- Step 3: Outline chemical control timing and selection
- Step 4: Integrate application methods and safety precautions

FOR DISEASE IDENTIFICATION QUESTIONS:
- Step 1: Plant identification based on morphological features
- Step 2: Symptom observation and description
- Step 3: Disease pattern analysis
- Step 4: Preliminary diagnosis with confidence level

CRITICAL RULES:
- If question asks about CONTROL/MANAGEMENT/PREVENTION/TREATMENT/METHODS: Use PREVENTION/CONTROL guideline
- If question asks about IDENTIFICATION/WHAT/NAME/DISEASE: Use IDENTIFICATION guideline
- NEVER mix guidelines - choose one based on question type

Output Format:
<think>Step 1: ... Step 2: ... Step 3: ... Step 4: ...</think>
<answer>Your final answer here</answer>
```

### Domain Vocabularies

The system uses comprehensive domain-specific vocabularies to recognize multiple ways of referring to the same plant or disease.

#### Plant Variations

The system recognizes multiple ways to refer to the same plant:

```python
PLANT_VARIATIONS = {
    "tomato": ["tomato", "tomato plant", "tomatoes", "solanum lycopersicum",
               "lycopersicon esculentum", "nightshade", "tomato leaf", "tomato crop"],
    "potato": ["potato", "potato plant", "potatoes", "solanum tuberosum",
               "white potato", "irish potato", "potato tuber"],
    "corn": ["corn", "corn plant", "maize", "zea mays",
             "corn crop", "sweet corn", "field corn", "corn leaf"],
    "apple": ["apple", "apple tree", "malus domestica",
              "apple crop", "apple leaf", "apple plant"],
    "grape": ["grape", "grapevine", "vitis vinifera", "grape plant",
              "grape leaf", "vineyard"],
    # ... more plants
}
```

**Total Plants Covered:** 15+ major agricultural crops

#### Disease Variations

The system recognizes multiple names for the same disease:

```python
DISEASE_VARIATIONS = {
    "late blight": ["late blight", "phytophthora infestans", "phytophthora",
                    "oomycete disease", "late blight disease"],
    "early blight": ["early blight", "alternaria solani", "alternaria",
                     "target spot", "early blight disease", "alternaria leaf spot"],
    "powdery mildew": ["powdery mildew", "erysiphales", "white powdery coating",
                       "mildew", "powdery mildew fungus"],
    "healthy": ["healthy", "no disease", "disease-free", "normal plant",
                "no symptoms", "healthy plant"],
    # ... more diseases
}
```

**Total Diseases Covered:** 20 common plant diseases and healthy status

#### Treatment Keywords

For treatment/prevention questions, the system recognizes four categories:

```python
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
```

### Reward Function Design

The GRPO training uses a **data-driven reward structure** optimized for high-score convergence:

**Total Reward Range: [0, 3.0]**

#### 1. Format Reward [0, 0.5] (17%)

Evaluates structure integrity and content quality:

| Component | Score | Criteria |
|-----------|-------|----------|
| Basic Structure | 0.15 | Must have `<think> ... </think>` and `<answer> ... </answer>` tags |
| Step Structure | 0.15 | 4+ steps: 0.15, 3 steps: 0.12, 2 steps: 0.08, 1 step: 0.03 |
| Step Content Quality | 0.10 | Each step ≥30 chars: 4 steps: 0.10, 3 steps: 0.08, 2 steps: 0.05 |
| Think Length | 0.05 | 150-800 chars: 0.05, 100-1000 chars: 0.03, ≥80 chars: 0.01 |
| Answer Quality | 0.05 | 15-200 chars: 0.05, 10-300 chars: 0.03, ≥5 chars: 0.01 |

**Key Points:**
- Missing basic structure results in 0 score
- Encourages 4-step reasoning
- Dataset average: Think=487 chars, Answer=69 chars

#### 2. Answer Keyword Reward [0, 2.0] (67%)

Main scoring component with **dynamic evaluation** based on question type:

##### For Diagnosis Questions (99.6% of dataset)

| Component | Score | Description |
|-----------|-------|-------------|
| Plant Match | 0.8 | 5-tier fuzzy matching: Exact (1.0) → Weak (0.25) |
| Disease Match | 1.2 | 5-tier fuzzy matching with synonym recognition |

**Fuzzy Matching Tiers:**
- **Tier 1 (1.0):** Exact match (synonym from vocabulary)
- **Tier 2 (0.85):** High quality (multi-word term missing 1 word)
- **Tier 3 (0.7):** Partial (keyword stem matching)
- **Tier 4 (0.5):** Keyword (core words present)
- **Tier 5 (0.25):** Weak relevance (related terms)

**Example:**
```
Reference: "This is a tomato leaf affected by Early Blight (Alternaria solani)."
Generated: "The image shows tomato plant with alternaria leaf spot disease."

Plant Score: 0.8 × 1.0 = 0.8 (exact match: "tomato")
Disease Score: 1.2 × 0.85 = 1.02 (high-quality: "alternaria" matches "early blight")
Total: 1.82 / 2.0
```

##### For Treatment Questions (0.4% of dataset)

| Component | Score | Criteria |
|-----------|-------|----------|
| Pesticides | 0.6 | 3+ keywords: 0.6, 2 keywords: 0.45, 1 keyword: 0.3 |
| Cultural Practices | 0.5 | 3+ keywords: 0.5, 2 keywords: 0.35, 1 keyword: 0.2 |
| Application Methods | 0.5 | 3+ keywords: 0.5, 2 keywords: 0.35, 1 keyword: 0.2 |
| Application Timing | 0.4 | 3+ keywords: 0.4, 2 keywords: 0.3, 1 keyword: 0.15 |

#### 3. Reasoning Reward [0, 0.5] (17%)

Evaluates reasoning quality and logical coherence:

| Component | Score | Description |
|-----------|-------|-------------|
| Logical Coherence | 0.25 | Causal relationships, step connections |
| Professionalism | 0.15 | Professional terminology in context |
| Completeness | 0.10 | Observation → Analysis → Conclusion chain |

**Logical Coherence Patterns:**
- Causal relationships: "observe... because", "symptom... indicate"
- Step connections: Related keywords between consecutive steps

**Professional Context Patterns:**
- Pathogen usage: "pathogen/fungus/bacteria/virus + cause/infect/spread"
- Symptom description: "symptom + include/show/indicate/appear"
- Diagnosis reasoning: "diagnosis/identify + based on"
- Disease characteristics: "lesion/spot/blight + circular/brown/yellow"

**Completeness Chain:**
- **Observation** (0.40): observe, see, visible, symptoms, lesions, spots, affected, leaf
- **Analysis** (0.35): analyze, assess, indicate, diagnosis, infection, disease, caused
- **Conclusion** (0.35): therefore, conclude, diagnosis, control, prevent, apply

### Reward Function Summary

```
Total Reward = Format Reward + Answer Keyword Reward + Reasoning Reward
             = [0, 0.5]      + [0, 2.0]              + [0, 0.5]
             = [0, 3.0]

Weight Distribution:
- Answer Keyword: 67% (main driver of performance)
- Format Quality: 17% (structure integrity)
- Reasoning Logic: 17% (logical coherence)
```

---

## Evaluation Metrics

### Training Metrics

**GRPO Training Logs:**
- **Reward/Total**: Total reward score [0, 3.0]
- **Reward/Format**: Format quality [0, 0.5]
- **Reward/Answer_Keyword**: Answer matching [0, 2.0]
- **Reward/Reasoning**: Logic quality [0, 0.5]
- **Loss/Policy**: Policy gradient loss
- **KL Divergence**: Distance from reference model

### Inference Evaluation

**CDDMBench Metrics:**
- **Accuracy**: Percentage of correct predictions
- **Crop Classification Accuracy**: 15 crop types
- **Disease Classification Accuracy**: 20 disease categories
- **Knowledge QA Score**: 1-10 expert evaluation

**AgMMU Metrics:**
- **Overall Accuracy**: Out-of-distribution performance
- **Multiple Choice Accuracy**: Structured evaluation format

---

## Best Practices

### For Reasoning Generation

1. **Use appropriate temperature**: 0.3 for balanced creativity and consistency
2. **Validate JSON format**: Always parse and validate `{"think": "..."}` structure
3. **Ensure step count**: Aim for 3-4 clear, substantive steps
4. **Question type matters**: Detection accuracy improves Reasoning relevance
5. **API retry logic**: Use exponential backoff for reliability

### For Reasoning Enhancement

1. **Set appropriate threshold**: Default threshold=8 captures ~30-40% for optimization
2. **Preserve originals**: Always keep `original_caption` field
3. **Batch processing**: Process in batches with checkpoints
4. **Quality monitoring**: Track average rating improvements

### For GRPO Training

1. **Synonym coverage**: Ensure libraries cover your dataset's vocabulary
2. **Reward balance**: Monitor individual reward components
3. **KL divergence**: Keep below 0.1 to avoid distribution drift
4. **Dynamic batching**: Use warm-up steps to avoid early OOM
5. **Checkpoint frequently**: Save every 500-1000 steps

---

## Troubleshooting

### Reasoning Generation Issues

**Problem:** Generated Reasonings lack step structure

**Solution:**
- Check system prompt is correctly formatted
- Verify few-shot examples are appropriate for question type
- Increase temperature slightly (0.3 → 0.4) for more diverse outputs

**Problem:** API timeout errors

**Solution:**
- Reduce `max_tokens` to 512-768
- Implement retry logic with exponential backoff
- Use async batch processing for large datasets

### GRPO Training Issues

**Problem:** Low reward scores plateauing

**Solution:**
- Check synonym libraries cover dataset terms
- Verify fuzzy matching thresholds are appropriate
- Review sample outputs to identify systematic errors

**Problem:** High KL divergence

**Solution:**
- Reduce learning rate (5e-6 → 1e-6)
- Increase KL penalty coefficient
- Use more generations per prompt (4 → 8)

---

## References

**Key Files:**
- `src/stage1_reasoning/generate_reasoning.py` - Reasoning generation implementation
- `src/stage1_reasoning/enhance_reasoning.py` - Reasoning enhancement implementation
- `src/r1-v/src/open_r1/grpo_vqa.py` - GRPO reward functions
- `src/r1-v/src/open_r1/trainer/grpo_trainer.py` - Core GRPO trainer

**Related Documentation:**
- [DATA_FORMAT.md](DATA_FORMAT.md) - Dataset format specifications
- [README.md](README.md) - Project overview and quick start

---

**Last Updated:** 2025-01-01
