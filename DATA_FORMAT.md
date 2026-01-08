# Data Format Specification

This document details all data formats used throughout the Agri-R1 project, from raw training data to evaluation outputs.

---

## Table of Contents

1. [Training Data Formats](#training-data-formats)
2. [Reasoning Generation Data](#reasoning-generation-data)
3. [GRPO Training Data](#grpo-training-data)
4. [Evaluation Data Formats](#evaluation-data-formats)
5. [Model Output Formats](#model-output-formats)

---

## Training Data Formats

### Raw Training Data (Before Reasoning)

**File Format:** JSON (`.json`)

**Structure:**
```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat disease is affecting this tomato plant?"
      },
      {
        "from": "gpt",
        "value": "This image shows a tomato leaf affected by Early Blight (Alternaria solani). The disease is characterized by circular brown spots with concentric rings, typically starting on older leaves."
      }
    ]
  }
]
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Relative or absolute path to image file |
| `conversations` | array | Yes | List of conversation turns |
| `conversations[].from` | string | Yes | Speaker: `"human"` or `"gpt"` |
| `conversations[].value` | string | Yes | Message content. Human messages may include `<image>` token |

**Example Dataset Statistics:**
- Total samples: 200,005
- Images: 384×384 pixels (preprocessed)
- Average conversations per image: 1-3 turns
- Dataset size: ~50GB (with images)

---

## Reasoning Generation Data

### Stage 1 Output: Training Data with Reasoning

**File Format:** JSON (`.json`)

**Structure:**
```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat disease is affecting this tomato plant?"
      },
      {
        "from": "gpt",
        "value": "This image shows a tomato leaf affected by Early Blight (Alternaria solani). The disease is characterized by circular brown spots with concentric rings, typically starting on older leaves.",
        "think": "Step 1: Identify plant - leaf shape and venation pattern consistent with tomato (Solanum lycopersicum). Step 2: Observe symptoms - circular brown lesions with characteristic target-like concentric rings on leaf surface. Step 3: Assess lesion distribution - spots primarily on older, lower leaves, indicating typical early blight progression. Step 4: Diagnose disease - Alternaria solani based on distinctive bull's-eye lesion morphology and distribution pattern; confidence: high."
      }
    ]
  }
]
```

**Added Field:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversations[].think` | string | Yes (for Reasoning) | Chain-of-thought reasoning with 3-4 explicit steps |

**Reasoning Format Requirements:**
- Must contain 3-4 steps labeled as "Step 1:", "Step 2:", etc.
- Each step should be substantive (≥30 characters)
- Total think length: 150-800 characters (optimal)
- Language: English only

**Quality Metrics:**
- Average think length: 487 characters
- Average step count: 3.8 steps
- Average step length: 128 characters

---

## GRPO Training Data

### GRPO Dataset Format

**File Format:** HuggingFace Datasets (`.arrow` files via `datasets` library)

**Required Fields:**

```python
from datasets import load_from_disk

dataset = load_from_disk("path/to/grpo_dataset")

# Required columns:
# - image: str (path to image)
# - problem: str (question/prompt)
# - solution: str (reference answer with Reasoning)
```

**Structure:**
```json
{
  "image": "/path/to/tomato_early_blight_001.jpg",
  "problem": "What disease is affecting this tomato plant?",
  "solution": "<think>Step 1: Identify plant - leaf shape and venation pattern consistent with tomato (Solanum lycopersicum). Step 2: Observe symptoms - circular brown lesions with characteristic target-like concentric rings on leaf surface. Step 3: Assess lesion distribution - spots primarily on older, lower leaves, indicating typical early blight progression. Step 4: Diagnose disease - Alternaria solani based on distinctive bull's-eye lesion morphology and distribution pattern; confidence: high.</think><answer>Tomato Early Blight (Alternaria solani)</answer>"
}
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Absolute path to 384×384 image file |
| `problem` | string | Yes | Question text (without `<image>` token) |
| `solution` | string | Yes | Complete reference answer with `<think> ... </think>` and `<answer> ... </answer>` tags |

**Solution Format Structure:**
```
<think>
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
Step 3: [Third reasoning step]
Step 4: [Fourth reasoning step]
</think>
<answer>[Final answer]</answer>
```

**Dataset Preparation Script:**
```python
from datasets import Dataset, DatasetDict

# Convert from conversations format
def convert_to_grpo_format(conversations_data):
    grpo_samples = []
    for item in conversations_data:
        for i in range(0, len(item['conversations']), 2):
            if i+1 >= len(item['conversations']):
                break

            human_msg = item['conversations'][i]
            gpt_msg = item['conversations'][i+1]

            if human_msg['from'] == 'human' and gpt_msg['from'] == 'gpt':
                # Extract question (remove <image> token)
                question = human_msg['value'].replace('<image>\n', '').replace('<image>', '').strip()

                # Build solution with think and answer
                think = gpt_msg.get('think', '')
                answer = gpt_msg['value']
                solution = f"<think>{think}</think><answer>{answer}</answer>"

                grpo_samples.append({
                    'image': item['image'],
                    'problem': question,
                    'solution': solution
                })

    return Dataset.from_list(grpo_samples)

# Create and save dataset
dataset = convert_to_grpo_format(conversations_data)
dataset.save_to_disk("./grpo_training_dataset")
```

**Prompt Construction (Done Automatically by PromptDatasetWrapper):**
```python
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a plant disease management expert..."
    },
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "Question: What disease is affecting this tomato plant?\n\nPlease analyze the image and provide your answer in the required format."}
      ]
    }
  ]
}
```

---

## Evaluation Data Formats

### CDDMBench Format

#### Disease Classification

**Input File:** `test_data.json`

```json
[
  {
    "id": "cddm_disease_001",
    "image": "images/tomato_early_blight_test_01.jpg",
    "question": "What disease is shown in this image?",
    "options": {
      "A": "Early Blight",
      "B": "Late Blight",
      "C": "Septoria Leaf Spot",
      "D": "Bacterial Spot"
    },
    "answer": "A",
    "disease_name": "Tomato Early Blight",
    "category": "disease_classification"
  }
]
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique sample identifier |
| `image` | string | Yes | Path to test image |
| `question` | string | Yes | Question text |
| `options` | object | Yes | Multiple choice options (A, B, C, D) |
| `answer` | string | Yes | Correct option letter |
| `disease_name` | string | No | Full disease name for reference |
| `category` | string | No | Task category |

#### Knowledge QA

**Input File:** `knowledge_qa.json`

```json
[
  {
    "id": "cddm_knowledge_001",
    "image": "images/tomato_early_blight_symptoms.jpg",
    "question": "What are effective prevention methods for the disease shown in the image?",
    "reference_answer": "(1) Crop rotation with non-host plants (2) Remove and destroy infected leaves (3) Apply fungicides preventively starting in early season (4) Ensure adequate plant spacing for air circulation (5) Avoid overhead irrigation",
    "category": "knowledge_qa",
    "max_score": 10
  }
]
```

**Scoring:** Expert manual evaluation (1-10 scale)

### AgMMU Format

**Input File:** `agmmu_validation.json`

```json
[
  {
    "id": "agmmu_001",
    "images": ["images/wheat_stripe_rust_001.jpg"],
    "question": "What is the most likely disease affecting this wheat plant? A) Wheat Stripe Rust B) Powdery Mildew C) Fusarium Head Blight D) Septoria Leaf Spot",
    "answer": "A) Wheat Stripe Rust",
    "options": [
      "A) Wheat Stripe Rust",
      "B) Powdery Mildew",
      "C) Fusarium Head Blight",
      "D) Septoria Leaf Spot"
    ],
    "qtype": "disease_identification",
    "topic": "plant_disease_diagnosis"
  }
]
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique sample identifier |
| `images` | array[string] | Yes | List of image paths (usually 1 image) |
| `question` | string | Yes | Question with inline options |
| `answer` | string | Yes | Correct answer with option letter |
| `options` | array[string] | Yes | List of all options |
| `qtype` | string | No | Question type category |
| `topic` | string | No | Subject area |

---

## Model Output Formats

### Inference Output (with Reasoning)

**File Format:** JSON (`.json`)

**Structure:**
```json
[
  {
    "id": "sample_001",
    "question": "What disease is affecting this tomato plant?",
    "image": "images/tomato_early_blight_001.jpg",
    "model_output": "<think>Step 1: Identify plant - leaf morphology consistent with tomato (Solanum lycopersicum), showing characteristic pinnately compound structure. Step 2: Observe symptoms - multiple circular brown lesions with concentric rings (bull's-eye pattern) scattered across leaf surface. Step 3: Analyze distribution - lesions primarily on lower, older leaves, typical of foliar fungal diseases. Step 4: Diagnose - Alternaria solani based on distinctive target spot morphology and distribution pattern; confidence: high.</think><answer>Tomato Early Blight (Alternaria solani)</answer>",
    "extracted_think": "Step 1: Identify plant - leaf morphology consistent with tomato (Solanum lycopersicum), showing characteristic pinnately compound structure. Step 2: Observe symptoms - multiple circular brown lesions with concentric rings (bull's-eye pattern) scattered across leaf surface. Step 3: Analyze distribution - lesions primarily on lower, older leaves, typical of foliar fungal diseases. Step 4: Diagnose - Alternaria solani based on distinctive target spot morphology and distribution pattern; confidence: high.",
    "extracted_answer": "Tomato Early Blight (Alternaria solani)",
    "ground_truth": "A",
    "predicted_option": "A",
    "correct": true,
    "inference_time": 2.34
  }
]
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Sample identifier from input |
| `question` | string | Yes | Original question |
| `image` | string | Yes | Image path |
| `model_output` | string | Yes | Raw model output with tags |
| `extracted_think` | string | Yes | Extracted reasoning from `<think> ... </think>` tags |
| `extracted_answer` | string | Yes | Extracted answer from `<answer> ... </answer>` tags |
| `ground_truth` | string | Yes | Correct answer from dataset |
| `predicted_option` | string | No | Predicted option letter (for multiple choice) |
| `correct` | boolean | No | Whether prediction matches ground truth |
| `inference_time` | float | No | Time in seconds for inference |

### Evaluation Results

**File Format:** JSON (`.json`)

**Structure:**
```json
{
  "dataset": "CDDMBench_Disease_Classification",
  "model": "Agri-R1-GRPO-Reasoning",
  "total_samples": 1000,
  "correct_predictions": 843,
  "accuracy": 0.843,
  "metrics": {
    "precision": 0.851,
    "recall": 0.843,
    "f1_score": 0.847
  },
  "per_category": {
    "early_blight": {"accuracy": 0.92, "samples": 120},
    "late_blight": {"accuracy": 0.88, "samples": 115},
    "powdery_mildew": {"accuracy": 0.79, "samples": 95}
  },
  "errors": [
    {
      "id": "sample_042",
      "ground_truth": "Late Blight",
      "prediction": "Early Blight",
      "confidence": 0.73
    }
  ],
  "timestamp": "2025-01-01T12:00:00Z",
  "config": {
    "model_path": "/path/to/checkpoint",
    "temperature": 0.0,
    "max_new_tokens": 512
  }
}
```

---

## Data Conversion Examples

### Converting Conversations to GRPO Format

```python
import json
from datasets import Dataset

def conversations_to_grpo(input_file, output_dir):
    """
    Convert conversations format to GRPO format

    Args:
        input_file: Path to conversations JSON file
        output_dir: Directory to save GRPO dataset
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        image_path = item['image']

        for i in range(0, len(item['conversations']), 2):
            if i+1 >= len(item['conversations']):
                continue

            human = item['conversations'][i]
            assistant = item['conversations'][i+1]

            if human['from'] != 'human' or assistant['from'] != 'gpt':
                continue

            # Extract question
            question = human['value'].replace('<image>\n', '').replace('<image>', '').strip()

            # Build solution
            think = assistant.get('think', '')
            answer = assistant['value']

            if think:
                solution = f"<think>{think}</think><answer>{answer}</answer>"
            else:
                solution = f"<answer>{answer}</answer>"

            samples.append({
                'image': image_path,
                'problem': question,
                'solution': solution
            })

    # Create and save dataset
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(output_dir)
    print(f"Saved {len(samples)} samples to {output_dir}")

# Usage
conversations_to_grpo(
    input_file="data/training_with_reasoning.json",
    output_dir="data/grpo_dataset"
)
```

### Extracting Predictions from Model Output

```python
import re
import json

def extract_predictions(model_output_file, output_file):
    """
    Extract think and answer from model outputs

    Args:
        model_output_file: Raw model outputs
        output_file: Structured predictions
    """
    with open(model_output_file, 'r') as f:
        outputs = json.load(f)

    predictions = []
    for item in outputs:
        # Extract think
        think_match = re.search(r'<think>(.*?)</think>', item['model_output'], re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""

        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', item['model_output'], re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        predictions.append({
            'id': item['id'],
            'question': item['question'],
            'extracted_think': think,
            'extracted_answer': answer,
            'ground_truth': item['ground_truth']
        })

    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(predictions)} predictions to {output_file}")

# Usage
extract_predictions(
    model_output_file="results/raw_outputs.json",
    output_file="results/predictions.json"
)
```

---

## Data Validation

### Validating Training Data

```python
import json
import os
from pathlib import Path

def validate_training_data(data_file, image_dir):
    """
    Validate training data format and integrity

    Args:
        data_file: Path to training JSON file
        image_dir: Base directory for images

    Returns:
        dict: Validation report
    """
    with open(data_file, 'r') as f:
        data = json.load(f)

    report = {
        'total_samples': len(data),
        'valid_samples': 0,
        'errors': []
    }

    for idx, item in enumerate(data):
        # Check required fields
        if 'image' not in item:
            report['errors'].append(f"Sample {idx}: Missing 'image' field")
            continue

        if 'conversations' not in item:
            report['errors'].append(f"Sample {idx}: Missing 'conversations' field")
            continue

        # Check image exists
        image_path = item['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_dir, image_path)

        if not os.path.exists(image_path):
            report['errors'].append(f"Sample {idx}: Image not found: {item['image']}")
            continue

        # Check conversations structure
        for conv_idx, conv in enumerate(item['conversations']):
            if 'from' not in conv or 'value' not in conv:
                report['errors'].append(f"Sample {idx}, Conv {conv_idx}: Missing 'from' or 'value'")
                continue

            # Check for Reasoning in assistant messages
            if conv['from'] == 'gpt' and 'think' in conv:
                think = conv['think']
                step_count = len(re.findall(r'Step \d+:', think))
                if step_count < 2:
                    report['errors'].append(f"Sample {idx}, Conv {conv_idx}: Reasoning has only {step_count} steps")

        report['valid_samples'] += 1

    report['validity_rate'] = report['valid_samples'] / report['total_samples']
    return report

# Usage
report = validate_training_data(
    data_file="data/training_with_reasoning.json",
    image_dir="data/images"
)

print(f"Validation Report:")
print(f"Total Samples: {report['total_samples']}")
print(f"Valid Samples: {report['valid_samples']}")
print(f"Validity Rate: {report['validity_rate']:.2%}")
print(f"Errors: {len(report['errors'])}")
```

### Validating GRPO Dataset

```python
from datasets import load_from_disk
import re

def validate_grpo_dataset(dataset_path):
    """
    Validate GRPO dataset format

    Args:
        dataset_path: Path to HuggingFace dataset directory

    Returns:
        dict: Validation report
    """
    dataset = load_from_disk(dataset_path)

    report = {
        'total_samples': len(dataset),
        'valid_samples': 0,
        'errors': []
    }

    required_fields = ['image', 'problem', 'solution']

    for idx in range(len(dataset)):
        item = dataset[idx]

        # Check required fields
        missing_fields = [f for f in required_fields if f not in item]
        if missing_fields:
            report['errors'].append(f"Sample {idx}: Missing fields: {missing_fields}")
            continue

        # Check solution format
        solution = item['solution']
        if '<think>' not in solution or '</think>' not in solution:
            report['errors'].append(f"Sample {idx}: Solution missing <think> tags")
            continue

        if '<answer>' not in solution or '</answer>' not in solution:
            report['errors'].append(f"Sample {idx}: Solution missing <answer> tags")
            continue

        # Check think has steps
        think_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
        if think_match:
            think = think_match.group(1)
            step_count = len(re.findall(r'Step \d+:', think))
            if step_count < 2:
                report['errors'].append(f"Sample {idx}: Think has only {step_count} steps")

        report['valid_samples'] += 1

    report['validity_rate'] = report['valid_samples'] / report['total_samples']
    return report

# Usage
report = validate_grpo_dataset("data/grpo_dataset")

print(f"GRPO Dataset Validation:")
print(f"Total: {report['total_samples']}")
print(f"Valid: {report['valid_samples']}")
print(f"Rate: {report['validity_rate']:.2%}")
```

---

## Dataset Statistics

### Training Dataset Statistics (Example)

```
Total Samples: 200,005
Image Resolution: 384×384 pixels
Total Size: ~50GB

Conversations per Sample:
  - Min: 1
  - Max: 3
  - Average: 1.8

Question Types:
  - Disease Identification: 99.6% (199,205)
  - Prevention/Treatment: 0.4% (800)

Reasoning Statistics:
  - Average Think Length: 487 characters
  - Average Answer Length: 69 characters
  - Average Step Count: 3.8 steps
  - Average Step Length: 128 characters

Plant Distribution (Top 5):
  1. Tomato: 28%
  2. Corn: 18%
  3. Potato: 15%
  4. Apple: 12%
  5. Grape: 10%

Disease Distribution (Top 5):
  1. Healthy: 15%
  2. Early Blight: 8%
  3. Late Blight: 7%
  4. Powdery Mildew: 6%
  5. Bacterial Spot: 5%
```

---

## Best Practices

### Data Preparation

1. **Image Preprocessing**
   - Resize all images to 384×384 pixels
   - Maintain aspect ratio with padding if needed
   - Use consistent image format (JPEG or PNG)
   - Validate images can be opened before training

2. **Text Normalization**
   - Remove extra whitespace
   - Ensure consistent capitalization in answers
   - Validate special characters are properly encoded
   - Check for empty or null fields

3. **Reasoning Quality**
   - Ensure 3-4 clear steps per Reasoning
   - Validate step labels (Step 1:, Step 2:, etc.)
   - Check step content is substantive (≥30 chars)
   - Verify think and answer are aligned

### Data Storage

1. **File Organization**
   ```
   data/
   ├── images/                # All images
   │   ├── train/
   │   └── test/
   ├── raw/                   # Raw data before Reasoning
   ├── reasoning_generated/         # After Reasoning generation
   ├── reasoning_enhanced/          # After quality enhancement
   └── grpo_dataset/          # Final HF dataset
   ```

2. **Versioning**
   - Use semantic versioning (v1.0.0, v1.1.0)
   - Document changes in CHANGELOG.md
   - Keep separate directories for each version
   - Archive old versions for reproducibility

3. **Backup**
   - Regular backups of processed datasets
   - Store raw data separately
   - Use cloud storage for redundancy
   - Test restore procedures

---

## Troubleshooting

### Common Issues

**Issue:** Image paths not found during training

**Solution:**
```python
# Use absolute paths
import os
data['image'] = os.path.abspath(data['image'])

# Or update base path
data['image'] = data['image'].replace('/old/path/', '/new/path/')
```

**Issue:** Reasoning format inconsistencies

**Solution:**
```python
# Validate and fix Reasoning format
def fix_reasoning_format(solution):
    import re

    # Extract parts
    think_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)

    if not think_match or not answer_match:
        return None

    think = think_match.group(1).strip()
    answer = answer_match.group(1).strip()

    # Rebuild with proper format
    return f"<think>{think}</think><answer>{answer}</answer>"
```

**Issue:** Dataset too large for memory

**Solution:**
```python
# Use IterableDataset for streaming
from datasets import IterableDataset

dataset = IterableDataset.from_generator(
    generator_function,
    gen_kwargs={'data_path': 'data/large_dataset.json'}
)
```

---

## Related Documentation

- [PROMPTS_AND_EVALUATION.md](PROMPTS_AND_EVALUATION.md) - Prompts and reward functions
- [README.md](README.md) - Project overview and quick start
- [src/stage1_reasoning/README.md](src/stage1_reasoning/README.md) - Reasoning generation details

---

**Last Updated:** 2025-01-01
