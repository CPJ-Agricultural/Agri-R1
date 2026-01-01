<div align="center">

# Agri-R1: Automated Chain-of-Thought for Agricultural Disease Diagnosis

**Reinforcement Learning-Enhanced Vision-Language Model for Agricultural Disease Diagnosis**

[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](#)
[![ACL 2025](https://img.shields.io/badge/ACL-2025-orange.svg)](#)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=agri-r1)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/Agri-R1.svg?style=social)](#)

<img src="Images/pipeline.png" width="100%" alt="Agri-R1 Pipeline">

</div>

---

## Overview

**Agri-R1** is a cutting-edge vision-language model built on **Qwen2.5-VL-3B-Instruct** that leverages **Group Relative Policy Optimization (GRPO)** with automated **Chain-of-Thought (COT)** reasoning for enhanced agricultural disease diagnosis.

### Key Highlights

- **Automated COT Generation** - Self-distilled reasoning from powerful VLMs (GPT-4o, Claude 3.5) without manual annotation
- **Reinforcement Learning** - GRPO training for robust and generalizable diagnostic reasoning
- **Two-Stage Pipeline** - Stage 1: COT data generation → Stage 2: GRPO optimization
- **Strong Generalization** - Validated on out-of-distribution benchmarks (AgMMU, MIRAGE)
- **Interpretable Reasoning** - Step-by-step diagnostic process with `<think>` and `<answer>` tags

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Agri-R1.git
cd Agri-R1

# Create conda environment
conda create -n agri-r1 python=3.11
conda activate agri-r1

# Install dependencies
pip install -r requirements.txt

# Install the r1-v training framework
cd src/r1-v
pip install -e .
cd ../..
```

### System Requirements

- **GPU**: 4×A800 80GB (training) / 1×A100 40GB (inference)
- **CUDA**: 11.8+ with Flash Attention 2 support
- **Python**: 3.11+

---

## Repository Structure

```
Agri-R1/
├── Images/                           # Pipeline visualizations
│   └── pipeline.png                  # Main architecture diagram
│
├── datasets/                         # Training and evaluation data
│   ├── train and evaluation datasets/
│   │   ├── train_data_examples/      # Training data samples
│   │   └── evaluation_data/          # Benchmark datasets
│   └── evaluation datasets results/  # Model inference outputs
│
├── src/
│   ├── scripts/                      # Training launch scripts
│   │   ├── train_grpo_with_cot.sh    # GRPO + COT training (recommended)
│   │   ├── train_grpo_no_cot.sh      # GRPO without COT
│   │   └── train_sft.sh              # Supervised fine-tuning baseline
│   │
│   ├── r1-v/                         # GRPO training framework
│   │   ├── src/open_r1/
│   │   │   ├── grpo_vqa.py           # GRPO trainer with COT
│   │   │   ├── grpo_no_cot.py        # GRPO trainer without COT
│   │   │   ├── sft.py                # SFT trainer
│   │   │   └── trainer/
│   │   │       ├── grpo_trainer.py   # Core GRPO implementation
│   │   │       └── dynamic_callbacks.py
│   │   └── configs/                  # DeepSpeed ZeRO-3 configurations
│   │
│   ├── stage1_cot/                   # COT data generation pipeline
│   │   ├── resize_images_384.py      # Image preprocessing (384×384)
│   │   ├── sample_dataset_20k.py     # Dataset sampling strategy
│   │   ├── generate_cot.py           # COT generation via API
│   │   ├── enhance_cot.py            # COT quality enhancement
│   │   └── README.md                 # Stage 1 detailed documentation
│   │
│   └── eval_vqa/                     # Comprehensive evaluation suite
│       ├── cddmbench/                # In-distribution evaluation
│       │   ├── crop_disease/         # Disease classification tasks
│       │   │   ├── inference_zeroshot.py
│       │   │   ├── inference_fiveshot.py
│       │   │   ├── inference_grpo_cot.py
│       │   │   ├── evaluate.py
│       │   │   └── run_inference.sh
│       │   └── knowledge/            # Knowledge QA tasks
│       │       ├── inference_zeroshot.py
│       │       ├── inference_grpo_cot.py
│       │       └── run_inference.sh
│       └── agmmu/                    # Out-of-distribution evaluation
│           ├── inference_base.py
│           ├── evaluate.py
│           └── run_inference.sh
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Training Pipeline

### Stage 1: Automated COT Data Generation

Generate high-quality chain-of-thought reasoning data using powerful VLMs:

```bash
cd src/stage1_cot

# 1. Resize images for efficient processing
python resize_images_384.py \
  --input_dir /path/to/images \
  --output_dir ./images_384

# 2. Sample training data
python sample_dataset_20k.py \
  --input_data /path/to/full_dataset.json \
  --output_data ./sampled_20k.json \
  --num_samples 20000

# 3. Generate COT annotations
python generate_cot.py \
  --input_data ./sampled_20k.json \
  --output_dir ./cot_data \
  --api_key YOUR_API_KEY \
  --model gpt-4o

# 4. Enhance COT quality
python enhance_cot.py \
  --input_dir ./cot_data \
  --output_dir ./cot_enhanced
```

#### Data Format Example

```json
{
  "image": "images/tomato_leaf_001.jpg",
  "question": "What disease is affecting this tomato plant?",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nWhat disease is affecting this tomato plant?"
    },
    {
      "from": "assistant",
      "value": "<think>Step 1: Observe leaf characteristics - yellowing between veins, dark circular spots.\nStep 2: Analyze lesion patterns - concentric rings typical of fungal infection.\nStep 3: Identify pathogen - Alternaria solani based on symptom morphology.\nStep 4: Confirm diagnosis.</think><answer>Tomato Early Blight (Alternaria solani)</answer>"
    }
  ]
}
```

### Stage 2: GRPO Reinforcement Learning

Train the model using Group Relative Policy Optimization:

```bash
# Recommended: GRPO with COT
bash src/scripts/train_grpo_with_cot.sh

# Alternative: Supervised Fine-Tuning (baseline)
bash src/scripts/train_sft.sh

# Alternative: GRPO without COT (ablation)
bash src/scripts/train_grpo_no_cot.sh
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-3B-Instruct |
| Training Data | 200k samples with COT |
| Hardware | 4×A800 80GB GPUs |
| Batch Size | 160 (10/device × 4 accum × 4 GPUs) |
| Epochs | 3 |
| Optimizer | AdamW (lr: 5e-6) |
| Training Time | ~69 hours |
| Optimization | DeepSpeed ZeRO-3 |

---

## Inference & Evaluation

### CDDMBench Evaluation (In-Distribution)

Evaluate on crop and disease classification tasks:

```bash
cd src/eval_vqa/cddmbench/crop_disease

# Run inference with COT reasoning
python inference_grpo_cot.py \
  --model_path /path/to/agri-r1-checkpoint \
  --input_json test_data.json \
  --output_json predictions.json

# Evaluate results
python evaluate.py \
  --ground-truth-file test_data.json \
  --model-answers-file predictions.json \
  --output-file results.json
```

### AgMMU Evaluation (Out-of-Distribution)

Test generalization on unseen agricultural data:

```bash
cd src/eval_vqa/agmmu

# Inference with COT
python inference_with_cot.py \
  --model_path /path/to/agri-r1-checkpoint \
  --data_path agmmu_validation.json \
  --image_dir ./images \
  --output predictions.json

# Evaluate
python evaluate.py \
  --predictions predictions.json \
  --ground_truth agmmu_validation.json
```

### Python API Example

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/agri-r1-checkpoint",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("path/to/agri-r1-checkpoint")

# Prepare input
image = Image.open("tomato_disease.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What disease is this? Provide detailed reasoning."}
        ]
    }
]

# Generate prediction
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False
)

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)

# Example output:
# <think>
# Step 1: Observe leaf characteristics - yellowing between veins with dark spots
# Step 2: Analyze spot patterns - circular lesions with concentric rings
# Step 3: Consider disease etiology - fungal pathogen Alternaria solani
# Step 4: Confirm diagnosis based on symptom morphology
# </think>
# <answer>Tomato Early Blight (Alternaria solani)</answer>
```

---

## Evaluation Benchmarks

### CDDMBench
- **Crop Classification**: 40 crop species identification
- **Disease Classification**: 120+ disease types across major crops
- **Knowledge QA**: Expert-level questions on disease control, symptoms, and pathogens

### AgMMU (Out-of-Distribution)
- **Global Coverage**: Agricultural diseases from diverse geographical regions
- **Multiple Choice**: Structured evaluation format
- **Generalization Test**: Measures model robustness on unseen data

### MIRAGE
- **Multimodal Reasoning**: Complex agricultural scenarios
- **Multi-step Inference**: Tests advanced reasoning capabilities

---

## Implementation Details

### GRPO Training Framework

The core GRPO implementation is in `src/r1-v/src/open_r1/trainer/grpo_trainer.py`:

```python
# Key components:
# 1. Policy gradient calculation with KL divergence penalty
# 2. Group-wise advantage normalization
# 3. Dynamic reward scaling based on reference model
# 4. Efficient memory management with DeepSpeed ZeRO-3
```

**Configuration files:**
- `src/r1-v/configs/zero3.json` - DeepSpeed ZeRO-3 optimization
- `src/scripts/train_grpo_with_cot.sh` - Training hyperparameters

### COT Generation Pipeline

Automated reasoning generation in `src/stage1_cot/generate_cot.py`:

```python
# Pipeline:
# 1. Load image + question
# 2. Query GPT-4o/Claude-3.5 with specialized prompt
# 3. Parse <think> and <answer> tags
# 4. Quality filtering and enhancement
# 5. Format for GRPO training
```

**Key Scripts:**
- `generate_cot.py` - Main COT generation
- `enhance_cot.py` - Quality improvement and filtering
- `resize_images_384.py` - Efficient image preprocessing

---

## Example Results

Model outputs with COT reasoning can be found in `datasets/evaluation datasets results/`:

**Disease Classification Example:**
```
Input: Image of tomato leaf with disease symptoms
Question: "Identify the disease affecting this tomato plant."

Output:
<think>
Step 1: Observe leaf condition - yellowing areas with dark lesions
Step 2: Examine lesion morphology - circular spots with target-like rings
Step 3: Assess distribution pattern - lesions starting from older leaves
Step 4: Identify pathogen characteristics - Alternaria solani fungal infection
</think>
<answer>Tomato Early Blight (Alternaria solani)</answer>
```

---

## Acknowledgements

This project builds upon excellent open-source work:

- **[R1-V](https://github.com/Deep-Agent/R1-V)** - GRPO training framework for vision-language models
- **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL)** - Powerful base vision-language model
- **[CDDMBench](https://github.com/CDDMBench/CDDMBench)** - Comprehensive agricultural VQA benchmark
- **[AgMMU](https://github.com/AgMMU/AgMMU)** - Out-of-distribution generalization benchmark

---

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{agri-r1-2025,
  title={Agri-R1: Automated Chain-of-Thought for Agricultural Disease Diagnosis},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025}
}
```

---

## Contact

For questions, collaboration, or issues:

- **Email**: your.email@university.edu
- **Issues**: [GitHub Issues](https://github.com/your-username/Agri-R1/issues)

---

## License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) for details.

> **Note**: Code and models will be released upon ACL 2025 acceptance.

---

<div align="center">
Made with ❤️ for the agricultural AI community
</div>
