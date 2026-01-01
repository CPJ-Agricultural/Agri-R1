<p align="center">
  <h2 align="center">Agri-R1: Automated Chain-of-Thought for Agricultural Disease Diagnosis</h2>
</p>

**Agri-R1** is a reinforcement learning-enhanced vision-language model (VLM) for agricultural disease diagnosis. Built on Qwen2.5-VL-3B-Instruct, Agri-R1 uses **Group Relative Policy Optimization (GRPO)** with automated **Chain-of-Thought (COT) reasoning** to improve diagnostic accuracy and interpretability.

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](#)
[![ACL 2025](https://img.shields.io/badge/ACL-2025-orange.svg)](#)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=agri-r1)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/Agri-R1.svg?style=social)](#)

</div>

> **ACL 2025 Submission** - Code and models to be released upon acceptance

---

## 🔍 Overview

Agri-R1 addresses the challenge of agricultural disease diagnosis through:
- **Automated COT Generation**: Self-distilled reasoning from powerful VLMs (GPT-4o, Claude 3.5)
- **Reinforcement Learning**: GRPO training for robust, generalizable reasoning
- **Two-Stage Pipeline**: Stage 1 (COT data generation) → Stage 2 (GRPO optimization)
- **Strong Generalization**: Evaluated on out-of-distribution benchmarks (AgMMU, MIRAGE)

![Agri-R1 Pipeline](Images/pipeline.pdf)

### Key Features

- ✅ **Automated COT**: No manual annotation required
- ✅ **Efficient Training**: 200k samples, 3 epochs on 4×A800 GPUs
- ✅ **Strong Performance**: Competitive with GPT-4o on CDDMBench
- ✅ **Generalizable**: Robust performance on unseen datasets
- ✅ **Interpretable**: Step-by-step reasoning with `<think>` and `<answer>` tags

---

## 📊 Performance

### CDDMBench (In-Distribution)

| Model | Crop Classification | Disease Classification | Knowledge QA |
|-------|---------------------|------------------------|--------------|
| GPT-4o | 89.2% | 87.5% | 8.2/10 |
| LLaVA-1.5-13B | 72.3% | 68.9% | 6.5/10 |
| **Agri-R1 (GRPO+COT)** | **86.7%** | **84.3%** | **7.8/10** |
| Agri-R1 (SFT) | 82.1% | 79.6% | 7.2/10 |

### AgMMU (Out-of-Distribution)

| Model | Overall Accuracy |
|-------|------------------|
| GPT-4o | 85.25% |
| Gemini 1.5 Pro | 80.42% |
| **Agri-R1 (GRPO+COT)** | **78.9%** |
| LLaVA-1.5-13B | 66.73% |

---

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Agri-R1.git
cd Agri-R1

# Create environment
conda create -n agri-r1 python=3.11
conda activate agri-r1

# Install dependencies
pip install -r requirements.txt

# Install the r1-v package
cd src/r1-v
pip install -e .
cd ../..
```

> [!NOTE]
> - Flash Attention 2 requires CUDA 11.8+ and compatible GPU (A100/A800 recommended)
> - For training, at least 4×A800 80GB GPUs are recommended
> - For inference, 1×A100 40GB GPU is sufficient

---

## 📂 Repository Structure

```
Agri-R1/
├── Images/                    # Pipeline figures and visualizations
├── datasets/                  # Training and evaluation datasets
│   ├── train and evaluation datasets/
│   │   ├── train_data_examples/       # Sample training data
│   │   └── evaluation_data/           # Evaluation benchmarks
│   └── evaluation datasets results/   # Inference results
├── src/
│   ├── scripts/              # Training launch scripts
│   │   ├── train_grpo_with_cot.sh   # GRPO training with COT
│   │   ├── train_grpo_no_cot.sh     # GRPO training without COT
│   │   └── train_sft.sh             # Supervised fine-tuning
│   ├── r1-v/                 # GRPO training framework
│   │   ├── src/open_r1/
│   │   │   ├── grpo_vqa.py           # GRPO trainer with COT
│   │   │   ├── grpo_no_cot.py        # GRPO trainer without COT
│   │   │   ├── sft.py                # SFT trainer
│   │   │   └── trainer/
│   │   │       ├── grpo_trainer.py   # Core GRPO implementation
│   │   │       └── dynamic_callbacks.py
│   │   └── configs/          # DeepSpeed ZeRO configurations
│   ├── stage1_cot/           # COT data generation pipeline
│   │   ├── resize_images_384.py      # Image preprocessing
│   │   ├── sample_dataset_20k.py     # Dataset sampling
│   │   ├── generate_cot.py           # COT generation
│   │   ├── enhance_cot.py            # COT enhancement
│   │   └── README.md                 # Stage 1 documentation
│   └── eval_vqa/             # Evaluation scripts
│       ├── cddmbench/
│       │   ├── crop_disease/         # Disease classification
│       │   │   ├── inference_zeroshot.py
│       │   │   ├── inference_fiveshot.py
│       │   │   ├── inference_grpo_cot.py
│       │   │   ├── evaluate.py
│       │   │   └── run_inference.sh
│       │   └── knowledge/            # Knowledge QA
│       │       ├── inference_zeroshot.py
│       │       ├── inference_grpo_cot.py
│       │       └── run_inference.sh
│       └── agmmu/            # Generalization evaluation
│           ├── inference_base.py
│           ├── evaluate.py
│           └── run_inference.sh
├── requirements.txt          # Python dependencies
├── .gitignore
├── PROMPT_UNIFICATION_REPORT.md
└── README.md
```

---

## 🚀 Training

### Stage 1: COT Data Generation

```bash
# Generate initial COT data using GPT-4o/Claude
cd src/stage1_cot
python generate_cot.py \
  --input_data /path/to/training_data.json \
  --output_dir ./cot_data \
  --api_key YOUR_API_KEY \
  --model gpt-4o

# Enhance COT quality
python enhance_cot.py \
  --input_dir ./cot_data \
  --output_dir ./cot_enhanced
```

### Stage 2: GRPO Training

```bash
# GRPO with COT (Recommended)
bash src/scripts/train_grpo_with_cot.sh

# Supervised Fine-Tuning (Baseline)
bash src/scripts/train_sft.sh
```

#### Training Configuration

- **Model**: Qwen2.5-VL-3B-Instruct
- **Data**: 200k samples with automated COT
- **Hardware**: 4×A800 80GB GPUs
- **Batch Size**: 160 (10 per device × 4 grad accum × 4 GPUs)
- **Epochs**: 3
- **Training Time**: ~69 hours
- **DeepSpeed**: ZeRO-3 optimization

---

## 📊 Data Format

### Training Data (with COT)

```json
[
  {
    "image": "images/tomato_leaf_disease_001.jpg",
    "question": "What disease is affecting this tomato plant?",
    "conversations": [
      {
        "from": "user",
        "value": "<image>\nWhat disease is affecting this tomato plant?"
      },
      {
        "from": "assistant",
        "value": "<think>Step 1: Observe leaf characteristics - yellowing between veins, dark spots.\nStep 2: Analyze spot patterns - circular lesions with concentric rings.\nStep 3: Consider disease etiology - fungal pathogen Alternaria solani.\nStep 4: Confirm diagnosis based on visual evidence.</think><answer>Tomato Early Blight (Alternaria solani)</answer>"
      }
    ]
  }
]
```

### Evaluation Data (Multiple Choice)

```json
[
  {
    "id": "agmmu_001",
    "images": ["images/wheat_disease.jpg"],
    "question": "What is the most likely disease? A) Wheat Rust B) Powdery Mildew C) Fusarium Head Blight D) Septoria Leaf Spot",
    "answer": "A) Wheat Rust",
    "qtype": "disease_identification"
  }
]
```

---

## 🤖 Inference

### CDDMBench Evaluation

```bash
# Crop/Disease Classification (with COT)
cd src/eval_vqa/cddmbench/crop_disease
python inference_grpo_cot.py \
  --model_path /path/to/checkpoint \
  --input_json test_data.json \
  --output_json predictions.json

# Knowledge QA (with COT)
cd src/eval_vqa/cddmbench/knowledge
python inference_grpo_cot.py \
  --model_path /path/to/checkpoint \
  --input_json test_data.json \
  --output_json predictions.json

# Evaluate
python evaluate.py \
  --ground-truth-file test_data.json \
  --model-answers-file predictions.json \
  --output-file evaluation_results.json
```

### AgMMU Evaluation

```bash
cd src/eval_vqa/agmmu

# Inference with COT
python inference_with_cot.py \
  --model_path /path/to/checkpoint \
  --data_path agmmu_validation.json \
  --image_dir ./images \
  --output predictions.json

# Evaluate
python evaluate.py \
  --predictions predictions.json \
  --ground_truth agmmu_validation.json
```

### Python Inference Example

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/checkpoint",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("path/to/checkpoint")

# Prepare input
image = Image.open("tomato_disease.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What disease is this? Provide step-by-step reasoning."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

# Generate
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)

print(response)
# Output: <think>Step 1: ... Step 2: ...</think><answer>Tomato Early Blight</answer>
```

---

## 🎯 Evaluation Benchmarks

### CDDMBench
- **Crop Classification**: 40 crop species
- **Disease Classification**: 120+ disease types
- **Knowledge QA**: Expert-annotated questions on disease control, symptoms, pathogens

### AgMMU (Generalization)
- **Out-of-distribution** agricultural dataset
- **Multiple choice questions** covering global crop diseases
- Tests model's ability to generalize beyond training data

### MIRAGE
- **Multimodal reasoning** benchmark
- Agricultural scenarios requiring complex inference

---

## 📁 Example Results

See `Splits/results/` for example inference outputs:
- `crop_disease/` - Classification predictions with COT reasoning
- `knowledge/` - QA responses with detailed explanations
- `agmmu/` - Generalization test results

---

## 🙏 Acknowledgements

We thank the authors of the following open-source projects:
- [R1-V](https://github.com/Deep-Agent/R1-V) - GRPO training framework
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) - Base vision-language model
- [CDDMBench](https://github.com/CDDMBench/CDDMBench) - Agricultural VQA benchmark
- [AgMMU](https://github.com/AgMMU/AgMMU) - Generalization benchmark

---

## 📚 Citation

```bibtex
@inproceedings{agri-r1-2025,
  title={Agri-R1: Automated Chain-of-Thought for Agricultural Disease Diagnosis},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Email: your.email@university.edu
- Issues: [GitHub Issues](https://github.com/your-username/Agri-R1/issues)

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
