# Stage 1: Reasoning Data Generation Pipeline

This directory contains scripts for generating Chain-of-Thought (Reasoning) training data for agricultural Visual Question Answering (VQA) tasks.

## Overview

The Reasoning generation pipeline consists of 4 main steps:

1. **Image Preprocessing** - Resize all images to 384×384 resolution
2. **Dataset Sampling** - Extract 20k samples from the full dataset
3. **Reasoning Generation** - Generate step-by-step reasoning using external VLM
4. **Reasoning Enhancement** - Evaluate and optimize generated Reasoning quality

## Directory Structure

```
stage1_reasoning/
├── README.md                    # This file
├── resize_images_384.py         # Step 1: Image preprocessing
├── sample_dataset_20k.py        # Step 2: Dataset sampling
├── generate_reasoning.py              # Step 3: Reasoning generation
├── enhance_reasoning.py               # Step 4: Reasoning enhancement
└── (deprecated files...)        # Old files kept for reference
```

---

## Step 1: Image Preprocessing

### Script: `resize_images_384.py`

Resizes all images in the dataset to 384×384 pixels using smart padding to preserve content.

### Features:
- Batch processing of agricultural disease images
- Smart padding based on aspect ratio (no content cropping)
- Intelligent background color filling using dominant color
- Converts all formats to high-quality JPG

### Usage:

```bash
python resize_images_384.py \
    --dataset_path /path/to/dataset \
    --target_size 384
```

### Parameters:
- `--dataset_path` (required): Path to dataset directory
- `--target_size` (optional): Target image size in pixels (default: 384)

### Output:
- All images resized to 384×384 JPG format
- Statistics printed: total processed, successful, failed, format changed

### Example:

```bash
python resize_images_384.py --dataset_path ./agricultural_dataset
```

---

## Step 2: Dataset Sampling

### Script: `sample_dataset_20k.py`

Extracts 20,000 samples from a larger dataset with optional stratified sampling.

### Features:
- Random sampling with reproducible seed
- Stratified sampling to maintain class distribution
- Image path validation before sampling
- JSON output compatible with training pipeline

### Usage:

```bash
python sample_dataset_20k.py \
    --input input_dataset.json \
    --output sampled_20k.json \
    --sample_size 20000 \
    --stratified \
    --validate \
    --seed 42
```

### Parameters:
- `--input` (required): Input JSON file path
- `--output` (required): Output JSON file path
- `--sample_size` (optional): Number of samples (default: 20000)
- `--stratified` (optional): Use stratified sampling
- `--validate` (optional): Validate image paths before sampling
- `--class_key` (optional): Key for stratification (default: 'answer')
- `--seed` (optional): Random seed (default: 42)

### Input Format:

```json
[
  {
    "image": "/path/to/image.jpg",
    "question": "What disease is this?",
    "answer": "Tomato Early Blight"
  },
  ...
]
```

### Output Format:

Same as input, with 20k randomly/stratified sampled entries.

### Example:

```bash
# Random sampling
python sample_dataset_20k.py \
    --input full_dataset.json \
    --output train_20k.json \
    --sample_size 20000 \
    --validate

# Stratified sampling (recommended)
python sample_dataset_20k.py \
    --input full_dataset.json \
    --output train_20k_stratified.json \
    --sample_size 20000 \
    --stratified \
    --validate
```

---

## Step 3: Reasoning Generation

### Script: `generate_reasoning.py`

Generates Chain-of-Thought reasoning for each question using an external VLM API (e.g., DeepSeek-VL2, GPT-4V).

### Features:
- Automatic question type detection (diagnosis vs. treatment)
- Few-shot prompting with curated examples
- Structured 4-step reasoning format
- Batch processing with progress tracking
- Auto-save every 10k samples

### Usage:

```bash
python generate_reasoning.py \
    --input train_20k.json \
    --output train_20k_with_reasoning.json \
    --api_key YOUR_API_KEY \
    --api_base https://api.provider.com/v1 \
    --model_name deepseek-vl2 \
    --target_count 20000
```

### Parameters:
- `--input` (required): Input JSON file
- `--output` (required): Output JSON file
- `--api_key` (required): API key for VLM service
- `--api_base` (required): API base URL
- `--model_name` (optional): Model name (default: deepseek-vl2)
- `--target_count` (optional): Total Reasoning samples to generate (default: 200000)
- `--save_interval` (optional): Save checkpoint every N samples (default: 10000)
- `--temperature` (optional): Sampling temperature (default: 0.3)

### Output Format:

```json
[
  {
    "image": "/path/to/image.jpg",
    "problem": "What disease is this?",
    "solution": "<think>Step 1: Identify plant - tomato leaf based on morphology.\nStep 2: Observe symptoms - brown circular lesions with concentric rings.\nStep 3: Analyze pattern - target-like spots characteristic of fungal infection.\nStep 4: Diagnosis - Tomato Early Blight (Alternaria solani).</think><answer>Tomato Early Blight</answer>"
  },
  ...
]
```

### Question Type Detection:

The script automatically detects two question types:

1. **Diagnosis Questions** (99.6% of dataset)
   - Keywords: "what is", "identify", "name", "diagnose"
   - Reasoning Structure: Plant ID → Symptom observation → Pattern analysis → Diagnosis

2. **Treatment Questions** (0.4% of dataset)
   - Keywords: "control", "prevent", "treatment", "methods"
   - Reasoning Structure: Disease characteristics → Cultural practices → Chemical control → Application methods

### Example:

```bash
python generate_reasoning.py \
    --input train_20k.json \
    --output train_20k_reasoning.json \
    --api_key sk-xxxxx \
    --api_base https://api.siliconflow.cn/v1 \
    --model_name deepseek-vl2 \
    --target_count 20000
```

---

## Step 4: Reasoning Enhancement

### Script: `enhance_reasoning.py`

Evaluates generated Reasoning quality and automatically optimizes low-quality samples.

### Features:
- Multi-criteria evaluation (accuracy, completeness, detail, relevance, clarity)
- Automatic optimization for samples below threshold
- Structured output parsing with retry logic
- Progress tracking with statistics

### Usage:

```bash
python enhance_reasoning.py \
    --input train_20k_reasoning.json \
    --output train_20k_reasoning_enhanced.json \
    --api_key YOUR_OPENAI_API_KEY \
    --api_base YOUR_API_BASE \
    --threshold 8 \
    --model gpt-4o
```

### Parameters:
- `--input` (required): Input JSON file with Reasoning
- `--output` (required): Output JSON file with enhanced Reasoning
- `--api_key` (required): OpenAI-compatible API key
- `--api_base` (required): API base URL
- `--threshold` (optional): Quality threshold for optimization (default: 8)
- `--model` (optional): Model name (default: gpt-4o)

### Evaluation Criteria:

Each Reasoning is scored 1-10 on:

1. **Accuracy** (30%) - Correct plant and disease identification
2. **Completeness** (25%) - All key elements present
3. **Detail** (20%) - Specific symptom descriptions
4. **Relevance** (15%) - Agricultural diagnosis relevance
5. **Clarity** (10%) - Professional language, 80-120 words

### Rating Scale:

- **1-3**: Poor (vague, inaccurate, missing information) → **Auto-optimize**
- **4-6**: Fair (useful but incomplete) → **Auto-optimize**
- **7-8**: Good (clear, mostly accurate) → **Keep as-is**
- **9-10**: Excellent (precise, highly relevant) → **Keep as-is**

### Output:

Same format as input, with optimized Reasoning for low-quality samples.

### Example:

```bash
python enhance_reasoning.py \
    --input train_20k_reasoning.json \
    --output train_20k_reasoning_enhanced.json \
    --api_key sk-xxxxx \
    --api_base https://api.openai.com/v1 \
    --threshold 8 \
    --model gpt-4o
```

---

## Complete Pipeline Example

Here's a complete example of running all 4 steps:

```bash
# Step 1: Resize images to 384×384
python resize_images_384.py \
    --dataset_path /data/agricultural_images

# Step 2: Sample 20k entries with stratified sampling
python sample_dataset_20k.py \
    --input /data/full_dataset.json \
    --output /data/sampled_20k.json \
    --sample_size 20000 \
    --stratified \
    --validate

# Step 3: Generate Reasoning reasoning
python generate_reasoning.py \
    --input /data/sampled_20k.json \
    --output /data/sampled_20k_reasoning.json \
    --api_key sk-deepseek-xxxxx \
    --api_base https://api.siliconflow.cn/v1 \
    --model_name deepseek-vl2 \
    --target_count 20000

# Step 4: Enhance Reasoning quality
python enhance_reasoning.py \
    --input /data/sampled_20k_reasoning.json \
    --output /data/sampled_20k_reasoning_enhanced.json \
    --api_key sk-openai-xxxxx \
    --api_base https://api.openai.com/v1 \
    --threshold 8 \
    --model gpt-4o
```

---

## Output Data Format

### Final Training Data Format:

After completing all steps, the final training data will be in this format:

```json
[
  {
    "image": "/data/agricultural_images/tomato_001.jpg",
    "problem": "What disease is shown in this image?",
    "solution": "<think>Step 1: Identify plant species - This is a tomato leaf based on the ovate shape with serrated margins and pinnate venation pattern.\nStep 2: Observe symptoms - Multiple circular brown lesions (3-8mm diameter) with concentric rings are visible on the leaf surface, showing the characteristic 'target spot' pattern.\nStep 3: Analyze disease pattern - The lesions display dark brown centers surrounded by lighter brown rings, typical of fungal infection. Distribution is scattered across the leaf with some coalescence.\nStep 4: Preliminary diagnosis - Based on the target-like lesions and tomato host, this is Early Blight caused by Alternaria solani; confidence level: high.</think><answer>Tomato Early Blight (Alternaria solani)</answer>"
  },
  ...
]
```

### Field Descriptions:

- `image`: Path to 384×384 JPG image
- `problem`: The question to be answered
- `solution`: Complete solution with Reasoning reasoning
  - `<think>...</think>`: Step-by-step reasoning (4 steps)
  - `<answer>...</answer>`: Final answer

---

## Configuration

### API Configuration:

Edit the API configuration in `generate_reasoning.py` and `enhance_reasoning.py`:

```python
# For Reasoning Generation (generate_reasoning.py)
API_BASE = "https://api.siliconflow.cn/v1"  # Your VLM API base URL
API_KEY = "sk-xxxxx"                         # Your API key

# For Reasoning Enhancement (enhance_reasoning.py)
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-xxxxx"
```

### Model Selection:

Recommended models for each step:

- **Reasoning Generation**: DeepSeek-VL2, GPT-4V, Claude-3.5-Sonnet
- **Reasoning Enhancement**: GPT-4o, GPT-4-Turbo, Claude-3-Opus

---

## Quality Control

### Reasoning Quality Metrics:

After enhancement, expect:

- **90%+** samples with rating ≥ 7
- **70%+** samples with rating ≥ 8
- **40%+** samples with rating ≥ 9

### Manual Review:

Recommended to manually review:
- 100 random samples for overall quality
- 50 samples with rating 6-7 (edge cases)
- 20 samples with rating ≤ 5 (if any remain)

---

## Troubleshooting

### Common Issues:

1. **API Rate Limits**
   - Solution: Increase `time.sleep()` delays in generate_reasoning.py
   - Or use `--save_interval` to checkpoint progress

2. **Out of Memory**
   - Solution: Reduce batch size in image preprocessing
   - Or process dataset in smaller chunks

3. **Invalid Image Paths**
   - Solution: Use `--validate` flag in sampling step
   - Or fix paths before running pipeline

4. **Low Reasoning Quality Scores**
   - Solution: Adjust `--threshold` in enhancement step
   - Or improve few-shot examples in generate_reasoning.py

---

## Performance

### Expected Processing Times:

Based on 20k dataset:

- **Step 1 (Resize)**: ~10-30 minutes (depends on dataset size)
- **Step 2 (Sample)**: ~1-2 minutes
- **Step 3 (Generate Reasoning)**: ~6-10 hours (API-dependent)
- **Step 4 (Enhance)**: ~2-4 hours (API-dependent)

### Total Pipeline Time: **~8-15 hours**

---

## Cost Estimation

### API Costs (approximate):

For 20k samples:

- **DeepSeek-VL2**: ~$20-40
- **GPT-4V**: ~$100-200
- **GPT-4o** (enhancement): ~$40-80

### Total Estimated Cost: **$60-120** for 20k high-quality Reasoning samples

---

## Citation

If you use this Reasoning generation pipeline in your research, please cite:

```bibtex
@article{agri-r1-2025,
  title={Agri-R1: Enhancing Agricultural Visual Question Answering with Chain-of-Thought Reasoning},
  author={Your Name},
  journal={ACL 2025},
  year={2025}
}
```

---

## License

Apache 2.0 License - See LICENSE file for details

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
