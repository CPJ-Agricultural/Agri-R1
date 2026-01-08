# DeepSpeed Configuration for Agri-R1 Training

## Overview

This directory contains DeepSpeed ZeRO configurations optimized for multi-GPU training of the Agri-R1 vision-language model. We use **ZeRO Stage 3** to achieve balanced GPU memory utilization across all devices.

## Configuration Files

### `ds_zero3.json` - ZeRO Stage 3 (Recommended)

**Purpose**: Balanced memory distribution across 4×A800 80GB GPUs for stable, efficient training.

#### Key Parameters

```json
{
  "train_micro_batch_size_per_gpu": 10,
  "gradient_accumulation_steps": 4,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 5e8,
    "reduce_bucket_size": 2e8,
    "stage3_prefetch_bucket_size": 2e8,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "round_robin_gradients": true
  },
  "gradient_clipping": 0.3,
  "steps_per_print": 10,
  "wall_clock_breakdown": false,
  "dump_state": false
}
```

---

## Why ZeRO Stage 3?

### Problem with ZeRO Stage 2

During initial experiments with ZeRO Stage 2 (`ds_zero2.json`), we observed **severe memory imbalance** across GPUs:

| GPU | Memory Usage | Status |
|-----|--------------|--------|
| GPU 0 | **~90%** | Bottleneck |
| GPU 1 | 20-30% | Underutilized |
| GPU 2 | 20-30% | Underutilized |
| GPU 3 | 20-30% | Underutilized |

**Root Cause**: ZeRO-2 only partitions optimizer states and gradients, but keeps **full model parameters** on each GPU. For vision-language models like Qwen2.5-VL-3B with large vision encoders, this causes:
- GPU 0 (rank 0) holds the full parameter set during forward/backward passes
- Other GPUs only store sharded optimizer states
- Memory bottleneck limits effective batch size and training stability

### Solution with ZeRO Stage 3

ZeRO-3 partitions **model parameters, gradients, and optimizer states** across all GPUs, achieving balanced utilization:

| GPU | Memory Usage | Improvement |
|-----|--------------|-------------|
| GPU 0 | **80-85%** | -5-10% ✅ |
| GPU 1 | **80-85%** | +50-65% ✅ |
| GPU 2 | **80-85%** | +50-65% ✅ |
| GPU 3 | **80-85%** | +50-65% ✅ |

**Benefits**:
1. **Balanced workload**: All GPUs contribute equally
2. **Higher effective batch size**: Can increase from 8→10 samples/GPU
3. **Stable training**: No single-GPU memory bottleneck
4. **Better scalability**: Linear scaling with more GPUs

---

## Parameter Explanations

### Core ZeRO-3 Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `stage` | 3 | Enable full parameter partitioning |
| `offload_optimizer` | `"none"` | Keep optimizer on GPU for speed |
| `offload_param` | `"none"` | Keep parameters on GPU (A800 has sufficient 80GB memory) |

**Note**: We disable CPU offloading because A800 80GB GPUs have sufficient memory. Offloading would slow training by 2-3× due to PCIe transfer overhead.

### Memory Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sub_group_size` | 5e8 (500M) | Parameters per partition group; smaller values = finer-grained partitioning but higher communication overhead |
| `reduce_bucket_size` | 2e8 (200M) | Gradient reduction bucket size; balances communication frequency vs. latency |
| `stage3_prefetch_bucket_size` | 2e8 (200M) | Parameter prefetch bucket size; larger values improve throughput but increase memory spikes |
| `stage3_param_persistence_threshold` | 1e5 (100K) | Parameters smaller than this stay in GPU memory permanently; reduces gather/scatter overhead |
| `stage3_max_live_parameters` | 1e9 (1B) | Maximum parameters kept in GPU memory simultaneously; tuned for 80GB A800 |
| `stage3_max_reuse_distance` | 1e9 (1B) | Parameter reuse distance for caching; prevents redundant fetches in recurrent layers |

### Communication Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `overlap_comm` | `true` | Overlap gradient communication with backward computation; reduces training time by ~15% |
| `contiguous_gradients` | `true` | Store gradients in contiguous memory; improves all-reduce efficiency |
| `round_robin_gradients` | `true` | Distribute gradients evenly across GPUs; prevents memory spikes on rank 0 |
| `stage3_gather_16bit_weights_on_model_save` | `true` | Gather BF16 weights (not FP32) when saving; reduces checkpoint size by 50% |

### Training Stability

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bf16.enabled` | `true` | Use BF16 mixed precision; better numerical stability than FP16 for vision-language models |
| `gradient_clipping` | 0.3 | Clip gradients to max norm 0.3; prevents training instability from reward spikes in GRPO |
| `train_micro_batch_size_per_gpu` | 10 | Per-GPU batch size; maximized for 80GB memory |
| `gradient_accumulation_steps` | 4 | Accumulate gradients over 4 steps; effective batch size = 10×4 GPUs×4 steps = **160** |

---

## Comparison: ZeRO-2 vs ZeRO-3

| Aspect | ZeRO-2 | ZeRO-3 | Winner |
|--------|--------|--------|--------|
| **Parameter Partitioning** | ❌ Full copy on each GPU | ✅ Sharded across GPUs | ZeRO-3 |
| **Gradient Partitioning** | ✅ Sharded | ✅ Sharded | Tie |
| **Optimizer Partitioning** | ✅ Sharded | ✅ Sharded | Tie |
| **Memory Balance (4 GPUs)** | 90% / 30% / 30% / 30% | 82% / 82% / 82% / 82% | ZeRO-3 |
| **Max Batch Size/GPU** | 8 | **10** (+25%) | ZeRO-3 |
| **Communication Overhead** | Lower | ~10% higher | ZeRO-2 |
| **Training Speed** | Baseline | -5% (due to communication) | ZeRO-2 |
| **Scalability** | Poor (bottlenecked by GPU 0) | Excellent (linear scaling) | ZeRO-3 |
| **Overall Recommendation** | ❌ Not recommended for VLMs | ✅ **Recommended** | ZeRO-3 |

**Bottom Line**: For vision-language models with large parameter counts, ZeRO-3's balanced memory distribution outweighs the minor communication overhead.

---

## Usage

### Training Script

```bash
deepspeed --num_gpus=4 train_grpo_with_reasoning.py \
  --model_name_or_path Qwen2.5-VL-3B-Instruct \
  --dataset_name ./data/training_data \
  --deepspeed configs/ds_zero3.json \
  --learning_rate 8e-7 \
  --num_train_epochs 3 \
  --gradient_checkpointing true \
  --bf16 true \
  --max_grad_norm 0.3
```

### Monitoring Memory Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Expected output with ZeRO-3:
# GPU 0: 63-68 GB / 80 GB (80-85%)
# GPU 1: 63-68 GB / 80 GB (80-85%)
# GPU 2: 63-68 GB / 80 GB (80-85%)
# GPU 3: 63-68 GB / 80 GB (80-85%)
```

---

## Tuning Guidelines

### If You See Memory Imbalance

If GPU 0 still uses significantly more memory than others:

1. **Check parameter partitioning is enabled**:
   ```json
   "stage": 3  // Must be 3, not 2
   ```

2. **Reduce `sub_group_size`** for finer partitioning:
   ```json
   "sub_group_size": 2e8  // Try 200M instead of 500M
   ```

3. **Enable CPU offloading** if necessary (sacrifices speed):
   ```json
   "offload_param": {
     "device": "cpu",
     "pin_memory": true
   }
   ```

### If Training is Too Slow

If ZeRO-3 communication overhead is bottleneck:

1. **Increase bucket sizes** to reduce communication frequency:
   ```json
   "reduce_bucket_size": 5e8,
   "stage3_prefetch_bucket_size": 5e8
   ```

2. **Enable `overlap_comm`** (already enabled in our config):
   ```json
   "overlap_comm": true
   ```

3. **Ensure InfiniBand/NVLink** is being used (check with `nvidia-smi topo -m`)

### If Out-of-Memory (OOM)

If you still encounter OOM errors despite ZeRO-3:

1. **Reduce `train_micro_batch_size_per_gpu`**:
   ```json
   "train_micro_batch_size_per_gpu": 8  // Down from 10
   ```

2. **Increase `gradient_accumulation_steps`** to maintain effective batch size:
   ```json
   "gradient_accumulation_steps": 5  // Up from 4
   // Effective batch size: 8×4×5 = 160 (unchanged)
   ```

3. **Reduce `stage3_max_live_parameters`**:
   ```json
   "stage3_max_live_parameters": 5e8  // Down from 1e9
   ```

---

## Performance Benchmarks

Based on training logs from Agri-R1 experiments:

| Metric | Value |
|--------|-------|
| **Training Speed** | 116 seconds/step |
| **Throughput** | 31 steps/hour |
| **Samples/Hour** | 4,960 (batch size 160) |
| **GPU Utilization** | ~95% average |
| **Memory per GPU** | 63-68 GB / 80 GB (80-85%) |
| **Total Training Time** | ~98 hours for 3,027 steps (2.4 epochs) |

---

## Troubleshooting

### Common Issues

**Issue 1: "Expected all tensors to be on the same device"**
- **Cause**: Mixed CPU/GPU offloading with incompatible operations
- **Fix**: Set both `offload_optimizer` and `offload_param` to `"none"`

**Issue 2: "CUDA out of memory" on GPU 0 only**
- **Cause**: Still using ZeRO-2 or parameter partitioning disabled
- **Fix**: Verify `"stage": 3` in config file

**Issue 3: Slow training (>200 sec/step)**
- **Cause**: CPU offloading or small bucket sizes causing communication bottleneck
- **Fix**: Disable offloading, increase bucket sizes to 5e8

**Issue 4: Checkpoints are too large (>40GB)**
- **Cause**: Saving FP32 weights instead of BF16
- **Fix**: Ensure `"stage3_gather_16bit_weights_on_model_save": true`

---

## References

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO Paper (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Qwen2.5-VL Training Guide](https://github.com/QwenLM/Qwen2.5-VL)
- Med-R1 used ZeRO-2; Agri-R1 upgraded to ZeRO-3 for better memory balance

---

## Contact

For questions about this configuration, please refer to:
- Training script: `src/r1-v/grpo_agri_vqa_v3.py`
- Training logs: `论文训练实验完整报告_GRPO_Qwen2.5-VL-3B (1).md`
- GitHub Issues: https://github.com/CPJ-Agricultural/Agri-R1/issues
