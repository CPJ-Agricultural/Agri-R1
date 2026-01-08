#!/bin/bash
# Agri-R1 Environment Setup Script

echo "Setting up Agri-R1 environment..."

# Install the packages in r1-v
cd src/r1-v
pip install -e ".[dev]"

# Additional modules for multimodal
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support for fast inference
pip install vllm==0.7.2

# Fix transformers version (required for Qwen2.5-VL)
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

# Additional dependencies for evaluation
pip install langchain-openai
pip install langchain-core

echo "Setup complete! You can now run training and evaluation scripts."
