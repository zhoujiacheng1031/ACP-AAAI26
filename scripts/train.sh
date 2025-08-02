#!/bin/bash

# MACAP Training Script with Multi-Agent System

echo "🚀 Starting MACAP training with multi-agent system"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p results/experiments
mkdir -p logs

# Load environment variables
if [ -f ".env" ]; then
    echo "📋 Loading environment variables..."
    export $(cat .env | xargs)
fi

# Run training
echo "🎯 Starting training..."
python -m acp.utils.train \
    --config acp/config/agent_config.json \
    --output_dir results/experiments \
    --debug

echo "✅ Training completed"
echo "📊 Results saved to: results/experiments/"