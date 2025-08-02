#!/bin/bash

# MACAP Evaluation Script

echo "📊 Starting MACAP evaluation"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load environment variables
if [ -f ".env" ]; then
    echo "📋 Loading environment variables..."
    export $(cat .env | xargs)
fi

# Run evaluation
echo "🔍 Running evaluation..."
python -m macap.utils.evaluate \
    --checkpoint results/experiments/best_model.pt \
    --config macap/config/agent_config.json \
    --output_dir results/evaluation

echo "✅ Evaluation completed"