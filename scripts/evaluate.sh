#!/bin/bash

# Evaluation Script

echo "📊 Starting evaluation"

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
python -m acp.utils.evaluate \
    --checkpoint results/experiments/best_model.pt \
    --config acp/config/agent_config.json \
    --output_dir results/evaluation

echo "✅ Evaluation completed"