#!/bin/bash

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [fewrel|tacred]"
    exit 1
fi

DATASET=$1

# 测试模型
python test.py \
    --dataset ${DATASET} \
    --test_file data/${DATASET}/test.json \
    --model_path output/${DATASET}/best_model.pt \
    --n_way 5 \
    --k_shot 5 \
    --pretrained_model bert-base-uncased \
    --hidden_size 768 \
    --num_attention_heads 8 \
    --output_dir output/${DATASET} \
    --cache_dir cache/${DATASET} \
    --batch_size 4 \
    --num_workers 4 \
    --seed 42 \
    2>&1 | tee logs/${DATASET}/test.log 