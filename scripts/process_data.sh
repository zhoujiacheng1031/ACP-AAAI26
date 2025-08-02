#!/bin/bash

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [fewrel|tacred|all]"
    exit 1
fi

DATASET=$1

process_dataset() {
    local dataset=$1
    echo "Processing ${dataset} dataset..."
    
    # 处理数据集
    python data/process_datasets.py \
        --input_dir data/${dataset} \
        --output_dir data/${dataset} \
        --dataset ${dataset}
    
    # 生成负样本
    echo "Generating negative samples for ${dataset}..."
    python data/generate_negative_samples.py \
        --input_file data/${dataset}/train.json \
        --output_file data/${dataset}/train_with_neg.json \
        --num_samples 2
    
    # 检索概念
    echo "Retrieving concepts for ${dataset}..."
    python utils/concept_retriever.py \
        --input_file data/${dataset}/train_with_neg.json \
        --cache_dir cache/concept/${dataset}
}

case $DATASET in
    "fewrel")
        process_dataset "fewrel"
        ;;
    "tacred")
        process_dataset "tacred"
        ;;
    "all")
        process_dataset "fewrel"
        process_dataset "tacred"
        ;;
    *)
        echo "Invalid dataset. Use 'fewrel', 'tacred' or 'all'"
        exit 1
        ;;
esac

echo "Data processing completed!" 