#!/bin/bash

# 检查命令行参数
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 [train|test|process] [dataset]"
    echo "dataset: fewrel|tacred|all (only for process)"
    exit 1
fi

ACTION=$1
DATASET=${2:-"fewrel"}  # 默认使用fewrel

# 设置基础目录
BASE_DIR=$(pwd)
cd $BASE_DIR

# 根据参数执行相应脚本
case $ACTION in
    "train")
        echo "Starting training on ${DATASET}..."
        bash scripts/train.sh ${DATASET}
        ;;
    "test")
        echo "Starting testing on ${DATASET}..."
        bash scripts/test.sh ${DATASET}
        ;;
    "process")
        echo "Starting data processing for ${DATASET}..."
        bash scripts/process_data.sh ${DATASET}
        ;;
    *)
        echo "Invalid action. Use 'train', 'test' or 'process'"
        exit 1
        ;;
esac 