# -*- coding: utf-8 -*-
"""
@purpose: merge multiple datasets
"""

import json
import random
import argparse
import os
from typing import List, Dict
from tqdm import tqdm

def merge_datasets(
    input_dirs: List[str],
    output_dir: str,
    splits: List[str] = ['train', 'dev', 'test']
):
    """合并多个数据集
    
    Args:
        input_dirs: 输入数据集目录列表
        output_dir: 输出目录
        splits: 数据集划分
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个划分
    for split in splits:
        merged_data = []
        
        # 从每个数据集读取数据
        for data_dir in input_dirs:
            input_file = os.path.join(data_dir, f'{split}.json')
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.extend(data)
        
        # 随机打乱
        random.shuffle(merged_data)
        
        # 保存合并后的数据
        output_file = os.path.join(output_dir, f'{split}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True,
                       help='输入数据集目录列表')
    parser.add_argument('--output_dir', required=True,
                       help='输出目录')
    args = parser.parse_args()
    
    merge_datasets(args.input_dirs, args.output_dir)

if __name__ == '__main__':
    main()
