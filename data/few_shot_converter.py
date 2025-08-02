# -*- coding: utf-8 -*-
"""
@purpose: Convert datasets to few-shot learning format
"""

import json
import os
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

class FewShotConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convert_dataset(
        self,
        data_dir: str,
        output_dir: str,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 5
    ):
        """将数据集转换为小样本学习格式
        
        Args:
            data_dir: 输入数据目录
            output_dir: 输出目录
            n_way: 分类数量
            k_shot: 支持集样本数
            n_query: 查询集样本数
        """
        # 加载数据
        train_data = self._load_split(data_dir, 'train')
        dev_data = self._load_split(data_dir, 'dev')
        test_data = self._load_split(data_dir, 'test')
        rel2id = self._load_rel2id(data_dir)
        
        # 按关系分组
        train_by_rel = self._group_by_relation(train_data)
        dev_by_rel = self._group_by_relation(dev_data)
        test_by_rel = self._group_by_relation(test_data)
        
        # 生成任务
        train_tasks = self._generate_tasks(
            train_by_rel,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_tasks=1000
        )
        
        dev_tasks = self._generate_tasks(
            dev_by_rel,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_tasks=200
        )
        
        test_tasks = self._generate_tasks(
            test_by_rel,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_tasks=200
        )
        
        # 保存转换后的数据
        os.makedirs(output_dir, exist_ok=True)
        
        self._save_tasks(train_tasks, os.path.join(output_dir, 'train.json'))
        self._save_tasks(dev_tasks, os.path.join(output_dir, 'dev.json'))
        self._save_tasks(test_tasks, os.path.join(output_dir, 'test.json'))
        
        # 保存关系映射
        with open(os.path.join(output_dir, 'rel2id.json'), 'w') as f:
            json.dump(rel2id, f, indent=2)
            
    def _load_split(self, data_dir: str, split: str) -> List[Dict]:
        """加载数据集划分"""
        file_path = os.path.join(data_dir, f'{split}.json')
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def _load_rel2id(self, data_dir: str) -> Dict[str, int]:
        """加载关系ID映射"""
        file_path = os.path.join(data_dir, 'rel2id.json')
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def _group_by_relation(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """按关系对数据分组"""
        grouped = defaultdict(list)
        for instance in data:
            grouped[instance['relation']].append(instance)
        return grouped
        
    def _generate_tasks(
        self,
        data_by_rel: Dict[str, List[Dict]],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_tasks: int
    ) -> List[Dict]:
        """生成小样本学习任务"""
        tasks = []
        relations = list(data_by_rel.keys())
        
        for _ in range(n_tasks):
            # 随机选择n_way个关系
            selected_rels = random.sample(relations, n_way)
            
            support_set = []
            query_set = []
            
            # 为每个关系采样支持集和查询集
            for rel in selected_rels:
                instances = data_by_rel[rel]
                if len(instances) < k_shot + n_query:
                    # 如果样本不足则跳过
                    continue
                    
                # 随机采样不重叠的支持集和查询集
                sampled = random.sample(instances, k_shot + n_query)
                support_set.extend(sampled[:k_shot])
                query_set.extend(sampled[k_shot:k_shot + n_query])
            
            tasks.append({
                'support': support_set,
                'query': query_set,
                'relations': selected_rels
            })
            
        return tasks
        
    def _save_tasks(self, tasks: List[Dict], output_file: str):
        """保存任务数据"""
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    converter = FewShotConverter()
    
    # 转换FewRel数据集
    converter.convert_dataset(
        data_dir='CoIn/data/processed/fewrel',
        output_dir='CoIn/data/few_shot/fewrel'
    )
    
    # 转换TACRED数据集
    converter.convert_dataset(
        data_dir='CoIn/data/processed/tacred', 
        output_dir='CoIn/data/few_shot/tacred'
    )

if __name__ == '__main__':
    main() 