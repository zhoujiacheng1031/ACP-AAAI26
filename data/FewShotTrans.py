# -*- coding: utf-8 -*-
"""
Created on 2024-07-26

@author: zhoujiacheng
@file: FewShotTrans.py
@purpose: Transform dataset for few-shot learning with balanced sampling
"""

import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import logging
import json
from tqdm import tqdm

class FewShotTransformer:
    """小样本学习数据转换器"""
    
    def __init__(
        self,
        dataset_path: str,
        relation_types: List[str],
        n_way: int = 5,
        k_shot: int = 5,
        output_dir: str = None,
        pos_neg_ratio: float = 0.2
    ):
        self.logger = logging.getLogger(__name__)
        self.dataset_path = dataset_path
        self.relation_types = relation_types
        self.n_way = n_way
        self.k_shot = k_shot
        self.output_dir = output_dir
        self.pos_neg_ratio = pos_neg_ratio
        
        # 加载数据
        self.data = self._load_data()
        
        # 按关系分组
        self.rel2insts = self._group_by_relation()
        
    def transform(self) -> Tuple[List[Dict], List[Dict]]:
        """转换数据集
        
        Returns:
            (C_FS, D_FS): 支持集和查询集任务列表
        """
        C_FS = []  # 支持集任务
        D_FS = []  # 查询集任务
        
        # 生成任务
        for _ in tqdm(range(1000), desc='Generating tasks'):
            # 随机选择n_way个关系
            selected_rels = random.sample(self.relation_types, self.n_way)
            
            support_set = []
            query_set = []
            
            # 为每个关系采样实例
            for rel in selected_rels:
                instances = self.rel2insts[rel]
                if len(instances) < self.k_shot * 2:
                    continue
                    
                # 随机采样不重叠的支持集和查询集
                sampled = random.sample(instances, self.k_shot * 2)
                support_set.extend(sampled[:self.k_shot])
                query_set.extend(sampled[self.k_shot:])
            
            # 生成NOTA样本
            if random.random() < self.pos_neg_ratio:
                num_nota = int(len(query_set) * self.pos_neg_ratio)
                nota_samples = self._generate_nota_samples(num_nota)
                query_set.extend(nota_samples)
            
            task = {
                'support': support_set,
                'query': query_set,
                'relations': selected_rels
            }
            
            # 80%作为训练任务,20%作为开发任务
            if random.random() < 0.8:
                C_FS.append(task)
            else:
                D_FS.append(task)
                
        # 保存转换后的数据
        if self.output_dir:
            self._save_tasks(C_FS, D_FS)
            
        return C_FS, D_FS
        
    def _load_data(self) -> List[Dict]:
        """加载数据集"""
        data = []
        for split in ['train', 'dev', 'test']:
            file_path = os.path.join(self.dataset_path, f'{split}.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    split_data = json.load(f)
                    data.extend(split_data)
        return data
        
    def _group_by_relation(self) -> Dict[str, List[Dict]]:
        """按关系对实例分组"""
        grouped = defaultdict(list)
        for inst in self.data:
            grouped[inst['relation']].append(inst)
        return grouped
        
    def _generate_nota_samples(self, num_samples: int) -> List[Dict]:
        """生成NOTA样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            NOTA样本列表
        """
        nota_samples = []
        all_instances = [
            inst for insts in self.rel2insts.values()
            for inst in insts
        ]
        
        for _ in range(num_samples):
            # 随机选择一个实例作为NOTA样本
            inst = random.choice(all_instances)
            nota_inst = inst.copy()
            nota_inst['relation'] = 'NOTA'
            nota_samples.append(nota_inst)
            
        return nota_samples
        
    def _save_tasks(self, C_FS: List[Dict], D_FS: List[Dict]):
        """保存任务数据"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存训练任务
        train_file = os.path.join(self.output_dir, 'train.json')
        with open(train_file, 'w') as f:
            json.dump(C_FS, f, indent=2)
            
        # 保存开发任务
        dev_file = os.path.join(self.output_dir, 'dev.json')
        with open(dev_file, 'w') as f:
            json.dump(D_FS, f, indent=2)

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载关系类型
    with open('CoIn/data/processed/fewrel/rel2id.json', 'r') as f:
        rel2id = json.load(f)
    relation_types = list(rel2id.keys())
    
    # 转换FewRel数据集
    transformer = FewShotTransformer(
        dataset_path='CoIn/data/processed/fewrel',
        relation_types=relation_types,
        n_way=5,
        k_shot=5,
        output_dir='CoIn/data/few_shot/fewrel',
        pos_neg_ratio=0.2
    )
    
    C_FS, D_FS = transformer.transform()
    print("Dataset transformation completed!")

if __name__ == '__main__':
    main()