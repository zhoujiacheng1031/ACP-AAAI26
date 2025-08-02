# -*- coding: utf-8 -*-
"""
Created on 2024-07-26

@author: zhoujiacheng
@file: process_datasets.py
@purpose: Process datasets with NOTA samples
"""

import json
import os
import random
from typing import Dict, List
from tqdm import tqdm
import logging

class DatasetProcessor:
    def __init__(self, pos_neg_ratio: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.pos_neg_ratio = pos_neg_ratio
        
    def _generate_nota_samples(
        self,
        instances: List[Dict],
        support_relations: List[str],
        num_samples: int
    ) -> List[Dict]:
        """生成NOTA样本
        
        Args:
            instances: 原始实例列表
            support_relations: 当前episode的支持集关系
            num_samples: 需要生成的样本数量
            
        Returns:
            NOTA样本列表
        """
        nota_samples = []
        
        # 1. 背景关系样本 - 从support_relations中选择,但上下文不适用
        background_samples = []
        for inst in instances:
            if inst['relation'] in support_relations:
                # 检查上下文是否使该关系无效
                if self._is_invalid_context(inst):
                    background_inst = inst.copy()
                    background_inst['relation'] = 'NOTA'
                    background_inst['nota_type'] = 'background'
                    background_samples.append(background_inst)
        
        # 2. 未见过的关系样本 - 不在support_relations中的关系
        unseen_samples = []
        for inst in instances:
            if inst['relation'] not in support_relations:
                unseen_inst = inst.copy()
                unseen_inst['relation'] = 'NOTA'
                unseen_inst['nota_type'] = 'unseen'
                unseen_samples.append(unseen_inst)
        
        # 3. 无具体关系样本 - 原本的no_relation
        no_relation_samples = []
        for inst in instances:
            if inst['relation'] == 'no_relation':
                no_rel_inst = inst.copy()
                no_rel_inst['relation'] = 'NOTA'
                no_rel_inst['nota_type'] = 'no_relation'
                no_relation_samples.append(no_rel_inst)
        
        # 4. 不同实体样本 - 使用LLM生成的样本
        different_entity_samples = self._generate_different_entity_samples(
            instances,
            support_relations
        )
        
        # 按比例采样各类NOTA样本
        num_each_type = num_samples // 4
        nota_samples.extend(random.sample(background_samples, min(num_each_type, len(background_samples))))
        nota_samples.extend(random.sample(unseen_samples, min(num_each_type, len(unseen_samples))))
        nota_samples.extend(random.sample(no_relation_samples, min(num_each_type, len(no_relation_samples))))
        nota_samples.extend(random.sample(different_entity_samples, min(num_each_type, len(different_entity_samples))))
        
        # 如果样本不足,从其他类型中补充
        remaining = num_samples - len(nota_samples)
        if remaining > 0:
            all_samples = background_samples + unseen_samples + no_relation_samples + different_entity_samples
            additional_samples = random.sample(all_samples, min(remaining, len(all_samples)))
            nota_samples.extend(additional_samples)
        
        return nota_samples

    def _is_invalid_context(self, instance: Dict) -> bool:
        """检查上下文是否使关系无效
        
        Args:
            instance: 实例
            
        Returns:
            是否无效
        """
        # 实现上下文有效性检查的逻辑
        # 例如:检查时态、情态、否定等
        # 这里需要根据具体数据集特点来实现
        pass

    def _generate_different_entity_samples(
        self,
        instances: List[Dict],
        support_relations: List[str]
    ) -> List[Dict]:
        """使用LLM生成不同实体的NOTA样本
        
        Args:
            instances: 原始实例列表
            support_relations: 支持集关系
            
        Returns:
            生成的样本列表
        """
        # 使用negative_sampler生成样本
        samples = []
        if self.negative_sampler:
            for inst in random.sample(instances, min(len(instances), 10)):
                neg_samples = self.negative_sampler.generate_negative_samples(
                    inst,
                    num_samples=2,
                    sample_type='different_entity'
                )
                if neg_samples:
                    for sample in neg_samples:
                        sample['relation'] = 'NOTA'
                        sample['nota_type'] = 'different_entity'
                    samples.extend(neg_samples)
        return samples

    def _process_split(
        self,
        instances: List[Dict],
        include_nota: bool = True
    ) -> List[Dict]:
        """处理数据集划分
        
        Args:
            instances: 原始实例列表
            include_nota: 是否包含NOTA样本
            
        Returns:
            处理后的实例列表
        """
        processed = []
        
        # 获取所有可能的关系
        all_relations = set(inst['relation'] for inst in instances)
        
        # 处理原始样本
        for inst in instances:
            processed.append({
                'tokens': inst['tokens'],
                'h': inst['h'],
                't': inst['t'],
                'relation': inst['relation']
            })
        
        # 添加NOTA样本
        if include_nota:
            # 模拟N-way场景,随机选择N个关系作为support set
            n_way = self.n_way if hasattr(self, 'n_way') else 5
            support_relations = random.sample(list(all_relations), n_way)
            
            num_nota = int(len(processed) * self.pos_neg_ratio)
            nota_samples = self._generate_nota_samples(
                instances,
                support_relations,
                num_nota
            )
            processed.extend(nota_samples)
        
        return processed
        
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        dataset: str = 'fewrel'
    ):
        """处理数据集
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            dataset: 数据集名称
        """
        # 处理训练集、开发集、测试集
        for split in ['train', 'dev', 'test']:
            # 根据数据集类型选择输入文件
            if dataset == 'fewrel':
                input_file = os.path.join(input_dir, f'my_{split}.json')
            else:
                input_file = os.path.join(input_dir, f'{split}_process.json')
                
            output_file = os.path.join(output_dir, f'{split}.json')
            
            self.logger.info(f'Processing {dataset} {split} set...')
            
            # 加载数据
            with open(input_file, 'r') as f:
                instances = json.load(f)
                
            # 处理数据
            processed = self._process_split(
                instances,
                include_nota=(split != 'train')  # 训练集不包含NOTA样本
            )
            
            # 保存处理后的数据
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(processed, f, indent=2)
                
        # 处理关系映射
        if dataset == 'fewrel':
            rel2id_file = os.path.join(input_dir, 'my_rel2id.json')
        else:
            rel2id_file = os.path.join(input_dir, 'rel2id.json')
            
        output_rel2id = os.path.join(output_dir, 'rel2id.json')
        
        with open(rel2id_file, 'r') as f:
            rel2id = json.load(f)
            
        # 添加NOTA关系
        rel2id['NOTA'] = len(rel2id)
        
        with open(output_rel2id, 'w') as f:
            json.dump(rel2id, f, indent=2)

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = DatasetProcessor()
    
    # 处理FewRel数据集
    processor.process_dataset(
        input_dir='CoIn/data/fewrel',
        output_dir='CoIn/data/processed/fewrel',
        dataset='fewrel'
    )
    
    # 处理TACRED数据集
    processor.process_dataset(
        input_dir='CoIn/data/tacred',
        output_dir='CoIn/data/processed/tacred',
        dataset='tacred'
    )

if __name__ == '__main__':
    main() 