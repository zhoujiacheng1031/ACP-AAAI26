# -*- coding: utf-8 -*-
"""
Created on 2024-07-26

@author: zhoujiacheng
@file: dataset_analyzer.py
@purpose: Analyze dataset statistics and quality
"""

import os
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import seaborn as sns

class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, data_dir: str):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # 加载数据
        self.data = self._load_data()
        self.rel2id = self._load_rel2id()
        
    def analyze(self, output_dir: str = None):
        """分析数据集
        
        Args:
            output_dir: 输出目录,用于保存分析结果和图表
        """
        # 1. 基本统计
        basic_stats = self._compute_basic_stats()
        
        # 2. 关系分布
        rel_dist = self._analyze_relation_distribution()
        
        # 3. 实体类型分布
        entity_dist = self._analyze_entity_distribution()
        
        # 4. 句子长度分布
        sent_lens = self._analyze_sentence_lengths()
        
        # 5. 实体距离分布
        entity_dists = self._analyze_entity_distances()
        
        # 6. NOTA样本分析
        nota_stats = self._analyze_nota_samples()
        
        # 保存分析结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_analysis(
                output_dir,
                basic_stats,
                rel_dist,
                entity_dist,
                sent_lens,
                entity_dists,
                nota_stats
            )
            
        return {
            'basic_stats': basic_stats,
            'relation_distribution': rel_dist,
            'entity_distribution': entity_dist,
            'sentence_lengths': sent_lens,
            'entity_distances': entity_dists,
            'nota_statistics': nota_stats
        }
        
    def _compute_basic_stats(self) -> Dict:
        """计算基本统计信息"""
        stats = {
            'total_instances': len(self.data),
            'num_relations': len(self.rel2id),
            'splits': {}
        }
        
        # 按数据集划分统计
        split_counts = defaultdict(int)
        for inst in self.data:
            split = inst.get('split', 'unknown')
            split_counts[split] += 1
            
        stats['splits'] = dict(split_counts)
        
        return stats
        
    def _analyze_relation_distribution(self) -> Dict:
        """分析关系分布"""
        rel_counts = Counter(inst['relation'] for inst in self.data)
        
        return {
            'counts': dict(rel_counts),
            'min_instances': min(rel_counts.values()),
            'max_instances': max(rel_counts.values()),
            'avg_instances': np.mean(list(rel_counts.values()))
        }
        
    def _analyze_entity_distribution(self) -> Dict:
        """分析实体分布"""
        head_types = Counter(inst['h'].get('type', 'unknown') for inst in self.data)
        tail_types = Counter(inst['t'].get('type', 'unknown') for inst in self.data)
        
        return {
            'head_types': dict(head_types),
            'tail_types': dict(tail_types)
        }
        
    def _analyze_sentence_lengths(self) -> Dict:
        """分析句子长度分布"""
        lengths = [len(inst['tokens']) for inst in self.data]
        
        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'distribution': np.histogram(lengths, bins=20)
        }
        
    def _analyze_entity_distances(self) -> Dict:
        """分析实体间距离分布"""
        distances = []
        for inst in self.data:
            h_start = inst['h']['pos'][0]
            t_start = inst['t']['pos'][0]
            distances.append(abs(h_start - t_start))
            
        return {
            'min_distance': min(distances),
            'max_distance': max(distances),
            'avg_distance': np.mean(distances),
            'distribution': np.histogram(distances, bins=20)
        }
        
    def _analyze_nota_samples(self) -> Dict:
        """分析NOTA样本"""
        nota_count = sum(1 for inst in self.data if inst['relation'] == 'NOTA')
        
        return {
            'total_nota': nota_count,
            'nota_ratio': nota_count / len(self.data)
        }
        
    def _save_analysis(
        self,
        output_dir: str,
        basic_stats: Dict,
        rel_dist: Dict,
        entity_dist: Dict,
        sent_lens: Dict,
        entity_dists: Dict,
        nota_stats: Dict
    ):
        """保存分析结果"""
        # 保存统计数据
        stats = {
            'basic_statistics': basic_stats,
            'relation_distribution': rel_dist,
            'entity_distribution': entity_dist,
            'sentence_lengths': {
                k: v for k, v in sent_lens.items()
                if k != 'distribution'
            },
            'entity_distances': {
                k: v for k, v in entity_dists.items()
                if k != 'distribution'
            },
            'nota_statistics': nota_stats
        }
        
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # 绘制关系分布图
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(rel_dist['counts'].keys()),
            y=list(rel_dist['counts'].values())
        )
        plt.xticks(rotation=45)
        plt.title('Relation Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relation_dist.png'))
        plt.close()
        
        # 绘制句子长度分布图
        plt.figure(figsize=(10, 6))
        plt.hist(sent_lens['distribution'][0], bins=sent_lens['distribution'][1])
        plt.title('Sentence Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'sentence_lengths.png'))
        plt.close()
        
        # 绘制实体距离分布图
        plt.figure(figsize=(10, 6))
        plt.hist(
            entity_dists['distribution'][0],
            bins=entity_dists['distribution'][1]
        )
        plt.title('Entity Distance Distribution')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'entity_distances.png'))
        plt.close()
        
    def _load_data(self) -> List[Dict]:
        """加载数据集"""
        data = []
        for split in ['train', 'dev', 'test']:
            file_path = os.path.join(self.data_dir, f'{split}.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    split_data = json.load(f)
                    for inst in split_data:
                        inst['split'] = split
                    data.extend(split_data)
        return data
        
    def _load_rel2id(self) -> Dict[str, int]:
        """加载关系ID映射"""
        file_path = os.path.join(self.data_dir, 'rel2id.json')
        with open(file_path, 'r') as f:
            return json.load(f)

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 分析FewRel数据集
    analyzer = DatasetAnalyzer('CoIn/data/processed/fewrel')
    analyzer.analyze('CoIn/data/analysis/fewrel')
    
    # 分析TACRED数据集
    analyzer = DatasetAnalyzer('CoIn/data/processed/tacred')
    analyzer.analyze('CoIn/data/analysis/tacred')

if __name__ == '__main__':
    main() 