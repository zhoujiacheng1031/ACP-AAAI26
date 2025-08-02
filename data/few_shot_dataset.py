# -*- coding: utf-8 -*-
"""
@purpose: Dataset class for few-shot learning tasks with NOTA
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import logging
from utils.concept_retriever import ConceptRetriever

class FewShotDataset(Dataset):
    """小样本学习数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int = 128,
        use_concepts: bool = True,
        pos_neg_ratio: float = 0.2  # 正负样本比例
    ):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_concepts = use_concepts
        self.pos_neg_ratio = pos_neg_ratio
        
        # 加载数据
        self.tasks = self._load_tasks(data_dir, split)
        self.rel2id = self._load_rel2id(data_dir)
        
        if use_concepts:
            self.concept_retriever = ConceptRetriever()
        
    def __len__(self) -> int:
        return len(self.tasks)
        
    def __getitem__(self, idx: int) -> Dict:
        """获取一个小样本学习任务
        
        Args:
            idx: 任务索引
            
        Returns:
            包含support_set和query_set的任务字典
        """
        task = self.tasks[idx]
        
        # 处理支持集
        support_inputs = self._process_instances(task['support'])
        
        # 处理查询集(包含NOTA样本)
        query_inputs = self._process_instances(task['query'])
        
        # 获取标签映射(添加NOTA标签)
        label_map = {rel: i for i, rel in enumerate(task['relations'])}
        label_map['NOTA'] = len(label_map)  # NOTA标签
        
        # 转换标签
        support_labels = torch.tensor([
            label_map[inst['relation']] 
            for inst in task['support']
        ])
        
        query_labels = torch.tensor([
            label_map.get(inst['relation'], label_map['NOTA'])  # 对NOTA样本使用NOTA标签
            for inst in task['query']
        ])
        
        return {
            'support_inputs': support_inputs,
            'support_labels': support_labels,
            'query_inputs': query_inputs,
            'query_labels': query_labels,
            'relations': task['relations'] + ['NOTA']  # 添加NOTA关系
        }
        
    def _process_instances(
        self,
        instances: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """处理实例列表
        
        Args:
            instances: 实例列表
            
        Returns:
            处理后的输入特征字典
        """
        # 准备输入
        texts = []
        entity_spans = []
        concepts = [] if self.use_concepts else None
        
        for inst in instances:
            # 构建输入文本
            text = ' '.join(inst['tokens'])
            texts.append(text)
            
            # 获取实体位置
            h_start, h_end = inst['h']['pos']
            t_start, t_end = inst['t']['pos']
            entity_spans.append(((h_start, h_end), (t_start, t_end)))
            
            # 获取概念信息
            if self.use_concepts:
                h_concepts = self.concept_retriever.get_concepts(inst['h']['name'])
                t_concepts = self.concept_retriever.get_concepts(inst['t']['name'])
                concepts.append((h_concepts, t_concepts))
                
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 调整实体位置
        entity_positions = self._align_entity_positions(
            texts,
            entity_spans,
            tokenized
        )
        
        outputs = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'entity_positions': entity_positions
        }
        
        # 添加概念特征
        if self.use_concepts:
            concept_features = self._get_concept_features(concepts)
            outputs['concept_features'] = concept_features
            
        return outputs
        
    def _align_entity_positions(
        self,
        texts: List[str],
        spans: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        tokenized
    ) -> torch.Tensor:
        """调整实体位置以对齐分词结果"""
        batch_size = len(texts)
        positions = torch.zeros((batch_size, 4), dtype=torch.long)
        
        for i, (text, (h_span, t_span)) in enumerate(zip(texts, spans)):
            # 获取原始文本中的字符偏移
            char_to_token = tokenized.char_to_token(i)
            
            # 调整头实体位置
            h_start = char_to_token[h_span[0]]
            h_end = char_to_token[h_span[1]]
            
            # 调整尾实体位置
            t_start = char_to_token[t_span[0]]
            t_end = char_to_token[t_span[1]]
            
            positions[i] = torch.tensor([h_start, h_end, t_start, t_end])
            
        return positions
        
    def _get_concept_features(
        self,
        concepts: List[Tuple[List[str], List[str]]]
    ) -> torch.Tensor:
        """获取概念特征"""
        batch_size = len(concepts)
        features = torch.zeros((batch_size, 2, self.concept_dim))
        
        for i, (h_concepts, t_concepts) in enumerate(concepts):
            # 获取头实体概念嵌入
            h_embeds = self.concept_retriever.get_embeddings(h_concepts)
            features[i, 0] = torch.mean(h_embeds, dim=0)
            
            # 获取尾实体概念嵌入
            t_embeds = self.concept_retriever.get_embeddings(t_concepts)
            features[i, 1] = torch.mean(t_embeds, dim=0)
            
        return features
        
    def _load_tasks(self, data_dir: str, split: str) -> List[Dict]:
        """加载任务数据"""
        file_path = os.path.join(data_dir, f'{split}.json')
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def _load_rel2id(self, data_dir: str) -> Dict[str, int]:
        """加载关系ID映射"""
        file_path = os.path.join(data_dir, 'rel2id.json')
        with open(file_path, 'r') as f:
            return json.load(f)

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """数据批处理函数"""
    # 合并支持集
    support_inputs = {
        k: torch.stack([b['support_inputs'][k] for b in batch])
        for k in batch[0]['support_inputs']
    }
    support_labels = torch.stack([b['support_labels'] for b in batch])
    
    # 合并查询集
    query_inputs = {
        k: torch.stack([b['query_inputs'][k] for b in batch])
        for k in batch[0]['query_inputs']
    }
    query_labels = torch.stack([b['query_labels'] for b in batch])
    
    return {
        'support_inputs': support_inputs,
        'support_labels': support_labels,
        'query_inputs': query_inputs,
        'query_labels': query_labels,
        'relations': [b['relations'] for b in batch]
    } 