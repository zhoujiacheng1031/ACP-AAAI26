# -*- coding: utf-8 -*-
"""
Enhanced data loader for LangGraph agent system with NOTA support
"""

import json
import random
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

from ..agents.state import AgentState, StateManager
from ..agents.graph import AgentGraph
from ..config.config import AgentConfig


class FewShotRelationDataset(Dataset):
    """小样本关系分类数据集，支持NOTA检测和智能体增强"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        agent_graph: Optional[AgentGraph] = None,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 5,
        max_length: int = 128,
        nota_ratio: float = 0.3,
        mode: str = 'train',
        use_agent_enhancement: bool = True,
        cache_agent_results: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.agent_graph = agent_graph
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.max_length = max_length
        self.nota_ratio = nota_ratio
        self.mode = mode
        self.use_agent_enhancement = use_agent_enhancement
        self.cache_agent_results = cache_agent_results
        
        # 加载数据
        self.data = self._load_data(data_path)
        self.rel2id = self._load_rel2id(data_path)
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        
        # 按关系分组
        self.rel2instances = self._group_by_relation()
        
        # 生成任务
        self.episodes = self._generate_episodes()
        
        # 智能体结果缓存
        self.agent_cache = {} if cache_agent_results else None
        
        self.logger.info(
            f"数据集初始化完成 - 关系数: {len(self.rel2instances)}, "
            f"任务数: {len(self.episodes)}, "
            f"智能体增强: {use_agent_enhancement}"
        )
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取一个小样本学习任务"""
        
        episode = self.episodes[idx]
        
        try:
            # 处理支持集
            support_data = self._process_instances(
                episode['support_set'], 
                episode['relations'],
                is_support=True
            )
            
            # 处理查询集（包含NOTA样本）
            query_data = self._process_instances(
                episode['query_set'], 
                episode['relations'],
                is_support=False
            )
            
            # 智能体增强（如果启用）
            if self.use_agent_enhancement and self.agent_graph:
                agent_results = self._get_agent_enhancement(
                    episode['support_set'] + episode['query_set']
                )
            else:
                agent_results = None
            
            return {
                'support_data': support_data,
                'query_data': query_data,
                'relations': episode['relations'],
                'n_way': self.n_way,
                'k_shot': self.k_shot,
                'agent_results': agent_results,
                'episode_id': idx
            }
            
        except Exception as e:
            self.logger.error(f"处理任务{idx}失败: {e}")
            # 返回空任务
            return self._create_empty_task(idx)
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    # JSONL格式
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                    return data
                else:
                    # JSON格式
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return []
    
    def _load_rel2id(self, data_path: str) -> Dict[str, int]:
        """加载关系ID映射"""
        try:
            # 尝试从同目录加载rel2id.json
            import os
            data_dir = os.path.dirname(data_path)
            rel2id_path = os.path.join(data_dir, 'rel2id.json')
            
            with open(rel2id_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载关系映射失败: {e}")
            # 从数据中自动生成
            return self._generate_rel2id()
    
    def _generate_rel2id(self) -> Dict[str, int]:
        """从数据中生成关系ID映射"""
        relations = set()
        for instance in self.data:
            relations.add(instance.get('relation', 'no_relation'))
        
        return {rel: idx for idx, rel in enumerate(sorted(relations))}
    
    def _group_by_relation(self) -> Dict[str, List[Dict[str, Any]]]:
        """按关系分组实例"""
        rel2instances = {}
        
        for instance in self.data:
            rel = instance.get('relation', 'no_relation')
            if rel not in rel2instances:
                rel2instances[rel] = []
            rel2instances[rel].append(instance)
        
        # 过滤掉样本数量不足的关系
        min_samples = self.k_shot * 2 + self.q_query
        filtered_rel2instances = {
            rel: instances for rel, instances in rel2instances.items()
            if len(instances) >= min_samples
        }
        
        self.logger.info(
            f"关系过滤: {len(rel2instances)} -> {len(filtered_rel2instances)}"
        )
        
        return filtered_rel2instances
    
    def _generate_episodes(self) -> List[Dict[str, Any]]:
        """生成小样本学习任务"""
        episodes = []
        relations = list(self.rel2instances.keys())
        
        if len(relations) < self.n_way:
            self.logger.warning(
                f"可用关系数({len(relations)})少于n_way({self.n_way})"
            )
            return []
        
        # 训练模式生成更多任务
        num_episodes = 1000 if self.mode == 'train' else 200
        
        for episode_id in range(num_episodes):
            try:
                # 随机选择N个关系
                selected_rels = random.sample(relations, self.n_way)
                
                support_set = []
                query_set = []
                
                # 为每个关系选择样本
                for rel in selected_rels:
                    instances = self.rel2instances[rel]
                    
                    # 确保有足够的样本
                    required_samples = self.k_shot + self.q_query
                    if len(instances) < required_samples:
                        continue
                    
                    # 随机选择样本
                    selected = random.sample(instances, required_samples)
                    support_set.extend(selected[:self.k_shot])
                    query_set.extend(selected[self.k_shot:])
                
                # 添加NOTA样本到查询集
                if self.nota_ratio > 0:
                    nota_samples = self._generate_nota_samples(
                        selected_rels, len(query_set)
                    )
                    query_set.extend(nota_samples)
                
                if support_set and query_set:
                    episodes.append({
                        'support_set': support_set,
                        'query_set': query_set,
                        'relations': selected_rels,
                        'episode_id': episode_id
                    })
                    
            except Exception as e:
                self.logger.warning(f"生成任务{episode_id}失败: {e}")
                continue
        
        self.logger.info(f"生成了{len(episodes)}个任务")
        return episodes
    
    def _generate_nota_samples(
        self, 
        selected_rels: List[str], 
        query_size: int
    ) -> List[Dict[str, Any]]:
        """生成NOTA样本"""
        
        # 计算NOTA样本数量
        num_nota = int(query_size * self.nota_ratio)
        if num_nota == 0:
            return []
        
        # 选择不在当前任务中的关系
        other_rels = [
            rel for rel in self.rel2instances.keys() 
            if rel not in selected_rels
        ]
        
        if not other_rels:
            return []
        
        nota_samples = []
        for _ in range(num_nota):
            try:
                # 随机选择一个其他关系
                nota_rel = random.choice(other_rels)
                nota_instances = self.rel2instances[nota_rel]
                
                # 随机选择一个实例
                nota_instance = random.choice(nota_instances)
                
                # 标记为NOTA
                nota_sample = nota_instance.copy()
                nota_sample['relation'] = 'NOTA'
                nota_sample['original_relation'] = nota_rel
                
                nota_samples.append(nota_sample)
                
            except Exception as e:
                self.logger.warning(f"生成NOTA样本失败: {e}")
                continue
        
        return nota_samples
    
    def _process_instances(
        self, 
        instances: List[Dict[str, Any]], 
        relations: List[str],
        is_support: bool = True
    ) -> Dict[str, torch.Tensor]:
        """处理实例列表"""
        
        if not instances:
            return self._create_empty_batch()
        
        # 提取文本和标签
        texts = []
        labels = []
        entity_positions = []
        
        # 创建标签映射（包含NOTA）
        label_map = {rel: idx for idx, rel in enumerate(relations)}
        label_map['NOTA'] = len(relations)  # NOTA标签
        
        for instance in instances:
            # 构建文本
            tokens = instance.get('token', instance.get('tokens', []))
            text = ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
            texts.append(text)
            
            # 获取标签
            rel = instance.get('relation', 'NOTA')
            label = label_map.get(rel, label_map['NOTA'])
            labels.append(label)
            
            # 获取实体位置
            h_pos = instance.get('h', {}).get('pos', [0, 0])
            t_pos = instance.get('t', {}).get('pos', [0, 0])
            entity_positions.append([h_pos[0], h_pos[-1], t_pos[0], t_pos[-1]])
        
        # 分词
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 调整实体位置
        adjusted_positions = self._adjust_entity_positions(
            texts, entity_positions, tokenized
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'entity_positions': adjusted_positions,
            'num_instances': len(instances)
        }
    
    def _adjust_entity_positions(
        self,
        texts: List[str],
        positions: List[List[int]],
        tokenized
    ) -> torch.Tensor:
        """调整实体位置以对齐分词结果"""
        
        batch_size = len(texts)
        adjusted_positions = torch.zeros((batch_size, 4), dtype=torch.long)
        
        for i, (text, pos) in enumerate(zip(texts, positions)):
            try:
                # 简化处理：直接使用原始位置
                # 在实际应用中，这里应该进行更精确的位置对齐
                adjusted_positions[i] = torch.tensor(pos, dtype=torch.long)
            except Exception as e:
                self.logger.warning(f"调整实体位置失败: {e}")
                # 使用默认位置
                adjusted_positions[i] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        
        return adjusted_positions
    
    def _get_agent_enhancement(
        self, 
        instances: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """获取智能体增强结果"""
        
        if not self.agent_graph:
            return None
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(instances)
            
            # 检查缓存
            if self.agent_cache and cache_key in self.agent_cache:
                return self.agent_cache[cache_key]
            
            # 异步执行智能体处理
            # 注意：这里需要在异步上下文中调用
            # 在实际使用中，可能需要预先处理或使用同步版本
            agent_results = None  # 暂时返回None，避免阻塞
            
            # 缓存结果
            if self.agent_cache:
                self.agent_cache[cache_key] = agent_results
            
            return agent_results
            
        except Exception as e:
            self.logger.error(f"智能体增强失败: {e}")
            return None
    
    def _generate_cache_key(self, instances: List[Dict[str, Any]]) -> str:
        """生成缓存键"""
        # 使用实例的哈希值作为缓存键
        instance_strs = []
        for instance in instances:
            tokens = instance.get('token', instance.get('tokens', []))
            h_name = instance.get('h', {}).get('name', '')
            t_name = instance.get('t', {}).get('name', '')
            instance_str = f"{' '.join(tokens)}_{h_name}_{t_name}"
            instance_strs.append(instance_str)
        
        return hash('|'.join(instance_strs))
    
    def _create_empty_batch(self) -> Dict[str, torch.Tensor]:
        """创建空批次"""
        return {
            'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long),
            'labels': torch.zeros(1, dtype=torch.long),
            'entity_positions': torch.zeros((1, 4), dtype=torch.long),
            'num_instances': 0
        }
    
    def _create_empty_task(self, idx: int) -> Dict[str, Any]:
        """创建空任务"""
        empty_batch = self._create_empty_batch()
        return {
            'support_data': empty_batch,
            'query_data': empty_batch,
            'relations': [],
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'agent_results': None,
            'episode_id': idx
        }


class AgentEnhancedDataLoader:
    """智能体增强数据加载器"""
    
    def __init__(
        self,
        dataset: FewShotRelationDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        agent_batch_size: int = 8
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.agent_batch_size = agent_batch_size
        self.logger = logging.getLogger(__name__)
        
        # 创建标准数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def __iter__(self):
        """迭代器"""
        for batch in self.dataloader:
            # 如果启用智能体增强，进行批量处理
            if self.dataset.use_agent_enhancement and self.dataset.agent_graph:
                batch = self._enhance_batch_with_agents(batch)
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批处理函数"""
        
        # 分离支持集和查询集
        support_batches = []
        query_batches = []
        relations_list = []
        agent_results_list = []
        
        for item in batch:
            support_batches.append(item['support_data'])
            query_batches.append(item['query_data'])
            relations_list.append(item['relations'])
            agent_results_list.append(item['agent_results'])
        
        # 合并支持集
        support_batch = self._merge_batches(support_batches)
        
        # 合并查询集
        query_batch = self._merge_batches(query_batches)
        
        return {
            'support': support_batch,
            'query': query_batch,
            'relations': relations_list,
            'agent_results': agent_results_list,
            'batch_size': len(batch)
        }
    
    def _merge_batches(
        self, 
        batches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """合并批次"""
        
        if not batches:
            return {}
        
        merged = {}
        for key in batches[0].keys():
            if key == 'num_instances':
                merged[key] = sum(batch[key] for batch in batches)
            else:
                try:
                    merged[key] = torch.stack([batch[key] for batch in batches])
                except Exception as e:
                    self.logger.warning(f"合并键{key}失败: {e}")
                    # 使用第一个批次的值
                    merged[key] = batches[0][key]
        
        return merged
    
    def _enhance_batch_with_agents(
        self, 
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用智能体增强批次"""
        
        try:
            # 这里可以实现批量智能体处理
            # 由于异步限制，暂时跳过实际处理
            self.logger.debug("智能体增强处理（暂时跳过）")
            
            return batch
            
        except Exception as e:
            self.logger.error(f"智能体增强失败: {e}")
            return batch


def create_data_loader(
    data_path: str,
    tokenizer: AutoTokenizer,
    agent_graph: Optional[AgentGraph] = None,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 5,
    batch_size: int = 4,
    max_length: int = 128,
    nota_ratio: float = 0.3,
    mode: str = 'train',
    use_agent_enhancement: bool = True,
    shuffle: bool = None,
    num_workers: int = 0
) -> AgentEnhancedDataLoader:
    """创建数据加载器的便利函数"""
    
    if shuffle is None:
        shuffle = (mode == 'train')
    
    # 创建数据集
    dataset = FewShotRelationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        agent_graph=agent_graph,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        max_length=max_length,
        nota_ratio=nota_ratio,
        mode=mode,
        use_agent_enhancement=use_agent_enhancement
    )
    
    # 创建数据加载器
    dataloader = AgentEnhancedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


# 异步数据处理器
class AsyncAgentDataProcessor:
    """异步智能体数据处理器"""
    
    def __init__(self, agent_graph: AgentGraph):
        self.agent_graph = agent_graph
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(
        self, 
        instances: List[Dict[str, Any]],
        n_way: int,
        k_shot: int
    ) -> Optional[Dict[str, Any]]:
        """异步处理批次数据"""
        
        try:
            # 执行智能体处理
            result = await self.agent_graph.execute(instances, n_way, k_shot)
            
            if result.get('error_info'):
                self.logger.warning(f"智能体处理有错误: {result['error_info']}")
                return None
            
            return {
                'aligned_concepts': result.get('aligned_concepts', []),
                'concept_weights': result.get('concept_weights', []),
                'meta_relations': result.get('meta_relations', []),
                'verified_relations': result.get('verified_relations', [])
            }
            
        except Exception as e:
            self.logger.error(f"异步处理失败: {e}")
            return None
    
    async def preprocess_dataset(
        self,
        dataset: FewShotRelationDataset,
        max_concurrent: int = 4
    ):
        """预处理整个数据集"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_episode(episode_idx: int):
            async with semaphore:
                episode = dataset.episodes[episode_idx]
                instances = episode['support_set'] + episode['query_set']
                
                result = await self.process_batch(
                    instances, dataset.n_way, dataset.k_shot
                )
                
                if result and dataset.agent_cache is not None:
                    cache_key = dataset._generate_cache_key(instances)
                    dataset.agent_cache[cache_key] = result
        
        # 创建任务
        tasks = [
            process_episode(i) for i in range(len(dataset.episodes))
        ]
        
        # 批量执行
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"预处理完成，缓存了{len(dataset.agent_cache)}个结果")


# 使用示例
async def example_usage():
    """使用示例"""
    
    from transformers import AutoTokenizer
    from .graph import create_agent_graph
    from .config import AgentConfig
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 创建智能体图
    config = AgentConfig()
    agent_graph = await create_agent_graph(config)
    
    try:
        # 创建数据加载器
        dataloader = create_data_loader(
            data_path='data/fewrel/train.json',
            tokenizer=tokenizer,
            agent_graph=agent_graph,
            n_way=5,
            k_shot=1,
            batch_size=2,
            use_agent_enhancement=True
        )
        
        # 预处理数据集（可选）
        processor = AsyncAgentDataProcessor(agent_graph)
        await processor.preprocess_dataset(dataloader.dataset)
        
        # 使用数据加载器
        for batch_idx, batch in enumerate(dataloader):
            print(f"批次 {batch_idx}:")
            print(f"  支持集大小: {batch['support']['input_ids'].shape}")
            print(f"  查询集大小: {batch['query']['input_ids'].shape}")
            
            if batch_idx >= 2:  # 只处理前几个批次
                break
                
    finally:
        # 清理资源
        await agent_graph.cleanup()


if __name__ == "__main__":
    asyncio.run(example_usage())