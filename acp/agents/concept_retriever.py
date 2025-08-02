# -*- coding: utf-8 -*-
"""
Enhanced concept retriever for LangGraph agent system
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from py2neo import Graph
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import json
import time
from ..config.config import ConceptGraphConfig
from .state import ConceptInfo, ProcessingResult


@dataclass
class ConceptQueryResult:
    """概念查询结果"""
    entity: str
    concepts: List[ConceptInfo]
    query_time: float
    hop_distribution: Dict[int, int]  # 每个跳数的概念数量
    total_concepts_found: int


class ConceptCache:
    """概念缓存管理器"""
    
    def __init__(self, config: ConceptGraphConfig):
        self.config = config
        self.concept_cache = {}
        self.embedding_cache = {}
        self.access_times = {}
        self.max_size = config.cache_size
        self.ttl = config.cache_ttl
        
    def get_concepts(self, key: str) -> Optional[List[ConceptInfo]]:
        """获取缓存的概念"""
        if not self.config.enable_cache:
            return None
            
        if key in self.concept_cache:
            # 检查TTL
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()
                return self.concept_cache[key]
            else:
                # 过期，删除
                del self.concept_cache[key]
                del self.access_times[key]
        
        return None
    
    def set_concepts(self, key: str, concepts: List[ConceptInfo]):
        """设置概念缓存"""
        if not self.config.enable_cache:
            return
            
        # 检查缓存大小
        if len(self.concept_cache) >= self.max_size:
            self._evict_oldest()
        
        self.concept_cache[key] = concepts
        self.access_times[key] = time.time()
    
    def get_embedding(self, concept_name: str) -> Optional[torch.Tensor]:
        """获取缓存的嵌入"""
        if not self.config.enable_cache:
            return None
            
        return self.embedding_cache.get(concept_name)
    
    def set_embedding(self, concept_name: str, embedding: torch.Tensor):
        """设置嵌入缓存"""
        if not self.config.enable_cache:
            return
            
        self.embedding_cache[concept_name] = embedding
    
    def _evict_oldest(self):
        """驱逐最旧的缓存项"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.concept_cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """清除所有缓存"""
        self.concept_cache.clear()
        self.embedding_cache.clear()
        self.access_times.clear()


class EnhancedConceptRetriever:
    """增强的概念检索器，支持LangGraph和异步处理"""
    
    def __init__(
        self,
        config: ConceptGraphConfig,
        model_name: str = "bert-base-uncased"
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化缓存
        self.cache = ConceptCache(config)
        
        # 初始化BERT模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder.eval()
        except Exception as e:
            self.logger.error(f"加载BERT模型失败: {e}")
            self.tokenizer = None
            self.encoder = None
        
        # 连接Neo4j
        self._init_neo4j_connection()
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.encoder:
            self.encoder.to(self.device)
    
    def _init_neo4j_connection(self):
        """初始化Neo4j连接"""
        try:
            load_dotenv()
            self.graph = Graph(
                os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                auth=(
                    os.getenv('NEO4J_USER', 'neo4j'),
                    os.getenv('NEO4J_PASSWORD', 'password')
                )
            )
            # 测试连接
            self.graph.run("RETURN 1").data()
            self.logger.info("Neo4j连接成功")
        except Exception as e:
            self.logger.error(f"Neo4j连接失败: {e}")
            self.graph = None
    
    async def get_multi_hop_concepts(
        self,
        entity: str,
        hop: Optional[int] = None,
        topk: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[ConceptInfo]:
        """异步获取实体的多跳概念"""
        
        # 使用配置默认值
        hop = hop or self.config.max_hop_distance
        topk = topk or self.config.max_concepts_per_entity
        score_threshold = score_threshold or self.config.concept_score_threshold
        
        # 检查缓存
        cache_key = f"{entity}_{hop}_{topk}_{score_threshold}"
        cached_concepts = self.cache.get_concepts(cache_key)
        if cached_concepts:
            return cached_concepts
        
        # 查询概念图
        concepts = await self._query_concept_graph(entity, hop)
        
        # 计算嵌入和分数
        enhanced_concepts = await self._enhance_concepts(concepts, entity)
        
        # 过滤和排序
        filtered_concepts = self._filter_and_rank_concepts(
            enhanced_concepts, topk, score_threshold
        )
        
        # 更新缓存
        self.cache.set_concepts(cache_key, filtered_concepts)
        
        return filtered_concepts
    
    async def batch_get_concepts(
        self,
        entities: List[str],
        hop: Optional[int] = None,
        topk: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_concurrent: int = 10
    ) -> Dict[str, List[ConceptInfo]]:
        """批量异步获取概念"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_entity(entity: str) -> Tuple[str, List[ConceptInfo]]:
            async with semaphore:
                concepts = await self.get_multi_hop_concepts(
                    entity, hop, topk, score_threshold
                )
                return entity, concepts
        
        tasks = [process_entity(entity) for entity in entities]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        concept_dict = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"批量概念查询失败: {result}")
                continue
            entity, concepts = result
            concept_dict[entity] = concepts
        
        return concept_dict
    
    async def _query_concept_graph(
        self,
        entity: str,
        hop: int
    ) -> List[Dict[str, Any]]:
        """查询概念图获取原始概念数据"""
        
        if not self.graph:
            self.logger.warning("Neo4j连接不可用，返回空概念列表")
            return []
        
        try:
            # 构建Cypher查询
            query = """
            MATCH (e:Entity {name: $entity})-[r:IS_A*1..$hop]->(c:Concept)
            RETURN c.name as name, 
                   c.definition as definition,
                   c.embedding as embedding,
                   length(r) as hops,
                   r[-1].weight as weight
            ORDER BY length(r), r[-1].weight DESC
            """
            
            # 执行查询
            results = self.graph.run(
                query,
                entity=entity,
                hop=hop
            ).data()
            
            return results
            
        except Exception as e:
            self.logger.error(f"概念图查询失败: {e}")
            return []
    
    async def _enhance_concepts(
        self,
        raw_concepts: List[Dict[str, Any]],
        entity: str
    ) -> List[ConceptInfo]:
        """增强概念信息，计算嵌入和相关性分数"""
        
        enhanced_concepts = []
        
        for concept_data in raw_concepts:
            concept_name = concept_data.get('name', '')
            definition = concept_data.get('definition', '')
            hops = concept_data.get('hops', 1)
            weight = concept_data.get('weight', 1.0)
            
            # 获取或计算嵌入
            embedding = await self._get_concept_embedding(concept_name, definition)
            
            # 计算相关性分数
            relevance_score = self._calculate_relevance_score(
                concept_name, entity, hops, weight
            )
            
            # 创建ConceptInfo对象
            concept_info = ConceptInfo(
                id=f"{entity}_{concept_name}_{hops}",
                name=concept_name,
                definition=definition,
                embedding=embedding,
                hop_distance=hops,
                relevance_score=relevance_score,
                source="neo4j_concept_graph",
                metadata={
                    'weight': weight,
                    'entity': entity,
                    'query_time': time.time()
                }
            )
            
            enhanced_concepts.append(concept_info)
        
        return enhanced_concepts
    
    async def _get_concept_embedding(
        self,
        concept_name: str,
        definition: str = ""
    ) -> Optional[torch.Tensor]:
        """获取概念嵌入"""
        
        # 检查缓存
        cached_embedding = self.cache.get_embedding(concept_name)
        if cached_embedding is not None:
            return cached_embedding
        
        if not self.encoder or not self.tokenizer:
            return None
        
        try:
            # 使用概念名称和定义生成嵌入
            text = f"{concept_name}: {definition}" if definition else concept_name
            
            # 分词
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # 使用[CLS]token的输出
                embedding = outputs.last_hidden_state[:, 0, :].cpu()
            
            # 更新缓存
            self.cache.set_embedding(concept_name, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"生成概念嵌入失败: {e}")
            return None
    
    def _calculate_relevance_score(
        self,
        concept_name: str,
        entity: str,
        hops: int,
        weight: float
    ) -> float:
        """计算概念相关性分数"""
        
        # 基础分数基于权重
        base_score = float(weight) if weight else 0.5
        
        # 跳数惩罚
        hop_penalty = 1.0 / (1.0 + hops * 0.3)
        
        # 名称相似性奖励
        name_similarity = self._calculate_name_similarity(concept_name, entity)
        
        # 综合分数
        relevance_score = base_score * hop_penalty + name_similarity * 0.2
        
        return min(max(relevance_score, 0.0), 1.0)
    
    def _calculate_name_similarity(self, concept_name: str, entity: str) -> float:
        """计算名称相似性"""
        concept_words = set(concept_name.lower().split())
        entity_words = set(entity.lower().split())
        
        if not concept_words or not entity_words:
            return 0.0
        
        intersection = concept_words.intersection(entity_words)
        union = concept_words.union(entity_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _filter_and_rank_concepts(
        self,
        concepts: List[ConceptInfo],
        topk: int,
        score_threshold: float
    ) -> List[ConceptInfo]:
        """过滤和排序概念"""
        
        # 按分数过滤
        filtered = [c for c in concepts if c.relevance_score >= score_threshold]
        
        # 按相关性分数排序
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 取前k个
        return filtered[:topk]
    
    def get_concept_embeddings_tensor(
        self,
        concepts: List[ConceptInfo]
    ) -> torch.Tensor:
        """获取概念嵌入张量"""
        
        if not concepts:
            # 返回零张量
            embedding_dim = 768  # BERT默认维度
            return torch.zeros(1, embedding_dim, dtype=torch.float32)
        
        embeddings = []
        for concept in concepts:
            if concept.embedding is not None:
                embeddings.append(concept.embedding.squeeze())
            else:
                # 使用零向量填充
                embedding_dim = 768
                embeddings.append(torch.zeros(embedding_dim))
        
        return torch.stack(embeddings).float()
    
    def compute_concept_similarity(
        self,
        concept1: ConceptInfo,
        concept2: ConceptInfo
    ) -> float:
        """计算两个概念的相似度"""
        
        if concept1.embedding is None or concept2.embedding is None:
            return 0.0
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            concept1.embedding.flatten().unsqueeze(0),
            concept2.embedding.flatten().unsqueeze(0)
        )
        
        return similarity.item()
    
    def get_concept_statistics(
        self,
        concepts: List[ConceptInfo]
    ) -> Dict[str, Any]:
        """获取概念统计信息"""
        
        if not concepts:
            return {
                'total_concepts': 0,
                'hop_distribution': {},
                'avg_relevance_score': 0.0,
                'score_distribution': {}
            }
        
        # 跳数分布
        hop_distribution = {}
        for concept in concepts:
            hop = concept.hop_distance
            hop_distribution[hop] = hop_distribution.get(hop, 0) + 1
        
        # 分数分布
        scores = [c.relevance_score for c in concepts]
        score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        score_distribution = {}
        
        for low, high in score_ranges:
            count = sum(1 for score in scores if low <= score < high)
            score_distribution[f"{low}-{high}"] = count
        
        return {
            'total_concepts': len(concepts),
            'hop_distribution': hop_distribution,
            'avg_relevance_score': sum(scores) / len(scores),
            'score_distribution': score_distribution,
            'max_score': max(scores),
            'min_score': min(scores)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'concept_cache_size': len(self.cache.concept_cache),
            'embedding_cache_size': len(self.cache.embedding_cache),
            'cache_enabled': self.config.enable_cache,
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率（简化实现）"""
        # 这里可以实现更复杂的命中率统计
        return 0.0


# 便利函数
async def create_concept_retriever(config: ConceptGraphConfig) -> EnhancedConceptRetriever:
    """创建概念检索器的便利函数"""
    return EnhancedConceptRetriever(config)