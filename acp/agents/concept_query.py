# -*- coding: utf-8 -*-
"""
Concept query node for LangGraph agent system
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import torch

from .state import AgentState, ConceptInfo, StateManager, StateValidator
from .concept_retriever import EnhancedConceptRetriever
from ..config.config import ConceptGraphConfig


class ConceptQueryNode:
    """概念查询节点，负责多跳概念查询和状态更新"""
    
    def __init__(
        self,
        concept_retriever: EnhancedConceptRetriever,
        config: ConceptGraphConfig
    ):
        self.concept_retriever = concept_retriever
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def process(self, state: AgentState) -> AgentState:
        """处理概念查询"""
        
        # 验证输入状态
        if not StateValidator.validate_input_state(state):
            error_msg = "输入状态验证失败"
            self.logger.error(error_msg)
            return StateManager.update_state(state, {
                'error_info': {'type': 'validation_error', 'message': error_msg}
            })
        
        start_time = time.time()
        
        try:
            # 提取实体列表
            head_entities = state['head_entities']
            tail_entities = state['tail_entities']
            
            self.logger.info(f"开始概念查询 - 头实体: {len(head_entities)}, 尾实体: {len(tail_entities)}")
            
            # 批量查询头实体概念
            head_concepts_dict = await self.concept_retriever.batch_get_concepts(
                entities=head_entities,
                hop=self.config.max_hop_distance,
                topk=self.config.max_concepts_per_entity,
                score_threshold=self.config.concept_score_threshold
            )
            
            # 批量查询尾实体概念
            tail_concepts_dict = await self.concept_retriever.batch_get_concepts(
                entities=tail_entities,
                hop=self.config.max_hop_distance,
                topk=self.config.max_concepts_per_entity,
                score_threshold=self.config.concept_score_threshold
            )
            
            # 组织概念数据
            head_concepts = []
            tail_concepts = []
            
            for entity in head_entities:
                concepts = head_concepts_dict.get(entity, [])
                head_concepts.append([concept.__dict__ for concept in concepts])
            
            for entity in tail_entities:
                concepts = tail_concepts_dict.get(entity, [])
                tail_concepts.append([concept.__dict__ for concept in concepts])
            
            # 生成概念嵌入张量
            concept_embeddings = self._generate_concept_embeddings(
                head_concepts_dict, tail_concepts_dict
            )
            
            processing_time = time.time() - start_time
            
            # 记录统计信息
            self._log_concept_statistics(head_concepts_dict, tail_concepts_dict, processing_time)
            
            # 更新状态
            updated_state = StateManager.update_state(state, {
                'head_concepts': head_concepts,
                'tail_concepts': tail_concepts,
                'concept_embeddings': concept_embeddings,
                'processing_time': processing_time
            })
            
            self.logger.info(f"概念查询完成 - 处理时间: {processing_time:.2f}s")
            
            return updated_state
            
        except Exception as e:
            error_msg = f"概念查询失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'concept_query_error',
                    'message': error_msg,
                    'processing_time': time.time() - start_time
                }
            })
    
    def _generate_concept_embeddings(
        self,
        head_concepts_dict: Dict[str, List[ConceptInfo]],
        tail_concepts_dict: Dict[str, List[ConceptInfo]]
    ) -> Optional[torch.Tensor]:
        """生成概念嵌入张量"""
        
        try:
            all_concepts = []
            
            # 收集所有概念
            for concepts in head_concepts_dict.values():
                all_concepts.extend(concepts)
            
            for concepts in tail_concepts_dict.values():
                all_concepts.extend(concepts)
            
            if not all_concepts:
                return None
            
            # 生成嵌入张量
            embeddings = self.concept_retriever.get_concept_embeddings_tensor(all_concepts)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成概念嵌入失败: {e}")
            return None
    
    def _log_concept_statistics(
        self,
        head_concepts_dict: Dict[str, List[ConceptInfo]],
        tail_concepts_dict: Dict[str, List[ConceptInfo]],
        processing_time: float
    ):
        """记录概念统计信息"""
        
        # 统计头实体概念
        head_stats = self._calculate_concept_stats(head_concepts_dict)
        tail_stats = self._calculate_concept_stats(tail_concepts_dict)
        
        self.logger.info(
            f"概念查询统计 - "
            f"头实体概念: {head_stats['total_concepts']}, "
            f"尾实体概念: {tail_stats['total_concepts']}, "
            f"平均相关性: {(head_stats['avg_relevance'] + tail_stats['avg_relevance']) / 2:.3f}, "
            f"处理时间: {processing_time:.2f}s"
        )
        
        # 详细统计（调试模式）
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"头实体概念分布: {head_stats['hop_distribution']}")
            self.logger.debug(f"尾实体概念分布: {tail_stats['hop_distribution']}")
    
    def _calculate_concept_stats(
        self,
        concepts_dict: Dict[str, List[ConceptInfo]]
    ) -> Dict[str, Any]:
        """计算概念统计信息"""
        
        all_concepts = []
        for concepts in concepts_dict.values():
            all_concepts.extend(concepts)
        
        if not all_concepts:
            return {
                'total_concepts': 0,
                'avg_relevance': 0.0,
                'hop_distribution': {}
            }
        
        # 跳数分布
        hop_distribution = {}
        relevance_scores = []
        
        for concept in all_concepts:
            hop = concept.hop_distance
            hop_distribution[hop] = hop_distribution.get(hop, 0) + 1
            relevance_scores.append(concept.relevance_score)
        
        return {
            'total_concepts': len(all_concepts),
            'avg_relevance': sum(relevance_scores) / len(relevance_scores),
            'hop_distribution': hop_distribution
        }


class MultiHopConceptQuery:
    """多跳概念查询的便利类"""
    
    def __init__(self, config: ConceptGraphConfig):
        self.config = config
        self.concept_retriever = None
        self.query_node = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """异步初始化"""
        self.concept_retriever = EnhancedConceptRetriever(self.config)
        self.query_node = ConceptQueryNode(self.concept_retriever, self.config)
        self.logger.info("多跳概念查询初始化完成")
    
    async def query_concepts(
        self,
        instances: List[Dict[str, Any]],
        n_way: int,
        k_shot: int
    ) -> AgentState:
        """查询概念的便利方法"""
        
        if not self.query_node:
            await self.initialize()
        
        # 创建初始状态
        initial_state = StateManager.create_initial_state(instances, n_way, k_shot)
        
        # 执行概念查询
        result_state = await self.query_node.process(initial_state)
        
        return result_state
    
    async def query_single_entity_concepts(
        self,
        entity: str,
        hop: Optional[int] = None,
        topk: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[ConceptInfo]:
        """查询单个实体的概念"""
        
        if not self.concept_retriever:
            await self.initialize()
        
        concepts = await self.concept_retriever.get_multi_hop_concepts(
            entity=entity,
            hop=hop,
            topk=topk,
            score_threshold=score_threshold
        )
        
        return concepts
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.concept_retriever:
            return self.concept_retriever.get_cache_stats()
        return {}
    
    def clear_cache(self):
        """清除缓存"""
        if self.concept_retriever:
            self.concept_retriever.clear_cache()


# LangGraph节点函数
async def multi_hop_concept_query(state: AgentState) -> AgentState:
    """LangGraph节点函数：多跳概念查询"""
    
    # 从状态中获取配置（这里使用默认配置）
    config = ConceptGraphConfig()
    
    # 创建概念检索器和查询节点
    concept_retriever = EnhancedConceptRetriever(config)
    query_node = ConceptQueryNode(concept_retriever, config)
    
    # 执行查询
    result_state = await query_node.process(state)
    
    return result_state


# 工厂函数
def create_concept_query_node(config: ConceptGraphConfig) -> ConceptQueryNode:
    """创建概念查询节点的工厂函数"""
    concept_retriever = EnhancedConceptRetriever(config)
    return ConceptQueryNode(concept_retriever, config)


async def create_async_concept_query_node(config: ConceptGraphConfig) -> ConceptQueryNode:
    """异步创建概念查询节点"""
    concept_retriever = EnhancedConceptRetriever(config)
    query_node = ConceptQueryNode(concept_retriever, config)
    
    # 可以在这里进行异步初始化
    # await query_node.initialize()  # 如果需要的话
    
    return query_node