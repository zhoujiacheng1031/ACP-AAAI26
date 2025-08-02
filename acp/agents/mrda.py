# -*- coding: utf-8 -*-
"""
Meta-Relation Discovery Agent (MRDA) main implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import torch

from .state import AgentState, ConceptInfo, MetaRelation, StateManager, StateValidator
from ..utils.llm_service import BaseLLMService
from ..config.config import MRDAConfig
from .mrda_components import (
    FineGrainedMetaRelationMiner,
    SemanticConsistencyVerifier,
    ContrastiveNegativeSampler
)


class MetaRelationDiscoveryAgent:
    """元关系发现智能体"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: MRDAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.relation_miner = FineGrainedMetaRelationMiner(llm_service, config)
        self.consistency_verifier = SemanticConsistencyVerifier(llm_service, config)
        self.negative_sampler = ContrastiveNegativeSampler(llm_service, config)
        
        # 统计信息
        self.processing_stats = {
            'total_processed': 0,
            'total_relations_discovered': 0,
            'total_relations_verified': 0,
            'total_negatives_generated': 0,
            'avg_processing_time': 0.0
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """处理MRDA主流程"""
        
        # 验证输入状态
        if not StateValidator.validate_concept_query_state(state):
            error_msg = "MRDA输入状态验证失败"
            self.logger.error(error_msg)
            return StateManager.update_state(state, {
                'error_info': {'type': 'mrda_validation_error', 'message': error_msg}
            })
        
        start_time = time.time()
        
        try:
            self.logger.info("开始MRDA处理流程")
            
            # 提取概念信息
            head_concepts_list = self._extract_concept_info(state['head_concepts'])
            tail_concepts_list = self._extract_concept_info(state['tail_concepts'])
            contexts = state['contexts']
            
            # 验证数据一致性
            if len(head_concepts_list) != len(tail_concepts_list) or len(head_concepts_list) != len(contexts):
                raise ValueError("概念和上下文数据长度不一致")
            
            # 批量处理实例
            all_meta_relations = []
            all_verified_relations = []
            all_negative_samples = []
            all_relation_confidences = []
            
            # 处理每个实例
            for i, (head_concepts, tail_concepts, context) in enumerate(
                zip(head_concepts_list, tail_concepts_list, contexts)
            ):
                self.logger.debug(f"处理实例 {i+1}/{len(contexts)}")
                
                # 1. 元关系挖掘
                meta_relations = await self.relation_miner.mine_relations(
                    head_concepts, tail_concepts, context
                )
                
                # 2. 语义一致性验证
                verified_relations = await self.consistency_verifier.verify_relations(
                    meta_relations,
                    state['head_entities'][i],
                    state['tail_entities'][i],
                    context
                )
                
                # 3. 负样本生成
                negative_samples = await self.negative_sampler.generate_negatives(
                    verified_relations, context
                )
                
                # 收集结果
                all_meta_relations.append([rel.__dict__ for rel in meta_relations])
                all_verified_relations.append([rel.__dict__ for rel in verified_relations])
                all_negative_samples.append([rel.__dict__ for rel in negative_samples])
                
                # 计算置信度
                confidences = [rel.confidence for rel in verified_relations]
                all_relation_confidences.append(confidences)
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_processing_stats(
                len(contexts),
                sum(len(rels) for rels in all_meta_relations),
                sum(len(rels) for rels in all_verified_relations),
                sum(len(rels) for rels in all_negative_samples),
                processing_time
            )
            
            # 记录处理结果
            self._log_processing_results(
                all_meta_relations, all_verified_relations, 
                all_negative_samples, processing_time
            )
            
            # 更新状态
            updated_state = StateManager.update_state(state, {
                'meta_relations': all_meta_relations,
                'verified_relations': all_verified_relations,
                'negative_samples': all_negative_samples,
                'relation_confidences': all_relation_confidences,
                'processing_time': processing_time
            })
            
            self.logger.info(f"MRDA处理完成 - 处理时间: {processing_time:.2f}s")
            
            return updated_state
            
        except Exception as e:
            error_msg = f"MRDA处理失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'mrda_processing_error',
                    'message': error_msg,
                    'processing_time': time.time() - start_time
                }
            })
    
    def _extract_concept_info(
        self, 
        concepts_data: List[List[Dict[str, Any]]]
    ) -> List[List[ConceptInfo]]:
        """从状态数据中提取ConceptInfo对象"""
        
        concept_info_list = []
        
        for instance_concepts in concepts_data:
            concepts = []
            for concept_dict in instance_concepts:
                try:
                    # 重建ConceptInfo对象
                    concept = ConceptInfo(
                        id=concept_dict.get('id', ''),
                        name=concept_dict.get('name', ''),
                        definition=concept_dict.get('definition', ''),
                        embedding=concept_dict.get('embedding'),
                        hop_distance=concept_dict.get('hop_distance', 1),
                        relevance_score=concept_dict.get('relevance_score', 0.0),
                        source=concept_dict.get('source', 'unknown'),
                        metadata=concept_dict.get('metadata')
                    )
                    concepts.append(concept)
                except Exception as e:
                    self.logger.warning(f"重建ConceptInfo失败: {e}")
                    continue
            
            concept_info_list.append(concepts)
        
        return concept_info_list
    
    def _update_processing_stats(
        self,
        num_instances: int,
        num_discovered: int,
        num_verified: int,
        num_negatives: int,
        processing_time: float
    ):
        """更新处理统计信息"""
        
        self.processing_stats['total_processed'] += num_instances
        self.processing_stats['total_relations_discovered'] += num_discovered
        self.processing_stats['total_relations_verified'] += num_verified
        self.processing_stats['total_negatives_generated'] += num_negatives
        
        # 更新平均处理时间
        total_processed = self.processing_stats['total_processed']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_processed - num_instances) + processing_time) / total_processed
        )
    
    def _log_processing_results(
        self,
        meta_relations: List[List[Dict]],
        verified_relations: List[List[Dict]],
        negative_samples: List[List[Dict]],
        processing_time: float
    ):
        """记录处理结果"""
        
        total_meta = sum(len(rels) for rels in meta_relations)
        total_verified = sum(len(rels) for rels in verified_relations)
        total_negatives = sum(len(rels) for rels in negative_samples)
        
        verification_rate = (total_verified / total_meta * 100) if total_meta > 0 else 0
        
        self.logger.info(
            f"MRDA处理统计 - "
            f"发现关系: {total_meta}, "
            f"验证通过: {total_verified} ({verification_rate:.1f}%), "
            f"负样本: {total_negatives}, "
            f"处理时间: {processing_time:.2f}s"
        )
        
        # 详细统计（调试模式）
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_detailed_statistics(meta_relations, verified_relations)
    
    def _log_detailed_statistics(
        self,
        meta_relations: List[List[Dict]],
        verified_relations: List[List[Dict]]
    ):
        """记录详细统计信息"""
        
        # 关系类型分布
        relation_types = {}
        confidence_scores = []
        
        for instance_relations in verified_relations:
            for rel_dict in instance_relations:
                rel_type = rel_dict.get('type', 'unknown')
                relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
                confidence_scores.append(rel_dict.get('confidence', 0.0))
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            self.logger.debug(f"关系类型分布: {relation_types}")
            self.logger.debug(f"平均置信度: {avg_confidence:.3f}")
    
    async def process_single_instance(
        self,
        head_concepts: List[ConceptInfo],
        tail_concepts: List[ConceptInfo],
        head_entity: str,
        tail_entity: str,
        context: str
    ) -> Dict[str, Any]:
        """处理单个实例的便利方法"""
        
        try:
            # 1. 元关系挖掘
            meta_relations = await self.relation_miner.mine_relations(
                head_concepts, tail_concepts, context
            )
            
            # 2. 语义一致性验证
            verified_relations = await self.consistency_verifier.verify_relations(
                meta_relations, head_entity, tail_entity, context
            )
            
            # 3. 负样本生成
            negative_samples = await self.negative_sampler.generate_negatives(
                verified_relations, context
            )
            
            return {
                'meta_relations': [rel.__dict__ for rel in meta_relations],
                'verified_relations': [rel.__dict__ for rel in verified_relations],
                'negative_samples': [rel.__dict__ for rel in negative_samples],
                'relation_confidences': [rel.confidence for rel in verified_relations]
            }
            
        except Exception as e:
            self.logger.error(f"单实例处理失败: {e}")
            return {
                'meta_relations': [],
                'verified_relations': [],
                'negative_samples': [],
                'relation_confidences': []
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.processing_stats = {
            'total_processed': 0,
            'total_relations_discovered': 0,
            'total_relations_verified': 0,
            'total_negatives_generated': 0,
            'avg_processing_time': 0.0
        }
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """获取各组件的统计信息"""
        return {
            'mrda_stats': self.processing_stats,
            'miner_available': self.relation_miner is not None,
            'verifier_available': self.consistency_verifier is not None,
            'sampler_available': self.negative_sampler is not None,
            'config': {
                'num_mining_stages': self.config.num_mining_stages,
                'verification_threshold': self.config.verification_threshold,
                'negative_ratio': self.config.negative_ratio
            }
        }


class MRDAFactory:
    """MRDA工厂类"""
    
    @staticmethod
    async def create_mrda(
        llm_service: BaseLLMService,
        config: MRDAConfig
    ) -> MetaRelationDiscoveryAgent:
        """创建MRDA实例"""
        
        mrda = MetaRelationDiscoveryAgent(llm_service, config)
        
        # 可以在这里进行异步初始化
        # await mrda.initialize()  # 如果需要的话
        
        return mrda
    
    @staticmethod
    def create_mrda_sync(
        llm_service: BaseLLMService,
        config: MRDAConfig
    ) -> MetaRelationDiscoveryAgent:
        """同步创建MRDA实例"""
        return MetaRelationDiscoveryAgent(llm_service, config)


# LangGraph节点函数
async def meta_relation_discovery(state: AgentState) -> AgentState:
    """LangGraph节点函数：元关系发现"""
    
    # 这里需要从某个地方获取LLM服务和配置
    # 在实际使用中，这些应该通过依赖注入或全局配置获取
    from .llm_service import create_llm_service
    from .config import MRDAConfig, LLMServiceConfig
    
    # 创建配置（使用默认值）
    llm_config = LLMServiceConfig()
    mrda_config = MRDAConfig()
    
    # 创建LLM服务
    llm_service = await create_llm_service(llm_config)
    
    try:
        # 创建MRDA
        mrda = await MRDAFactory.create_mrda(llm_service, mrda_config)
        
        # 处理状态
        result_state = await mrda.process(state)
        
        return result_state
        
    finally:
        # 清理LLM服务
        if hasattr(llm_service, '__aexit__'):
            await llm_service.__aexit__(None, None, None)


# 便利函数
async def create_mrda_with_service(
    service_type: str = "siliconflow",
    mrda_config: Optional[MRDAConfig] = None,
    llm_config: Optional[Any] = None
) -> MetaRelationDiscoveryAgent:
    """创建带有LLM服务的MRDA"""
    
    from .llm_service import create_llm_service
    from .config import LLMServiceConfig
    
    # 使用默认配置
    if llm_config is None:
        llm_config = LLMServiceConfig()
    if mrda_config is None:
        mrda_config = MRDAConfig()
    
    # 创建LLM服务
    llm_service = await create_llm_service(llm_config, service_type)
    
    # 创建MRDA
    mrda = await MRDAFactory.create_mrda(llm_service, mrda_config)
    
    return mrda