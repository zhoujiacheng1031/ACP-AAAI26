# -*- coding: utf-8 -*-
"""
Relevant Concept Alignment Agent (RCAA) main implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from .state import AgentState, ConceptInfo, ReasoningPath, StateManager, StateValidator
from ..utils.llm_service import BaseLLMService
from ..config.config import RCAAConfig
from .rcaa_components import (
    ToTRecursiveReasoner,
    DualAttentionMechanism,
    ImplicitInformativenessModeler,
    ReasoningTree
)


class RelevantConceptAlignmentAgent:
    """相关概念对齐智能体"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: RCAAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.recursive_reasoner = ToTRecursiveReasoner(llm_service, config)
        self.dual_attention = DualAttentionMechanism(config)
        self.informativeness_modeler = ImplicitInformativenessModeler(config)
        
        # 统计信息
        self.processing_stats = {
            'total_processed': 0,
            'total_reasoning_trees': 0,
            'total_aligned_concepts': 0,
            'avg_processing_time': 0.0,
            'avg_concepts_per_instance': 0.0
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """处理RCAA主流程"""
        
        # 验证输入状态
        if not StateValidator.validate_mrda_state(state):
            error_msg = "RCAA输入状态验证失败"
            self.logger.error(error_msg)
            return StateManager.update_state(state, {
                'error_info': {'type': 'rcaa_validation_error', 'message': error_msg}
            })
        
        start_time = time.time()
        
        try:
            self.logger.info("开始RCAA处理流程")
            
            # 提取数据
            instances = state['instances']
            head_concepts_list = self._extract_concept_info(state['head_concepts'])
            tail_concepts_list = self._extract_concept_info(state['tail_concepts'])
            verified_relations_list = state['verified_relations']
            
            # 验证数据一致性
            if not self._validate_data_consistency(
                instances, head_concepts_list, tail_concepts_list, verified_relations_list
            ):
                raise ValueError("RCAA输入数据不一致")
            
            # 批量处理实例
            all_reasoning_trees = []
            all_aligned_concepts = []
            all_concept_weights = []
            all_informativeness_scores = []
            
            # 处理每个实例
            for i, (instance, head_concepts, tail_concepts, verified_relations) in enumerate(
                zip(instances, head_concepts_list, tail_concepts_list, verified_relations_list)
            ):
                self.logger.debug(f"处理实例 {i+1}/{len(instances)}")
                
                # 合并头尾概念
                all_concepts = head_concepts + tail_concepts
                
                # 1. 构建推理树
                reasoning_tree = await self.recursive_reasoner.build_reasoning_tree(
                    instance, all_concepts
                )
                
                # 2. 概念对齐和选择
                aligned_concepts, concept_weights, informativeness_scores = await self._align_and_select_concepts(
                    reasoning_tree, all_concepts, instance, verified_relations
                )
                
                # 收集结果
                all_reasoning_trees.append(self._serialize_reasoning_tree(reasoning_tree))
                all_aligned_concepts.append([concept.__dict__ for concept in aligned_concepts])
                all_concept_weights.append(concept_weights)
                all_informativeness_scores.append(informativeness_scores)
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_processing_stats(
                len(instances),
                len(all_reasoning_trees),
                sum(len(concepts) for concepts in all_aligned_concepts),
                processing_time
            )
            
            # 记录处理结果
            self._log_processing_results(
                all_reasoning_trees, all_aligned_concepts, processing_time
            )
            
            # 更新状态
            updated_state = StateManager.update_state(state, {
                'reasoning_trees': all_reasoning_trees,
                'aligned_concepts': all_aligned_concepts,
                'concept_weights': all_concept_weights,
                'informativeness_scores': all_informativeness_scores,
                'processing_time': processing_time
            })
            
            self.logger.info(f"RCAA处理完成 - 处理时间: {processing_time:.2f}s")
            
            return updated_state
            
        except Exception as e:
            error_msg = f"RCAA处理失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'rcaa_processing_error',
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
    
    def _validate_data_consistency(
        self,
        instances: List[Dict[str, Any]],
        head_concepts_list: List[List[ConceptInfo]],
        tail_concepts_list: List[List[ConceptInfo]],
        verified_relations_list: List[List[Dict[str, Any]]]
    ) -> bool:
        """验证数据一致性"""
        
        lengths = [
            len(instances),
            len(head_concepts_list),
            len(tail_concepts_list),
            len(verified_relations_list)
        ]
        
        return len(set(lengths)) == 1  # 所有长度都相同
    
    async def _align_and_select_concepts(
        self,
        reasoning_tree: ReasoningTree,
        all_concepts: List[ConceptInfo],
        instance: Dict[str, Any],
        verified_relations: List[Dict[str, Any]]
    ) -> Tuple[List[ConceptInfo], List[float], List[float]]:
        """对齐和选择概念"""
        
        try:
            # 1. 从推理树中提取推理路径
            reasoning_paths = self._extract_reasoning_paths(reasoning_tree)
            
            # 2. 计算概念的隐式信息性
            concept_informativeness = {}
            for concept in all_concepts:
                informativeness = self.informativeness_modeler.compute_informativeness(
                    concept, reasoning_paths
                )
                concept_informativeness[concept.id] = informativeness
            
            # 3. 使用双重注意力机制计算概念权重
            concept_weights = await self._compute_attention_weights(
                all_concepts, instance, verified_relations
            )
            
            # 4. 综合选择最相关的概念
            selected_concepts, final_weights, final_informativeness = self._select_final_concepts(
                all_concepts, concept_weights, concept_informativeness
            )
            
            return selected_concepts, final_weights, final_informativeness
            
        except Exception as e:
            self.logger.error(f"概念对齐和选择失败: {e}")
            # 返回默认结果
            return all_concepts[:self.config.max_selected_concepts], \
                   [1.0] * min(len(all_concepts), self.config.max_selected_concepts), \
                   [0.5] * min(len(all_concepts), self.config.max_selected_concepts)
    
    def _extract_reasoning_paths(self, reasoning_tree: ReasoningTree) -> List[List[ConceptInfo]]:
        """从推理树中提取推理路径"""
        
        paths = reasoning_tree.get_paths_to_leaves()
        concept_paths = []
        
        for path in paths:
            concept_path = []
            for node in path:
                if node.concept is not None:
                    concept_path.append(node.concept)
            
            if concept_path:  # 只添加非空路径
                concept_paths.append(concept_path)
        
        return concept_paths
    
    async def _compute_attention_weights(
        self,
        concepts: List[ConceptInfo],
        instance: Dict[str, Any],
        verified_relations: List[Dict[str, Any]]
    ) -> List[float]:
        """计算注意力权重"""
        
        if not concepts:
            return []
        
        try:
            # 准备概念嵌入
            concept_embeddings = self._prepare_concept_embeddings(concepts)
            
            # 准备关系实例嵌入
            relation_embedding = self._prepare_relation_embedding(instance, verified_relations)
            
            # 创建概念掩码
            concept_mask = torch.ones(1, len(concepts), dtype=torch.bool)
            
            # 计算注意力权重
            with torch.no_grad():
                attention_weights, attention_details = self.dual_attention(
                    concept_embeddings.unsqueeze(0),  # 添加batch维度
                    relation_embedding.unsqueeze(0),  # 添加batch维度
                    concept_mask
                )
            
            # 提取权重
            weights = attention_weights.squeeze(0).tolist()
            
            return weights
            
        except Exception as e:
            self.logger.error(f"计算注意力权重失败: {e}")
            # 返回均匀权重
            return [1.0 / len(concepts)] * len(concepts)
    
    def _prepare_concept_embeddings(self, concepts: List[ConceptInfo]) -> torch.Tensor:
        """准备概念嵌入"""
        
        embeddings = []
        embedding_dim = self.config.hidden_size
        
        for concept in concepts:
            if concept.embedding is not None:
                # 确保嵌入维度正确
                emb = concept.embedding.flatten()
                if emb.size(0) != embedding_dim:
                    # 调整维度
                    if emb.size(0) > embedding_dim:
                        emb = emb[:embedding_dim]
                    else:
                        # 填充零
                        padding = torch.zeros(embedding_dim - emb.size(0))
                        emb = torch.cat([emb, padding])
                embeddings.append(emb)
            else:
                # 使用零向量
                embeddings.append(torch.zeros(embedding_dim))
        
        return torch.stack(embeddings)
    
    def _prepare_relation_embedding(
        self,
        instance: Dict[str, Any],
        verified_relations: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """准备关系实例嵌入"""
        
        embedding_dim = self.config.hidden_size
        
        # 简化实现：使用随机嵌入表示关系实例
        # 在实际应用中，这里应该使用更复杂的方法来编码关系实例
        relation_embedding = torch.randn(embedding_dim)
        
        # 可以基于verified_relations调整嵌入
        if verified_relations:
            # 基于关系数量和置信度调整
            num_relations = len(verified_relations)
            avg_confidence = sum(rel.get('confidence', 0.5) for rel in verified_relations) / num_relations
            
            # 简单的调整策略
            relation_embedding = relation_embedding * avg_confidence
        
        return relation_embedding
    
    def _select_final_concepts(
        self,
        all_concepts: List[ConceptInfo],
        attention_weights: List[float],
        informativeness_scores: Dict[str, float]
    ) -> Tuple[List[ConceptInfo], List[float], List[float]]:
        """选择最终概念"""
        
        if not all_concepts:
            return [], [], []
        
        # 计算综合分数
        concept_scores = []
        for i, concept in enumerate(all_concepts):
            attention_weight = attention_weights[i] if i < len(attention_weights) else 0.0
            informativeness = informativeness_scores.get(concept.id, 0.0)
            
            # 综合分数 = 注意力权重 * 0.6 + 信息性 * 0.4
            combined_score = attention_weight * 0.6 + informativeness * 0.4
            
            concept_scores.append((concept, attention_weight, informativeness, combined_score))
        
        # 按综合分数排序
        concept_scores.sort(key=lambda x: x[3], reverse=True)
        
        # 应用阈值过滤
        filtered_scores = [
            (concept, att_weight, info_score, combined_score)
            for concept, att_weight, info_score, combined_score in concept_scores
            if combined_score >= self.config.concept_selection_threshold
        ]
        
        # 限制数量
        selected_scores = filtered_scores[:self.config.max_selected_concepts]
        
        # 分离结果
        selected_concepts = [item[0] for item in selected_scores]
        final_weights = [item[1] for item in selected_scores]
        final_informativeness = [item[2] for item in selected_scores]
        
        return selected_concepts, final_weights, final_informativeness
    
    def _serialize_reasoning_tree(self, reasoning_tree: ReasoningTree) -> Dict[str, Any]:
        """序列化推理树"""
        
        serialized_nodes = {}
        for node_id, node in reasoning_tree.nodes.items():
            serialized_nodes[node_id] = {
                'node_id': node.node_id,
                'concept': node.concept.__dict__ if node.concept else None,
                'parent_id': node.parent_id,
                'children_ids': node.children_ids,
                'depth': node.depth,
                'reasoning_score': node.reasoning_score,
                'metadata': node.metadata
            }
        
        return {
            'tree_id': reasoning_tree.tree_id,
            'root_entity': reasoning_tree.root_entity,
            'nodes': serialized_nodes,
            'max_depth': reasoning_tree.max_depth,
            'total_nodes': reasoning_tree.total_nodes
        }
    
    def _update_processing_stats(
        self,
        num_instances: int,
        num_trees: int,
        num_aligned_concepts: int,
        processing_time: float
    ):
        """更新处理统计信息"""
        
        self.processing_stats['total_processed'] += num_instances
        self.processing_stats['total_reasoning_trees'] += num_trees
        self.processing_stats['total_aligned_concepts'] += num_aligned_concepts
        
        # 更新平均处理时间
        total_processed = self.processing_stats['total_processed']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_processed - num_instances) + processing_time) / total_processed
        )
        
        # 更新平均概念数
        if total_processed > 0:
            self.processing_stats['avg_concepts_per_instance'] = (
                self.processing_stats['total_aligned_concepts'] / total_processed
            )
    
    def _log_processing_results(
        self,
        reasoning_trees: List[Dict[str, Any]],
        aligned_concepts: List[List[Dict[str, Any]]],
        processing_time: float
    ):
        """记录处理结果"""
        
        total_trees = len(reasoning_trees)
        total_concepts = sum(len(concepts) for concepts in aligned_concepts)
        avg_concepts_per_instance = total_concepts / len(aligned_concepts) if aligned_concepts else 0
        
        # 推理树统计
        tree_depths = [tree['max_depth'] for tree in reasoning_trees]
        avg_tree_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0
        
        self.logger.info(
            f"RCAA处理统计 - "
            f"推理树: {total_trees}, "
            f"对齐概念: {total_concepts}, "
            f"平均概念/实例: {avg_concepts_per_instance:.1f}, "
            f"平均树深度: {avg_tree_depth:.1f}, "
            f"处理时间: {processing_time:.2f}s"
        )
        
        # 详细统计（调试模式）
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_detailed_statistics(reasoning_trees, aligned_concepts)
    
    def _log_detailed_statistics(
        self,
        reasoning_trees: List[Dict[str, Any]],
        aligned_concepts: List[List[Dict[str, Any]]]
    ):
        """记录详细统计信息"""
        
        # 概念类型分布
        concept_types = {}
        for concepts in aligned_concepts:
            for concept_dict in concepts:
                concept_name = concept_dict.get('name', 'unknown')
                concept_types[concept_name] = concept_types.get(concept_name, 0) + 1
        
        # 树节点数分布
        node_counts = [tree['total_nodes'] for tree in reasoning_trees]
        
        self.logger.debug(f"概念类型分布: {dict(list(concept_types.items())[:10])}")  # 只显示前10个
        self.logger.debug(f"树节点数范围: {min(node_counts) if node_counts else 0} - {max(node_counts) if node_counts else 0}")
    
    async def process_single_instance(
        self,
        instance: Dict[str, Any],
        head_concepts: List[ConceptInfo],
        tail_concepts: List[ConceptInfo],
        verified_relations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理单个实例的便利方法"""
        
        try:
            # 合并概念
            all_concepts = head_concepts + tail_concepts
            
            # 构建推理树
            reasoning_tree = await self.recursive_reasoner.build_reasoning_tree(
                instance, all_concepts
            )
            
            # 概念对齐和选择
            aligned_concepts, concept_weights, informativeness_scores = await self._align_and_select_concepts(
                reasoning_tree, all_concepts, instance, verified_relations
            )
            
            return {
                'reasoning_tree': self._serialize_reasoning_tree(reasoning_tree),
                'aligned_concepts': [concept.__dict__ for concept in aligned_concepts],
                'concept_weights': concept_weights,
                'informativeness_scores': informativeness_scores
            }
            
        except Exception as e:
            self.logger.error(f"单实例处理失败: {e}")
            return {
                'reasoning_tree': {},
                'aligned_concepts': [],
                'concept_weights': [],
                'informativeness_scores': []
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """获取各组件的统计信息"""
        return {
            'rcaa_stats': self.processing_stats,
            'reasoner_available': self.recursive_reasoner is not None,
            'attention_available': self.dual_attention is not None,
            'modeler_available': self.informativeness_modeler is not None,
            'informativeness_stats': self.informativeness_modeler.get_concept_statistics(),
            'config': {
                'max_reasoning_depth': self.config.max_reasoning_depth,
                'max_selected_concepts': self.config.max_selected_concepts,
                'concept_selection_threshold': self.config.concept_selection_threshold
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.processing_stats = {
            'total_processed': 0,
            'total_reasoning_trees': 0,
            'total_aligned_concepts': 0,
            'avg_processing_time': 0.0,
            'avg_concepts_per_instance': 0.0
        }
        self.informativeness_modeler.clear_statistics()


class RCAAFactory:
    """RCAA工厂类"""
    
    @staticmethod
    async def create_rcaa(
        llm_service: BaseLLMService,
        config: RCAAConfig
    ) -> RelevantConceptAlignmentAgent:
        """创建RCAA实例"""
        
        rcaa = RelevantConceptAlignmentAgent(llm_service, config)
        
        # 可以在这里进行异步初始化
        # await rcaa.initialize()  # 如果需要的话
        
        return rcaa
    
    @staticmethod
    def create_rcaa_sync(
        llm_service: BaseLLMService,
        config: RCAAConfig
    ) -> RelevantConceptAlignmentAgent:
        """同步创建RCAA实例"""
        return RelevantConceptAlignmentAgent(llm_service, config)


# LangGraph节点函数
async def relevant_concept_alignment(state: AgentState) -> AgentState:
    """LangGraph节点函数：相关概念对齐"""
    
    # 这里需要从某个地方获取LLM服务和配置
    from .llm_service import create_llm_service
    from .config import RCAAConfig, LLMServiceConfig
    
    # 创建配置（使用默认值）
    llm_config = LLMServiceConfig()
    rcaa_config = RCAAConfig()
    
    # 创建LLM服务
    llm_service = await create_llm_service(llm_config)
    
    try:
        # 创建RCAA
        rcaa = await RCAAFactory.create_rcaa(llm_service, rcaa_config)
        
        # 处理状态
        result_state = await rcaa.process(state)
        
        return result_state
        
    finally:
        # 清理LLM服务
        if hasattr(llm_service, '__aexit__'):
            await llm_service.__aexit__(None, None, None)


# 便利函数
async def create_rcaa_with_service(
    service_type: str = "siliconflow",
    rcaa_config: Optional[RCAAConfig] = None,
    llm_config: Optional[Any] = None
) -> RelevantConceptAlignmentAgent:
    """创建带有LLM服务的RCAA"""
    
    from .llm_service import create_llm_service
    from .config import LLMServiceConfig
    
    # 使用默认配置
    if llm_config is None:
        llm_config = LLMServiceConfig()
    if rcaa_config is None:
        rcaa_config = RCAAConfig()
    
    # 创建LLM服务
    llm_service = await create_llm_service(llm_config, service_type)
    
    # 创建RCAA
    rcaa = await RCAAFactory.create_rcaa(llm_service, rcaa_config)
    
    return rcaa