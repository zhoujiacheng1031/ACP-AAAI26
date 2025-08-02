# -*- coding: utf-8 -*-
"""
State definitions for LangGraph-based agent system
"""

import torch
from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from dataclasses import dataclass


class AgentState(TypedDict):
    """LangGraph状态定义，用于智能体间数据传递"""
    
    # 输入数据
    instances: List[Dict[str, Any]]  # 输入实例
    head_entities: List[str]  # 头实体列表
    tail_entities: List[str]  # 尾实体列表
    contexts: List[str]  # 上下文文本列表
    
    # 概念查询结果
    head_concepts: List[List[Dict[str, Any]]]  # 头实体概念集合
    tail_concepts: List[List[Dict[str, Any]]]  # 尾实体概念集合
    concept_embeddings: Optional[torch.Tensor]  # 概念嵌入
    
    # MRDA输出
    meta_relations: List[List[Dict[str, Any]]]  # 发现的元关系
    verified_relations: List[List[Dict[str, Any]]]  # 验证后的关系
    negative_samples: List[List[Dict[str, Any]]]  # 生成的负样本
    relation_confidences: List[List[float]]  # 关系置信度
    
    # RCAA输出
    reasoning_trees: List[Dict[str, Any]]  # 推理树结构
    aligned_concepts: List[List[Dict[str, Any]]]  # 对齐后的概念
    concept_weights: List[List[float]]  # 概念权重
    informativeness_scores: List[List[float]]  # 信息性分数
    
    # 最终输出
    enhanced_representations: Optional[torch.Tensor]  # 增强后的表示
    final_concept_embeddings: Optional[torch.Tensor]  # 最终概念嵌入
    
    # 元数据
    batch_size: int  # 批次大小
    n_way: int  # N-way设置
    k_shot: int  # K-shot设置
    processing_time: Optional[float]  # 处理时间
    error_info: Optional[Dict[str, Any]]  # 错误信息


@dataclass
class ConceptInfo:
    """概念信息数据类"""
    id: str
    name: str
    definition: str
    embedding: Optional[torch.Tensor] = None
    hop_distance: int = 1
    relevance_score: float = 0.0
    source: str = "concept_graph"  # 概念来源
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetaRelation:
    """元关系数据类"""
    type: str
    description: str
    confidence: float
    evidence: str
    verification_score: float = 0.0
    head_concept: str = ""
    tail_concept: str = ""
    stage: int = 1  # 发现阶段
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningPath:
    """推理路径数据类"""
    path_id: str
    concepts: List[ConceptInfo]
    attention_weights: List[float]
    informativeness_scores: List[float]
    final_representation: Optional[torch.Tensor] = None
    depth: int = 0
    parent_path: Optional[str] = None
    children_paths: List[str] = None
    
    def __post_init__(self):
        if self.children_paths is None:
            self.children_paths = []


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class StateValidator:
    """状态验证器"""
    
    @staticmethod
    def validate_input_state(state: AgentState) -> bool:
        """验证输入状态的完整性"""
        required_fields = ['instances', 'head_entities', 'tail_entities', 'contexts']
        
        for field in required_fields:
            if field not in state or not state[field]:
                return False
        
        # 检查列表长度一致性
        length = len(state['instances'])
        if (len(state['head_entities']) != length or 
            len(state['tail_entities']) != length or 
            len(state['contexts']) != length):
            return False
            
        return True
    
    @staticmethod
    def validate_concept_query_state(state: AgentState) -> bool:
        """验证概念查询后的状态"""
        if not StateValidator.validate_input_state(state):
            return False
            
        required_fields = ['head_concepts', 'tail_concepts']
        for field in required_fields:
            if field not in state or not state[field]:
                return False
                
        return len(state['head_concepts']) == len(state['instances'])
    
    @staticmethod
    def validate_mrda_state(state: AgentState) -> bool:
        """验证MRDA处理后的状态"""
        if not StateValidator.validate_concept_query_state(state):
            return False
            
        required_fields = ['meta_relations', 'verified_relations']
        for field in required_fields:
            if field not in state:
                return False
                
        return True
    
    @staticmethod
    def validate_final_state(state: AgentState) -> bool:
        """验证最终状态"""
        if not StateValidator.validate_mrda_state(state):
            return False
            
        required_fields = ['aligned_concepts', 'concept_weights']
        for field in required_fields:
            if field not in state:
                return False
                
        return True


class StateManager:
    """状态管理器"""
    
    @staticmethod
    def create_initial_state(
        instances: List[Dict[str, Any]],
        n_way: int,
        k_shot: int
    ) -> AgentState:
        """创建初始状态"""
        
        # 提取实体和上下文
        head_entities = []
        tail_entities = []
        contexts = []
        
        for instance in instances:
            head_entities.append(instance['h']['name'])
            tail_entities.append(instance['t']['name'])
            contexts.append(' '.join(instance['token']))
        
        return AgentState(
            instances=instances,
            head_entities=head_entities,
            tail_entities=tail_entities,
            contexts=contexts,
            head_concepts=[],
            tail_concepts=[],
            concept_embeddings=None,
            meta_relations=[],
            verified_relations=[],
            negative_samples=[],
            relation_confidences=[],
            reasoning_trees=[],
            aligned_concepts=[],
            concept_weights=[],
            informativeness_scores=[],
            enhanced_representations=None,
            final_concept_embeddings=None,
            batch_size=len(instances),
            n_way=n_way,
            k_shot=k_shot,
            processing_time=None,
            error_info=None
        )
    
    @staticmethod
    def update_state(
        state: AgentState, 
        updates: Dict[str, Any]
    ) -> AgentState:
        """更新状态"""
        new_state = state.copy()
        new_state.update(updates)
        return new_state
    
    @staticmethod
    def get_state_summary(state: AgentState) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'batch_size': state['batch_size'],
            'n_way': state['n_way'],
            'k_shot': state['k_shot'],
            'has_concepts': bool(state['head_concepts'] and state['tail_concepts']),
            'has_meta_relations': bool(state['meta_relations']),
            'has_aligned_concepts': bool(state['aligned_concepts']),
            'processing_time': state.get('processing_time'),
            'has_error': state.get('error_info') is not None
        }