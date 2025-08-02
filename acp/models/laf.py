# -*- coding: utf-8 -*-
"""
Language-Aware Feature (LAF) module enhanced with LangGraph agent outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from transformers import AutoModel, AutoTokenizer

from ..agents.state import ConceptInfo, MetaRelation


class EntityMarker(nn.Module):
    """实体标记器"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.entity_start_marker = nn.Parameter(torch.randn(hidden_size))
        self.entity_end_marker = nn.Parameter(torch.randn(hidden_size))
    
    def forward(
        self, 
        sequence_output: torch.Tensor, 
        entity_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            entity_positions: [batch_size, 4] (h_start, h_end, t_start, t_end)
        
        Returns:
            head_entity_repr: [batch_size, hidden_size]
            tail_entity_repr: [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # 提取实体表示
        head_reprs = []
        tail_reprs = []
        
        for i in range(batch_size):
            h_start, h_end, t_start, t_end = entity_positions[i]
            
            # 确保位置在有效范围内
            h_start = max(0, min(h_start.item(), seq_len - 1))
            h_end = max(h_start, min(h_end.item(), seq_len - 1))
            t_start = max(0, min(t_start.item(), seq_len - 1))
            t_end = max(t_start, min(t_end.item(), seq_len - 1))
            
            # 头实体表示（平均池化）
            if h_end > h_start:
                head_repr = torch.mean(sequence_output[i, h_start:h_end+1], dim=0)
            else:
                head_repr = sequence_output[i, h_start]
            
            # 尾实体表示（平均池化）
            if t_end > t_start:
                tail_repr = torch.mean(sequence_output[i, t_start:t_end+1], dim=0)
            else:
                tail_repr = sequence_output[i, t_start]
            
            head_reprs.append(head_repr)
            tail_reprs.append(tail_repr)
        
        head_entity_repr = torch.stack(head_reprs)
        tail_entity_repr = torch.stack(tail_reprs)
        
        return head_entity_repr, tail_entity_repr


class ConceptIntegrator(nn.Module):
    """概念集成器，整合智能体输出的概念信息"""
    
    def __init__(
        self, 
        hidden_size: int, 
        concept_dim: int = 768,
        max_concepts: int = 10,
        integration_method: str = 'attention'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.concept_dim = concept_dim
        self.max_concepts = max_concepts
        self.integration_method = integration_method
        
        # 概念投影层
        self.concept_projection = nn.Linear(concept_dim, hidden_size)
        
        # 注意力机制
        if integration_method == 'attention':
            self.concept_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True
            )
        
        # 门控机制
        self.concept_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        entity_repr: torch.Tensor,
        concept_embeddings: Optional[torch.Tensor] = None,
        concept_weights: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            entity_repr: [batch_size, hidden_size] 实体表示
            concept_embeddings: [batch_size, max_concepts, concept_dim] 概念嵌入
            concept_weights: [batch_size, max_concepts] 概念权重
            concept_mask: [batch_size, max_concepts] 概念掩码
        
        Returns:
            enhanced_repr: [batch_size, hidden_size] 增强后的表示
        """
        if concept_embeddings is None:
            return entity_repr
        
        batch_size = entity_repr.size(0)
        
        # 投影概念嵌入
        projected_concepts = self.concept_projection(concept_embeddings)
        
        # 应用概念权重
        if concept_weights is not None:
            concept_weights = concept_weights.unsqueeze(-1)  # [batch_size, max_concepts, 1]
            projected_concepts = projected_concepts * concept_weights
        
        # 概念聚合
        if self.integration_method == 'attention':
            # 使用注意力机制聚合概念
            entity_query = entity_repr.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            concept_repr, _ = self.concept_attention(
                query=entity_query,
                key=projected_concepts,
                value=projected_concepts,
                key_padding_mask=~concept_mask if concept_mask is not None else None
            )
            concept_repr = concept_repr.squeeze(1)  # [batch_size, hidden_size]
            
        elif self.integration_method == 'mean':
            # 平均池化
            if concept_mask is not None:
                masked_concepts = projected_concepts * concept_mask.unsqueeze(-1)
                concept_repr = masked_concepts.sum(dim=1) / concept_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                concept_repr = projected_concepts.mean(dim=1)
        
        else:
            # 加权平均
            if concept_weights is not None:
                concept_repr = (projected_concepts * concept_weights.unsqueeze(-1)).sum(dim=1)
            else:
                concept_repr = projected_concepts.mean(dim=1)
        
        # 门控融合
        gate = self.concept_gate(torch.cat([entity_repr, concept_repr], dim=-1))
        gated_concept = gate * concept_repr
        
        # 最终融合
        enhanced_repr = self.fusion_layer(
            torch.cat([entity_repr, gated_concept], dim=-1)
        )
        
        return enhanced_repr


class RelationIntegrator(nn.Module):
    """关系集成器，整合智能体发现的元关系"""
    
    def __init__(
        self, 
        hidden_size: int,
        max_relations: int = 5,
        relation_dim: int = 768
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relations = max_relations
        self.relation_dim = relation_dim
        
        # 关系编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 关系注意力
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # 关系融合
        self.relation_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        relation_embeddings: Optional[torch.Tensor] = None,
        relation_confidences: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pair_repr: [batch_size, hidden_size] 实体对表示
            relation_embeddings: [batch_size, max_relations, relation_dim] 关系嵌入
            relation_confidences: [batch_size, max_relations] 关系置信度
        
        Returns:
            enhanced_pair_repr: [batch_size, hidden_size] 增强后的实体对表示
        """
        if relation_embeddings is None:
            return pair_repr
        
        # 编码关系
        encoded_relations = self.relation_encoder(relation_embeddings)
        
        # 应用置信度权重
        if relation_confidences is not None:
            confidence_weights = relation_confidences.unsqueeze(-1)
            encoded_relations = encoded_relations * confidence_weights
        
        # 关系注意力
        pair_query = pair_repr.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        relation_repr, _ = self.relation_attention(
            query=pair_query,
            key=encoded_relations,
            value=encoded_relations
        )
        relation_repr = relation_repr.squeeze(1)  # [batch_size, hidden_size]
        
        # 融合关系信息
        enhanced_repr = self.relation_fusion(
            torch.cat([pair_repr, relation_repr], dim=-1)
        )
        
        return enhanced_repr


class AgentEnhancedLAF(nn.Module):
    """智能体增强的语言感知特征模块"""
    
    def __init__(
        self,
        pretrained_model: str = 'bert-base-uncased',
        hidden_size: int = 768,
        concept_dim: int = 768,
        max_concepts: int = 10,
        max_relations: int = 5,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.hidden_size = hidden_size
        self.max_concepts = max_concepts
        self.max_relations = max_relations
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(pretrained_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 实体标记器
        self.entity_marker = EntityMarker(hidden_size)
        
        # 概念集成器
        self.concept_integrator = ConceptIntegrator(
            hidden_size=hidden_size,
            concept_dim=concept_dim,
            max_concepts=max_concepts
        )
        
        # 关系集成器
        self.relation_integrator = RelationIntegrator(
            hidden_size=hidden_size,
            max_relations=max_relations
        )
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor,
        agent_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            entity_positions: [batch_size, 4] (h_start, h_end, t_start, t_end)
            agent_results: 智能体处理结果
        
        Returns:
            Dict containing various representations
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 提取实体表示
        head_entity_repr, tail_entity_repr = self.entity_marker(
            sequence_output, entity_positions
        )
        
        # 处理智能体结果
        if agent_results:
            # 提取概念信息
            concept_embeddings, concept_weights, concept_mask = self._process_concept_results(
                agent_results
            )
            
            # 提取关系信息
            relation_embeddings, relation_confidences = self._process_relation_results(
                agent_results
            )
        else:
            concept_embeddings = concept_weights = concept_mask = None
            relation_embeddings = relation_confidences = None
        
        # 概念增强
        enhanced_head = self.concept_integrator(
            head_entity_repr, concept_embeddings, concept_weights, concept_mask
        )
        enhanced_tail = self.concept_integrator(
            tail_entity_repr, concept_embeddings, concept_weights, concept_mask
        )
        
        # 实体对表示
        pair_repr = torch.cat([enhanced_head, enhanced_tail], dim=-1)
        pair_repr = self.context_encoder(pair_repr)
        
        # 关系增强
        enhanced_pair_repr = self.relation_integrator(
            pair_repr, relation_embeddings, relation_confidences
        )
        
        # 上下文表示
        context_repr = self.context_encoder(pooled_output)
        
        # 最终融合
        final_repr = self.final_fusion(
            torch.cat([enhanced_head, enhanced_tail, context_repr], dim=-1)
        )
        
        return {
            'sequence_output': sequence_output,
            'pooled_output': pooled_output,
            'head_entity_repr': head_entity_repr,
            'tail_entity_repr': tail_entity_repr,
            'enhanced_head': enhanced_head,
            'enhanced_tail': enhanced_tail,
            'pair_repr': pair_repr,
            'enhanced_pair_repr': enhanced_pair_repr,
            'context_repr': context_repr,
            'final_repr': final_repr,
            'concept_embeddings': concept_embeddings,
            'relation_embeddings': relation_embeddings
        }
    
    def _process_concept_results(
        self, 
        agent_results: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """处理智能体的概念结果"""
        
        try:
            aligned_concepts = agent_results.get('aligned_concepts', [])
            concept_weights = agent_results.get('concept_weights', [])
            
            if not aligned_concepts:
                return None, None, None
            
            batch_size = len(aligned_concepts)
            concept_embeddings_list = []
            concept_weights_list = []
            concept_mask_list = []
            
            for i in range(batch_size):
                instance_concepts = aligned_concepts[i]
                instance_weights = concept_weights[i] if i < len(concept_weights) else []
                
                # 处理概念嵌入
                embeddings = []
                weights = []
                
                for j, concept_dict in enumerate(instance_concepts[:self.max_concepts]):
                    # 从概念字典中提取嵌入
                    if 'embedding' in concept_dict and concept_dict['embedding'] is not None:
                        embedding = concept_dict['embedding']
                        if isinstance(embedding, list):
                            embedding = torch.tensor(embedding, dtype=torch.float32)
                        elif not isinstance(embedding, torch.Tensor):
                            embedding = torch.zeros(self.hidden_size, dtype=torch.float32)
                    else:
                        embedding = torch.zeros(self.hidden_size, dtype=torch.float32)
                    
                    embeddings.append(embedding)
                    
                    # 权重
                    if j < len(instance_weights):
                        weights.append(instance_weights[j])
                    else:
                        weights.append(1.0)
                
                # 填充到最大长度
                while len(embeddings) < self.max_concepts:
                    embeddings.append(torch.zeros(self.hidden_size, dtype=torch.float32))
                    weights.append(0.0)
                
                # 创建掩码
                mask = [1.0] * len(instance_concepts) + [0.0] * (self.max_concepts - len(instance_concepts))
                mask = mask[:self.max_concepts]
                
                concept_embeddings_list.append(torch.stack(embeddings))
                concept_weights_list.append(torch.tensor(weights, dtype=torch.float32))
                concept_mask_list.append(torch.tensor(mask, dtype=torch.bool))
            
            concept_embeddings = torch.stack(concept_embeddings_list)
            concept_weights = torch.stack(concept_weights_list)
            concept_mask = torch.stack(concept_mask_list)
            
            return concept_embeddings, concept_weights, concept_mask
            
        except Exception as e:
            self.logger.error(f"处理概念结果失败: {e}")
            return None, None, None
    
    def _process_relation_results(
        self, 
        agent_results: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """处理智能体的关系结果"""
        
        try:
            verified_relations = agent_results.get('verified_relations', [])
            
            if not verified_relations:
                return None, None
            
            batch_size = len(verified_relations)
            relation_embeddings_list = []
            relation_confidences_list = []
            
            for i in range(batch_size):
                instance_relations = verified_relations[i]
                
                embeddings = []
                confidences = []
                
                for j, relation_dict in enumerate(instance_relations[:self.max_relations]):
                    # 简单的关系嵌入（可以改进）
                    relation_type = relation_dict.get('type', '')
                    relation_desc = relation_dict.get('description', '')
                    
                    # 使用随机嵌入作为占位符（实际应用中应该使用更好的方法）
                    embedding = torch.randn(self.hidden_size, dtype=torch.float32)
                    embeddings.append(embedding)
                    
                    # 置信度
                    confidence = relation_dict.get('confidence', 0.5)
                    confidences.append(confidence)
                
                # 填充到最大长度
                while len(embeddings) < self.max_relations:
                    embeddings.append(torch.zeros(self.hidden_size, dtype=torch.float32))
                    confidences.append(0.0)
                
                relation_embeddings_list.append(torch.stack(embeddings))
                relation_confidences_list.append(torch.tensor(confidences, dtype=torch.float32))
            
            relation_embeddings = torch.stack(relation_embeddings_list)
            relation_confidences = torch.stack(relation_confidences_list)
            
            return relation_embeddings, relation_confidences
            
        except Exception as e:
            self.logger.error(f"处理关系结果失败: {e}")
            return None, None


# 向后兼容的LAF类
class LAF(AgentEnhancedLAF):
    """向后兼容的LAF类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("使用智能体增强的LAF模块")


# 工厂函数
def create_laf_model(
    pretrained_model: str = 'bert-base-uncased',
    hidden_size: int = 768,
    enable_agent_enhancement: bool = True,
    **kwargs
) -> nn.Module:
    """创建LAF模型的工厂函数"""
    
    if enable_agent_enhancement:
        return AgentEnhancedLAF(
            pretrained_model=pretrained_model,
            hidden_size=hidden_size,
            **kwargs
        )
    else:
        # 返回基础版本（需要实现）
        return AgentEnhancedLAF(
            pretrained_model=pretrained_model,
            hidden_size=hidden_size,
            **kwargs
        )


# 使用示例
def example_usage():
    """使用示例"""
    
    # 创建模型
    model = AgentEnhancedLAF()
    
    # 准备输入
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    entity_positions = torch.tensor([[10, 12, 20, 22], [5, 7, 15, 17]])
    
    # 模拟智能体结果
    agent_results = {
        'aligned_concepts': [
            [
                {'name': 'person', 'embedding': torch.randn(768).tolist()},
                {'name': 'organization', 'embedding': torch.randn(768).tolist()}
            ],
            [
                {'name': 'location', 'embedding': torch.randn(768).tolist()}
            ]
        ],
        'concept_weights': [[0.8, 0.6], [0.9]],
        'verified_relations': [
            [
                {'type': 'work_for', 'description': 'employment relation', 'confidence': 0.85}
            ],
            [
                {'type': 'located_in', 'description': 'location relation', 'confidence': 0.75}
            ]
        ]
    }
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_positions=entity_positions,
            agent_results=agent_results
        )
    
    print("LAF输出形状:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")


if __name__ == "__main__":
    example_usage()