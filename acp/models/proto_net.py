# -*- coding: utf-8 -*-
"""
Created on 2024-07-31
@purpose: Contrastive Prototypical Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .laf import LAF

class PDDM(nn.Module):
    """原型距离分布建模模块"""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """计算距离分布
        
        Args:
            distances: 原型距离 [n_query, n_way]
            
        Returns:
            距离分布 [n_query, n_way]
        """
        return F.softmax(-self.alpha * distances, dim=-1)
        
    def compute_loss(
        self,
        pred_dist: torch.Tensor,
        true_dist: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度损失
        
        Args:
            pred_dist: 预测分布 [n_query, n_way]
            true_dist: 真实分布 [n_query, n_way]
            
        Returns:
            KL散度损失
        """
        return F.kl_div(
            pred_dist.log(),
            true_dist,
            reduction='batchmean'
        )

class NOTAClassifier(nn.Module):
    """NOTA分类器"""
    def __init__(self, n_way: int, hidden_size: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_way*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, distances: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            distances: 原型距离 [n_query, n_way]
            mu: 均值 [n_way]
            sigma: 标准差 [n_way]
            
        Returns:
            NOTA预测 [n_query, 2]
        """
        # 距离归一化
        v = (distances - mu) / (sigma + 1e-5)
        x = torch.cat([distances, v], dim=-1)
        return self.mlp(x)

class ConceptEnhancedProtoNet(nn.Module):
    """概念感知适应型原型网络"""
    
    def __init__(
        self,
        laf_model: LAF,
        n_way: int,
        hidden_size: int = 768,
        temperature: float = 1.0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1
    ):
        super().__init__()
        self.laf = laf_model
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        
        # 原型投影层
        self.proto_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 概念融合层
        self.concept_fusion = nn.Linear(2 * hidden_size, hidden_size)
        
        # 距离度量
        self.distance = nn.CosineSimilarity(dim=-1)
        
        # PDDM模块
        self.pddm = PDDM(alpha=alpha)
        
        # NOTA分类器
        self.nota_classifier = NOTAClassifier(n_way=n_way)
        
    def forward(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            support_data: 支持集数据
            query_data: 查询集数据
            
        Returns:
            包含损失和预测的字典
        """
        # 编码支持集和查询集
        support_outputs = self.laf(**support_data)
        query_outputs = self.laf(**query_data)
        
        # 获取增强表示
        support_repr = support_outputs['enhanced_context']  # [n_way*k_shot, hidden]
        query_repr = query_outputs['enhanced_context']      # [n_query, hidden]
        
        # 获取概念表示
        support_concepts = support_outputs['concept_repr']  # [n_way*k_shot, hidden]
        
        # 计算类原型
        n_way = support_data['n_way']
        k_shot = support_data['k_shot']
        prototypes = self._compute_prototypes(
            support_repr,
            support_concepts,
            n_way,
            k_shot
        )  # [n_way, hidden]
        
        # 投影原型
        prototypes = self.proto_proj(prototypes)
        
        # 计算查询样本与原型的距离
        distances = self._compute_distances(
            query_repr,
            prototypes
        )  # [n_query, n_way]
        
        # 计算距离分布
        dist_probs = self.pddm(distances)
        
        # NOTA检测
        nota_logits = self.nota_classifier(distances, support_outputs['mu'], support_outputs['sigma'])
        
        return {
            'logits': distances / self.temperature,
            'dist_probs': dist_probs,
            'nota_logits': nota_logits,
            'prototypes': prototypes,
            'support_repr': support_repr,
            'query_repr': query_repr
        }
        
    def _compute_prototypes(
        self,
        support_repr: torch.Tensor,
        support_concepts: torch.Tensor,
        n_way: int,
        k_shot: int,
        support_relations: Optional[List[List[Dict]]] = None,  # 新增：每个样本的智能体概念与关系
        support_rel_weights: Optional[List[List[float]]] = None
    ) -> torch.Tensor:
        """融合实例、智能体概念与隐式关系，计算类原型"""
        support_repr = support_repr.view(n_way, k_shot, -1)
        support_concepts = support_concepts.view(n_way, k_shot, -1)
        # 智能体概念增强（如有）
        if support_relations is not None and support_rel_weights is not None:
            # 计算智能体概念和隐式关系的加权和
            agent_concept_repr = []
            for i in range(n_way):
                rel_vec = 0
                total_w = 0
                for j, rel in enumerate(support_relations[i]):
                    rel_vec += rel['vector'] * support_rel_weights[i][j]
                    total_w += support_rel_weights[i][j]
                if total_w > 0:
                    rel_vec = rel_vec / total_w
                agent_concept_repr.append(rel_vec)
            agent_concept_repr = torch.stack(agent_concept_repr).unsqueeze(1).expand(-1, k_shot, -1)
            support_repr = torch.cat([support_repr, support_concepts, agent_concept_repr], dim=-1)
        else:
            support_repr = torch.cat([support_repr, support_concepts], dim=-1)
        support_repr = self.concept_fusion(support_repr)
        prototypes = support_repr.mean(dim=1)
        return prototypes
        
    def _compute_distances(
        self,
        query_repr: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """计算查询样本与原型的距离
        
        Args:
            query_repr: 查询集表示 [n_query, hidden]
            prototypes: 类原型表示 [n_way, hidden]
            
        Returns:
            距离分数 [n_query, n_way]
        """
        # 扩展维度用于广播
        query_repr = query_repr.unsqueeze(1)    # [n_query, 1, hidden]
        prototypes = prototypes.unsqueeze(0)    # [1, n_way, hidden]
        
        # 计算余弦相似度
        distances = self.distance(query_repr, prototypes)  # [n_query, n_way]
        
        return distances
        
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        nota_labels: Optional[torch.Tensor] = None,
        proto_stats: Optional[Dict] = None,
        contrastive_pairs: Optional[Tuple] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """多任务损失聚合：关系分类、PDDM、NOTA、原型对比"""
        logits = outputs['logits']
        dist_probs = outputs['dist_probs']
        nota_logits = outputs['nota_logits']
        prototypes = outputs['prototypes']
        support_repr = outputs['support_repr']
        query_repr = outputs['query_repr']
        # 1. 关系分类损失（基于PDDM距离分布）
        cls_loss = F.cross_entropy(logits, labels)
        # 2. PDDM距离分布损失
        if proto_stats is not None:
            # 真实分布参数
            mu = proto_stats['mu']  # [n_way]
            sigma = proto_stats['sigma']  # [n_way]
            dists = outputs['logits']
            # 解析KL散度
            pddm_loss = ((torch.log(sigma) - torch.log(mu)) + (mu**2 + (mu - sigma)**2) / (2 * sigma**2) - 0.5).mean()
        else:
            pddm_loss = torch.tensor(0.0, device=logits.device)
        # 3. NOTA检测损失
        if nota_labels is not None:
            nota_loss = F.cross_entropy(nota_logits, nota_labels)
        else:
            nota_loss = torch.tensor(0.0, device=logits.device)
        # 4. 原型对比损失
        if contrastive_pairs is not None:
            proto_contrast_loss = self._compute_proto_contrast_loss(*contrastive_pairs)
        else:
            proto_contrast_loss = torch.tensor(0.0, device=logits.device)
        # 动态权重
        lw = loss_weights or {'cls':1.0, 'pddm':0.5, 'nota':0.5, 'contrast':0.2}
        total_loss = lw['cls']*cls_loss + lw['pddm']*pddm_loss + lw['nota']*nota_loss + lw['contrast']*proto_contrast_loss
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'pddm_loss': pddm_loss,
            'nota_loss': nota_loss,
            'proto_contrast_loss': proto_contrast_loss
        }
        
    def _compute_proto_contrast_loss(
        self,
        support_repr: torch.Tensor,
        query_repr: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算原型对比损失
        
        Args:
            support_repr: 支持集表示
            query_repr: 查询集表示
            prototypes: 类原型
            labels: 标签
            
        Returns:
            对比损失
        """
        # 计算正样本对
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # 计算表示相似度
        sim_matrix = F.cosine_similarity(
            query_repr.unsqueeze(1),
            support_repr.unsqueeze(0),
            dim=-1
        )
        
        # InfoNCE损失
        pos_sim = (sim_matrix * pos_mask).sum(dim=1)
        neg_sim = (sim_matrix * (1 - pos_mask)).sum(dim=1)
        
        loss = -torch.log(
            torch.exp(pos_sim / self.temperature) /
            (torch.exp(pos_sim / self.temperature) + 
             torch.exp(neg_sim / self.temperature))
        ).mean()
        
        return loss 