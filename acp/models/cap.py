# -*- coding: utf-8 -*-
"""
Concept-Aware Prototypical Network (CAP) with NOTA detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

from .laf import AgentEnhancedLAF


class PrototypeDistanceDistributionModeling(nn.Module):
    """原型距离分布建模（PDDM）"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """计算距离分布
        
        Args:
            distances: 原型距离 [batch_size, n_way]
            
        Returns:
            距离分布 [batch_size, n_way]
        """
        return F.softmax(-self.alpha * distances, dim=-1)
    
    def compute_kl_loss(
        self,
        pred_dist: torch.Tensor,
        target_dist: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度损失"""
        return F.kl_div(
            pred_dist.log(),
            target_dist,
            reduction='batchmean'
        )


class NormalizedDistanceDeviationModeling(nn.Module):
    """标准化距离偏差建模（NDDM）用于NOTA检测"""
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self, 
        distances: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            distances: [batch_size, n_way] 查询到原型的距离
            
        Returns:
            normalized_deviations: [batch_size, n_way] 标准化偏差
            stats: 统计信息字典
        """
        # 计算每个查询的距离统计
        mu_d = torch.mean(distances, dim=1, keepdim=True)  # [batch_size, 1]
        sigma_d = torch.std(distances, dim=1, keepdim=True)  # [batch_size, 1]
        
        # 标准化偏差
        normalized_deviations = (distances - mu_d) / (sigma_d + self.epsilon)
        
        stats = {
            'mu_d': mu_d,
            'sigma_d': sigma_d,
            'min_distance': torch.min(distances, dim=1, keepdim=True)[0],
            'max_distance': torch.max(distances, dim=1, keepdim=True)[0]
        }
        
        return normalized_deviations, stats


class NOTADetector(nn.Module):
    """NOTA检测器"""
    
    def __init__(
        self,
        hidden_size: int,
        n_way: int,
        detection_method: str = 'threshold',
        threshold: float = 0.5,
        use_nddm: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_way = n_way
        self.detection_method = detection_method
        self.threshold = threshold
        self.use_nddm = use_nddm
        self.logger = logging.getLogger(__name__)
        
        # NDDM模块
        if use_nddm:
            self.nddm = NormalizedDistanceDeviationModeling()
        
        # 基于MLP的NOTA分类器
        if detection_method == 'mlp':
            self.nota_classifier = nn.Sequential(
                nn.Linear(n_way * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 2)  # [not_nota, nota]
            )
        
        # 基于距离的置信度计算
        self.confidence_temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        distances: torch.Tensor,
        method: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            distances: [batch_size, n_way] 查询到原型的距离
            method: 检测方法，如果为None则使用初始化时的方法
            
        Returns:
            Dict containing NOTA predictions and confidence scores
        """
        method = method or self.detection_method
        batch_size = distances.size(0)
        
        results = {}
        
        # 计算基础置信度
        confidence_scores = self._compute_confidence(distances)
        results['confidence_scores'] = confidence_scores
        
        # 最小距离和对应的类别
        min_distances, predicted_classes = torch.min(distances, dim=1)
        results['min_distances'] = min_distances
        results['predicted_classes'] = predicted_classes
        
        if method == 'threshold':
            # 基于阈值的NOTA检测
            nota_predictions = (confidence_scores < self.threshold).long()
            
        elif method == 'nddm' and self.use_nddm:
            # 基于NDDM的NOTA检测
            normalized_deviations, nddm_stats = self.nddm(distances)
            results['normalized_deviations'] = normalized_deviations
            results['nddm_stats'] = nddm_stats
            
            # 使用标准化偏差进行NOTA检测
            # 如果最小距离的标准化偏差过大，则认为是NOTA
            min_deviations = torch.gather(
                normalized_deviations, 1, predicted_classes.unsqueeze(1)
            ).squeeze(1)
            
            nota_predictions = (min_deviations > self.threshold).long()
            
        elif method == 'mlp':
            # 基于MLP的NOTA检测
            if self.use_nddm:
                normalized_deviations, _ = self.nddm(distances)
                mlp_input = torch.cat([distances, normalized_deviations], dim=1)
            else:
                # 使用原始距离和统计特征
                distance_stats = torch.cat([
                    distances,
                    distances.mean(dim=1, keepdim=True).expand(-1, self.n_way),
                ], dim=1)
                mlp_input = distance_stats
            
            nota_logits = self.nota_classifier(mlp_input)
            nota_predictions = torch.argmax(nota_logits, dim=1)
            results['nota_logits'] = nota_logits
            
        else:
            # 默认使用置信度阈值
            nota_predictions = (confidence_scores < self.threshold).long()
        
        results['nota_predictions'] = nota_predictions
        results['nota_probabilities'] = nota_predictions.float()
        
        return results
    
    def _compute_confidence(self, distances: torch.Tensor) -> torch.Tensor:
        """计算置信度分数"""
        # 使用softmax计算置信度
        confidences = F.softmax(-distances / self.confidence_temperature, dim=1)
        max_confidences, _ = torch.max(confidences, dim=1)
        return max_confidences


class ConceptAwarePrototypicalNetwork(nn.Module):
    """概念感知原型网络"""
    
    def __init__(
        self,
        laf_model: AgentEnhancedLAF,
        n_way: int,
        hidden_size: int = 768,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_pddm: bool = True,
        use_nota_detection: bool = True,
        nota_detection_method: str = 'nddm',
        nota_threshold: float = 0.5,
        margin: float = 1.0
    ):
        super().__init__()
        self.laf = laf_model
        self.n_way = n_way
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.margin = margin
        self.use_pddm = use_pddm
        self.use_nota_detection = use_nota_detection
        self.logger = logging.getLogger(__name__)
        
        # 原型投影层
        self.prototype_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 查询投影层
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # PDDM模块
        if use_pddm:
            self.pddm = PrototypeDistanceDistributionModeling(alpha=1.0)
        
        # NOTA检测器
        if use_nota_detection:
            self.nota_detector = NOTADetector(
                hidden_size=hidden_size,
                n_way=n_way,
                detection_method=nota_detection_method,
                threshold=nota_threshold,
                use_nddm=True
            )
        
        # 距离度量
        self.distance_metric = 'euclidean'  # 'euclidean' or 'cosine'
    
    def forward(
        self,
        support_input_ids: torch.Tensor,
        support_attention_mask: torch.Tensor,
        support_entity_positions: torch.Tensor,
        support_labels: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_entity_positions: torch.Tensor,
        support_agent_results: Optional[List[Dict[str, Any]]] = None,
        query_agent_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            support_*: 支持集输入
            query_*: 查询集输入
            *_agent_results: 智能体处理结果
            
        Returns:
            Dict containing predictions, distances, and other outputs
        """
        # 编码支持集
        support_outputs = self._encode_instances(
            support_input_ids,
            support_attention_mask,
            support_entity_positions,
            support_agent_results
        )
        
        # 编码查询集
        query_outputs = self._encode_instances(
            query_input_ids,
            query_attention_mask,
            query_entity_positions,
            query_agent_results
        )
        
        # 计算原型
        prototypes = self._compute_prototypes(
            support_outputs['final_repr'],
            support_labels
        )
        
        # 投影原型和查询
        projected_prototypes = self.prototype_projection(prototypes)
        projected_queries = self.query_projection(query_outputs['final_repr'])
        
        # 计算距离
        distances = self._compute_distances(projected_queries, projected_prototypes)
        
        # 基础分类logits
        logits = -distances / self.temperature
        
        results = {
            'logits': logits,
            'distances': distances,
            'prototypes': projected_prototypes,
            'query_representations': projected_queries,
            'support_outputs': support_outputs,
            'query_outputs': query_outputs
        }
        
        # PDDM处理
        if self.use_pddm:
            distance_distributions = self.pddm(distances)
            results['distance_distributions'] = distance_distributions
        
        # NOTA检测
        if self.use_nota_detection:
            nota_results = self.nota_detector(distances)
            results.update(nota_results)
        
        return results
    
    def _encode_instances(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor,
        agent_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        """编码实例"""
        
        batch_size = input_ids.size(0)
        
        # 处理智能体结果
        if agent_results and len(agent_results) == batch_size:
            # 为每个实例准备智能体结果
            processed_agent_results = {}
            
            # 收集所有实例的概念和关系
            all_aligned_concepts = []
            all_concept_weights = []
            all_verified_relations = []
            
            for i, result in enumerate(agent_results):
                if result:
                    all_aligned_concepts.append(result.get('aligned_concepts', []))
                    all_concept_weights.append(result.get('concept_weights', []))
                    all_verified_relations.append(result.get('verified_relations', []))
                else:
                    all_aligned_concepts.append([])
                    all_concept_weights.append([])
                    all_verified_relations.append([])
            
            processed_agent_results = {
                'aligned_concepts': all_aligned_concepts,
                'concept_weights': all_concept_weights,
                'verified_relations': all_verified_relations
            }
        else:
            processed_agent_results = None
        
        # LAF编码
        laf_outputs = self.laf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_positions=entity_positions,
            agent_results=processed_agent_results
        )
        
        return laf_outputs
    
    def _compute_prototypes(
        self,
        support_representations: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算类原型
        
        Args:
            support_representations: [batch_size * n_way * k_shot, hidden_size]
            support_labels: [batch_size * n_way * k_shot]
            
        Returns:
            prototypes: [batch_size, n_way, hidden_size]
        """
        batch_size = support_representations.size(0) // (self.n_way * 1)  # 假设k_shot=1
        
        # 重塑为 [batch_size, n_way * k_shot, hidden_size]
        support_representations = support_representations.view(
            batch_size, -1, self.hidden_size
        )
        support_labels = support_labels.view(batch_size, -1)
        
        prototypes = []
        
        for b in range(batch_size):
            batch_prototypes = []
            
            for way in range(self.n_way):
                # 找到属于当前类的样本
                class_mask = (support_labels[b] == way)
                
                if class_mask.sum() > 0:
                    class_representations = support_representations[b][class_mask]
                    prototype = torch.mean(class_representations, dim=0)
                else:
                    # 如果没有样本，使用零向量
                    prototype = torch.zeros(self.hidden_size, device=support_representations.device)
                
                batch_prototypes.append(prototype)
            
            prototypes.append(torch.stack(batch_prototypes))
        
        return torch.stack(prototypes)  # [batch_size, n_way, hidden_size]
    
    def _compute_distances(
        self,
        queries: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """计算查询到原型的距离
        
        Args:
            queries: [batch_size * n_query, hidden_size]
            prototypes: [batch_size, n_way, hidden_size]
            
        Returns:
            distances: [batch_size * n_query, n_way]
        """
        batch_size, n_way, hidden_size = prototypes.shape
        n_query = queries.size(0) // batch_size
        
        # 重塑查询
        queries = queries.view(batch_size, n_query, hidden_size)
        
        # 计算距离
        distances = []
        
        for b in range(batch_size):
            batch_queries = queries[b]  # [n_query, hidden_size]
            batch_prototypes = prototypes[b]  # [n_way, hidden_size]
            
            if self.distance_metric == 'euclidean':
                # 欧几里得距离
                batch_distances = torch.cdist(
                    batch_queries.unsqueeze(0),
                    batch_prototypes.unsqueeze(0)
                ).squeeze(0)  # [n_query, n_way]
                
            elif self.distance_metric == 'cosine':
                # 余弦距离
                queries_norm = F.normalize(batch_queries, p=2, dim=1)
                prototypes_norm = F.normalize(batch_prototypes, p=2, dim=1)
                
                cosine_sim = torch.mm(queries_norm, prototypes_norm.t())
                batch_distances = 1 - cosine_sim  # [n_query, n_way]
            
            else:
                raise ValueError(f"不支持的距离度量: {self.distance_metric}")
            
            distances.append(batch_distances)
        
        # 合并所有批次
        distances = torch.cat(distances, dim=0)  # [batch_size * n_query, n_way]
        
        return distances
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        query_labels: torch.Tensor,
        nota_labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """计算多任务损失
        
        Args:
            outputs: 模型输出
            query_labels: 查询标签
            nota_labels: NOTA标签 (0: not NOTA, 1: NOTA)
            loss_weights: 损失权重
            
        Returns:
            Dict containing various losses
        """
        if loss_weights is None:
            loss_weights = {
                'classification': 1.0,
                'separation': 0.5,
                'nota': 0.5,
                'pddm': 0.3
            }
        
        losses = {}
        
        # 1. 分类损失
        logits = outputs['logits']
        classification_loss = F.cross_entropy(logits, query_labels)
        losses['classification_loss'] = classification_loss
        
        # 2. 已知样本分离损失（margin-based contrastive loss）
        if 'distances' in outputs:
            separation_loss = self._compute_separation_loss(
                outputs['distances'], query_labels
            )
            losses['separation_loss'] = separation_loss
        
        # 3. NOTA检测损失
        if self.use_nota_detection and nota_labels is not None:
            if 'nota_logits' in outputs:
                nota_loss = F.cross_entropy(outputs['nota_logits'], nota_labels)
            else:
                # 使用置信度进行NOTA损失计算
                confidence_scores = outputs.get('confidence_scores')
                if confidence_scores is not None:
                    # 将NOTA标签转换为置信度目标
                    confidence_targets = 1.0 - nota_labels.float()
                    nota_loss = F.mse_loss(confidence_scores, confidence_targets)
                else:
                    nota_loss = torch.tensor(0.0, device=logits.device)
            
            losses['nota_loss'] = nota_loss
        
        # 4. PDDM损失（如果使用）
        if self.use_pddm and 'distance_distributions' in outputs:
            # 创建目标分布（one-hot）
            target_dist = F.one_hot(query_labels, num_classes=self.n_way).float()
            pddm_loss = self.pddm.compute_kl_loss(
                outputs['distance_distributions'], target_dist
            )
            losses['pddm_loss'] = pddm_loss
        
        # 总损失
        total_loss = sum(
            loss_weights.get(key.replace('_loss', ''), 0.0) * loss
            for key, loss in losses.items()
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_separation_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算分离损失"""
        
        batch_size = distances.size(0)
        n_way = distances.size(1)
        
        # 获取正确类别的距离
        correct_distances = torch.gather(
            distances, 1, labels.unsqueeze(1)
        ).squeeze(1)  # [batch_size]
        
        # 获取最近的错误类别距离
        mask = torch.ones_like(distances)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        
        masked_distances = distances + (1 - mask) * 1e6  # 屏蔽正确类别
        nearest_wrong_distances, _ = torch.min(masked_distances, dim=1)
        
        # Margin-based loss
        separation_loss = F.relu(
            correct_distances - nearest_wrong_distances + self.margin
        ).mean()
        
        return separation_loss


# 工厂函数
def create_cap_model(
    pretrained_model: str = 'bert-base-uncased',
    n_way: int = 5,
    hidden_size: int = 768,
    enable_agent_enhancement: bool = True,
    **kwargs
) -> ConceptAwarePrototypicalNetwork:
    """创建CAP模型的工厂函数"""
    
    from .laf import create_laf_model
    
    # 创建LAF模型
    laf_model = create_laf_model(
        pretrained_model=pretrained_model,
        hidden_size=hidden_size,
        enable_agent_enhancement=enable_agent_enhancement
    )
    
    # 创建CAP模型
    cap_model = ConceptAwarePrototypicalNetwork(
        laf_model=laf_model,
        n_way=n_way,
        hidden_size=hidden_size,
        **kwargs
    )
    
    return cap_model


# 使用示例
def example_usage():
    """使用示例"""
    
    # 创建模型
    model = create_cap_model(n_way=5, enable_agent_enhancement=True)
    
    # 准备输入数据
    batch_size = 2
    n_way = 5
    k_shot = 1
    n_query = 5
    seq_len = 128
    
    # 支持集
    support_input_ids = torch.randint(0, 1000, (batch_size * n_way * k_shot, seq_len))
    support_attention_mask = torch.ones(batch_size * n_way * k_shot, seq_len)
    support_entity_positions = torch.randint(0, seq_len, (batch_size * n_way * k_shot, 4))
    support_labels = torch.arange(n_way).repeat(batch_size * k_shot)
    
    # 查询集
    query_input_ids = torch.randint(0, 1000, (batch_size * n_query, seq_len))
    query_attention_mask = torch.ones(batch_size * n_query, seq_len)
    query_entity_positions = torch.randint(0, seq_len, (batch_size * n_query, 4))
    query_labels = torch.randint(0, n_way, (batch_size * n_query,))
    nota_labels = torch.randint(0, 2, (batch_size * n_query,))
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            support_input_ids=support_input_ids,
            support_attention_mask=support_attention_mask,
            support_entity_positions=support_entity_positions,
            support_labels=support_labels,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_entity_positions=query_entity_positions
        )
        
        # 计算损失
        losses = model.compute_loss(outputs, query_labels, nota_labels)
    
    print("CAP模型输出:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n损失:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")


if __name__ == "__main__":
    example_usage()