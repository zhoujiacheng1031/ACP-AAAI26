# -*- coding: utf-8 -*-
"""
Relevant Concept Alignment Agent (RCAA) components
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import uuid

from .state import ConceptInfo, ReasoningPath, ProcessingResult
from .llm_service import BaseLLMService
from .config import RCAAConfig


@dataclass
class ReasoningNode:
    """推理树节点"""
    node_id: str
    concept: Optional[ConceptInfo] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    reasoning_score: float = 0.0
    path_representation: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def add_child(self, child_id: str):
        """添加子节点"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return len(self.children_ids) == 0


@dataclass
class ReasoningTree:
    """推理树"""
    tree_id: str
    root_entity: str
    nodes: Dict[str, ReasoningNode] = field(default_factory=dict)
    max_depth: int = 0
    total_nodes: int = 0
    
    def add_node(self, node: ReasoningNode):
        """添加节点"""
        self.nodes[node.node_id] = node
        self.max_depth = max(self.max_depth, node.depth)
        self.total_nodes += 1
        
        # 更新父节点的子节点列表
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].add_child(node.node_id)
    
    def get_root_node(self) -> Optional[ReasoningNode]:
        """获取根节点"""
        for node in self.nodes.values():
            if node.parent_id is None:
                return node
        return None
    
    def get_leaf_nodes(self) -> List[ReasoningNode]:
        """获取所有叶节点"""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_paths_to_leaves(self) -> List[List[ReasoningNode]]:
        """获取从根到所有叶节点的路径"""
        root = self.get_root_node()
        if not root:
            return []
        
        paths = []
        self._dfs_paths(root, [], paths)
        return paths
    
    def _dfs_paths(
        self, 
        node: ReasoningNode, 
        current_path: List[ReasoningNode], 
        all_paths: List[List[ReasoningNode]]
    ):
        """深度优先搜索获取路径"""
        current_path.append(node)
        
        if node.is_leaf():
            all_paths.append(current_path.copy())
        else:
            for child_id in node.children_ids:
                if child_id in self.nodes:
                    self._dfs_paths(self.nodes[child_id], current_path, all_paths)
        
        current_path.pop()


class ToTRecursiveReasoner:
    """Tree-of-Thought递归推理器"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: RCAAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def build_reasoning_tree(
        self,
        instance: Dict[str, Any],
        concepts: List[ConceptInfo]
    ) -> ReasoningTree:
        """构建推理树"""
        
        start_time = time.time()
        
        try:
            # 提取根实体
            root_entity = self._extract_root_entity(instance)
            
            # 创建推理树
            tree = ReasoningTree(
                tree_id=str(uuid.uuid4()),
                root_entity=root_entity
            )
            
            # 创建根节点
            root_node = ReasoningNode(
                node_id=str(uuid.uuid4()),
                concept=None,  # 根节点不对应具体概念
                depth=0,
                metadata={
                    'entity': root_entity,
                    'is_root': True,
                    'creation_time': time.time()
                }
            )
            
            tree.add_node(root_node)
            
            self.logger.debug(f"开始构建推理树 - 根实体: {root_entity}")
            
            # 递归扩展推理树
            await self._recursive_expand(
                tree, root_node, concepts, instance, 0
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"推理树构建完成 - 节点数: {tree.total_nodes}, "
                f"最大深度: {tree.max_depth}, "
                f"处理时间: {processing_time:.2f}s"
            )
            
            return tree
            
        except Exception as e:
            error_msg = f"推理树构建失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def _extract_root_entity(self, instance: Dict[str, Any]) -> str:
        """提取根实体"""
        # 优先使用头实体
        if 'h' in instance and 'name' in instance['h']:
            return instance['h']['name']
        elif 'head_entity' in instance:
            return instance['head_entity']
        elif 'entity' in instance:
            return instance['entity']
        else:
            return "unknown_entity"
    
    async def _recursive_expand(
        self,
        tree: ReasoningTree,
        parent_node: ReasoningNode,
        available_concepts: List[ConceptInfo],
        instance: Dict[str, Any],
        current_depth: int
    ):
        """递归扩展推理树"""
        
        # 检查深度限制
        if current_depth >= self.config.max_reasoning_depth:
            return
        
        # 为当前节点生成子概念
        child_concepts = await self._generate_child_concepts(
            parent_node, available_concepts, instance, current_depth
        )
        
        # 限制子概念数量
        child_concepts = child_concepts[:self.config.max_concepts_per_path]
        
        # 为每个子概念创建节点
        for concept in child_concepts:
            child_node = ReasoningNode(
                node_id=str(uuid.uuid4()),
                concept=concept,
                parent_id=parent_node.node_id,
                depth=current_depth + 1,
                metadata={
                    'generation_method': 'llm_reasoning',
                    'creation_time': time.time()
                }
            )
            
            tree.add_node(child_node)
            
            # 递归扩展子节点
            await self._recursive_expand(
                tree, child_node, available_concepts, instance, current_depth + 1
            )
    
    async def _generate_child_concepts(
        self,
        parent_node: ReasoningNode,
        available_concepts: List[ConceptInfo],
        instance: Dict[str, Any],
        depth: int
    ) -> List[ConceptInfo]:
        """为父节点生成子概念"""
        
        # 构建推理提示
        reasoning_prompt = self._create_reasoning_prompt(
            parent_node, available_concepts, instance, depth
        )
        
        try:
            # 调用LLM进行推理
            response = await self.llm_service.query(
                prompt=reasoning_prompt,
                task_type="concept_reasoning",
                temperature=self.config.reasoning_temperature
            )
            
            # 解析推理结果
            child_concepts = self._parse_reasoning_response(
                response.content, available_concepts, depth
            )
            
            return child_concepts
            
        except Exception as e:
            self.logger.error(f"子概念生成失败: {e}")
            return []
    
    def _create_reasoning_prompt(
        self,
        parent_node: ReasoningNode,
        available_concepts: List[ConceptInfo],
        instance: Dict[str, Any],
        depth: int
    ) -> str:
        """创建推理提示"""
        
        # 获取上下文信息
        context = ' '.join(instance.get('token', []))
        
        # 获取父概念信息
        if parent_node.concept:
            parent_info = f"父概念: {parent_node.concept.name} - {parent_node.concept.definition}"
        else:
            parent_info = f"根实体: {parent_node.metadata.get('entity', 'unknown')}"
        
        # 格式化可用概念
        concept_list = []
        for i, concept in enumerate(available_concepts[:20]):  # 限制概念数量
            concept_list.append(
                f"{i+1}. {concept.name}: {concept.definition} "
                f"(相关性: {concept.relevance_score:.2f})"
            )
        
        prompt = f"""基于以下信息进行概念推理：

{parent_info}
上下文: {context}
当前推理深度: {depth + 1}

可用概念列表:
{chr(10).join(concept_list)}

请从可用概念中选择与父概念/实体最相关的概念，进行下一步推理。

推理要求:
1. 选择2-3个最相关的概念
2. 解释选择理由
3. 评估每个概念的重要性分数(0-1)
4. 考虑概念间的逻辑关联

输出格式（每行一个概念）:
概念序号|重要性分数(0-1)|推理依据"""
        
        return prompt
    
    def _parse_reasoning_response(
        self,
        response_content: str,
        available_concepts: List[ConceptInfo],
        depth: int
    ) -> List[ConceptInfo]:
        """解析推理响应"""
        
        selected_concepts = []
        lines = response_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            try:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    concept_idx = int(parts[0]) - 1  # 转换为0索引
                    importance_score = float(parts[1])
                    reasoning_basis = parts[2]
                    
                    # 验证索引有效性
                    if 0 <= concept_idx < len(available_concepts):
                        concept = available_concepts[concept_idx]
                        
                        # 更新概念的元数据
                        if concept.metadata is None:
                            concept.metadata = {}
                        
                        concept.metadata.update({
                            'reasoning_importance': importance_score,
                            'reasoning_basis': reasoning_basis,
                            'reasoning_depth': depth + 1,
                            'selection_time': time.time()
                        })
                        
                        selected_concepts.append(concept)
                        
            except (ValueError, IndexError) as e:
                self.logger.warning(f"解析推理响应失败: {line}, 错误: {e}")
                continue
        
        # 按重要性分数排序
        selected_concepts.sort(
            key=lambda c: c.metadata.get('reasoning_importance', 0.0),
            reverse=True
        )
        
        return selected_concepts
    
    async def build_reasoning_trees_batch(
        self,
        instances: List[Dict[str, Any]],
        concepts_list: List[List[ConceptInfo]]
    ) -> List[ReasoningTree]:
        """批量构建推理树"""
        
        if len(instances) != len(concepts_list):
            raise ValueError("实例数量与概念列表数量不匹配")
        
        # 创建异步任务
        tasks = []
        for instance, concepts in zip(instances, concepts_list):
            task = self.build_reasoning_tree(instance, concepts)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        reasoning_trees = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"批量推理树构建第{i}项失败: {result}")
                # 创建空推理树
                empty_tree = ReasoningTree(
                    tree_id=str(uuid.uuid4()),
                    root_entity=f"failed_entity_{i}"
                )
                reasoning_trees.append(empty_tree)
            else:
                reasoning_trees.append(result)
        
        return reasoning_trees
    
    def get_tree_statistics(self, tree: ReasoningTree) -> Dict[str, Any]:
        """获取推理树统计信息"""
        
        paths = tree.get_paths_to_leaves()
        
        # 路径长度分布
        path_lengths = [len(path) for path in paths]
        
        # 概念类型分布
        concept_types = {}
        for node in tree.nodes.values():
            if node.concept:
                concept_name = node.concept.name
                concept_types[concept_name] = concept_types.get(concept_name, 0) + 1
        
        return {
            'total_nodes': tree.total_nodes,
            'max_depth': tree.max_depth,
            'num_paths': len(paths),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'concept_type_distribution': concept_types,
            'tree_id': tree.tree_id,
            'root_entity': tree.root_entity
        }


class DualAttentionMechanism(nn.Module):
    """双重注意力机制"""
    
    def __init__(self, config: RCAAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.alpha = config.attention_alpha
        
        # 语义对齐注意力
        self.semantic_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 隐式重要性注意力
        self.importance_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 投影层
        self.concept_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.relation_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        concepts: torch.Tensor,
        relation_instance: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            concepts: 概念嵌入 [batch_size, num_concepts, hidden_size]
            relation_instance: 关系实例嵌入 [batch_size, hidden_size]
            concept_mask: 概念掩码 [batch_size, num_concepts]
            
        Returns:
            final_weights: 最终注意力权重 [batch_size, num_concepts]
            attention_details: 注意力详情字典
        """
        
        batch_size, num_concepts, hidden_size = concepts.shape
        
        # 投影
        projected_concepts = self.concept_projection(concepts)
        projected_relation = self.relation_projection(relation_instance)
        
        # 扩展关系实例维度用于注意力计算
        relation_expanded = projected_relation.unsqueeze(1).expand(-1, num_concepts, -1)
        
        # 1. 语义对齐注意力
        semantic_query = projected_relation.unsqueeze(1)  # [batch_size, 1, hidden_size]
        semantic_weights, _ = self.semantic_attention(
            query=semantic_query,
            key=projected_concepts,
            value=projected_concepts,
            key_padding_mask=~concept_mask if concept_mask is not None else None
        )
        semantic_weights = semantic_weights.squeeze(1)  # [batch_size, num_concepts]
        
        # 2. 隐式重要性注意力
        importance_weights, _ = self.importance_attention(
            query=projected_concepts,
            key=projected_concepts,
            value=projected_concepts,
            key_padding_mask=~concept_mask if concept_mask is not None else None
        )
        # 取对角线元素作为自注意力权重
        importance_weights = torch.diagonal(importance_weights, dim1=-2, dim2=-1)
        
        # 3. 加权组合
        final_weights = (
            self.alpha * semantic_weights + 
            (1 - self.alpha) * importance_weights
        )
        
        # 应用掩码
        if concept_mask is not None:
            final_weights = final_weights.masked_fill(~concept_mask, float('-inf'))
        
        # Softmax归一化
        final_weights = F.softmax(final_weights, dim=-1)
        
        attention_details = {
            'semantic_weights': semantic_weights,
            'importance_weights': importance_weights,
            'final_weights': final_weights
        }
        
        return final_weights, attention_details
    
    def compute_attention_weighted_representation(
        self,
        concepts: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """计算注意力加权的概念表示
        
        Args:
            concepts: 概念嵌入 [batch_size, num_concepts, hidden_size]
            attention_weights: 注意力权重 [batch_size, num_concepts]
            
        Returns:
            weighted_representation: 加权表示 [batch_size, hidden_size]
        """
        
        # 扩展权重维度
        weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, num_concepts, 1]
        
        # 加权求和
        weighted_representation = torch.sum(
            concepts * weights_expanded, dim=1
        )  # [batch_size, hidden_size]
        
        return weighted_representation


class ImplicitInformativenessModeler:
    """隐式信息性建模器"""
    
    def __init__(self, config: RCAAConfig):
        self.config = config
        self.concept_stats = {}  # 概念统计信息
        self.classification_history = {}  # 分类历史
        self.logger = logging.getLogger(__name__)
    
    def compute_informativeness(
        self,
        concept: ConceptInfo,
        reasoning_chains: List[List[ConceptInfo]],
        target_relation: Optional[str] = None
    ) -> float:
        """计算概念的隐式信息性"""
        
        concept_id = concept.id
        
        try:
            # 1. 计算分类增益
            classification_gain = self._compute_classification_gain(
                concept_id, reasoning_chains, target_relation
            )
            
            # 2. 计算语义相似度增益
            similarity_gain = self._compute_similarity_gain(
                concept, reasoning_chains
            )
            
            # 3. 加权组合
            informativeness = (
                self.config.classification_weight * classification_gain +
                self.config.similarity_weight * similarity_gain
            )
            
            # 4. 更新统计信息
            self._update_concept_stats(concept_id, informativeness)
            
            return min(max(informativeness, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"计算信息性失败: {e}")
            return 0.5  # 返回默认值
    
    def _compute_classification_gain(
        self,
        concept_id: str,
        reasoning_chains: List[List[ConceptInfo]],
        target_relation: Optional[str]
    ) -> float:
        """计算分类增益"""
        
        if not reasoning_chains:
            return 0.0
        
        # 统计包含该概念的推理链
        chains_with_concept = []
        chains_without_concept = []
        
        for chain in reasoning_chains:
            concept_ids = [c.id for c in chain]
            if concept_id in concept_ids:
                chains_with_concept.append(chain)
            else:
                chains_without_concept.append(chain)
        
        if not chains_with_concept:
            return 0.0
        
        # 计算包含该概念的链的平均相关性
        with_concept_relevance = self._compute_chain_relevance(
            chains_with_concept, target_relation
        )
        
        # 计算不包含该概念的链的平均相关性
        without_concept_relevance = self._compute_chain_relevance(
            chains_without_concept, target_relation
        )
        
        # 分类增益 = 包含概念的链的相关性 - 不包含概念的链的相关性
        classification_gain = with_concept_relevance - without_concept_relevance
        
        return max(classification_gain, 0.0)
    
    def _compute_similarity_gain(
        self,
        concept: ConceptInfo,
        reasoning_chains: List[List[ConceptInfo]]
    ) -> float:
        """计算语义相似度增益"""
        
        if not reasoning_chains:
            return 0.0
        
        # 计算概念与所有推理链的平均相似度
        total_similarity = 0.0
        total_comparisons = 0
        
        for chain in reasoning_chains:
            for other_concept in chain:
                if other_concept.id != concept.id:
                    similarity = self._compute_concept_similarity(concept, other_concept)
                    total_similarity += similarity
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        avg_similarity = total_similarity / total_comparisons
        
        # 相似度增益基于平均相似度，但不是线性关系
        # 中等相似度的概念可能更有信息性
        if 0.3 <= avg_similarity <= 0.7:
            similarity_gain = avg_similarity * 1.5  # 提升中等相似度概念的权重
        else:
            similarity_gain = avg_similarity
        
        return min(similarity_gain, 1.0)
    
    def _compute_chain_relevance(
        self,
        chains: List[List[ConceptInfo]],
        target_relation: Optional[str]
    ) -> float:
        """计算推理链的相关性"""
        
        if not chains:
            return 0.0
        
        total_relevance = 0.0
        
        for chain in chains:
            # 链的相关性 = 链中所有概念的平均相关性分数
            chain_relevance = sum(c.relevance_score for c in chain) / len(chain)
            total_relevance += chain_relevance
        
        return total_relevance / len(chains)
    
    def _compute_concept_similarity(
        self,
        concept1: ConceptInfo,
        concept2: ConceptInfo
    ) -> float:
        """计算两个概念的相似度"""
        
        # 如果有嵌入，使用余弦相似度
        if concept1.embedding is not None and concept2.embedding is not None:
            try:
                emb1 = concept1.embedding.flatten()
                emb2 = concept2.embedding.flatten()
                similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                return similarity.item()
            except Exception:
                pass
        
        # 否则使用名称相似度
        return self._compute_name_similarity(concept1.name, concept2.name)
    
    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_concept_stats(self, concept_id: str, informativeness: float):
        """更新概念统计信息"""
        
        if concept_id not in self.concept_stats:
            self.concept_stats[concept_id] = {
                'count': 0,
                'total_informativeness': 0.0,
                'avg_informativeness': 0.0,
                'last_update': time.time()
            }
        
        stats = self.concept_stats[concept_id]
        stats['count'] += 1
        stats['total_informativeness'] += informativeness
        stats['avg_informativeness'] = stats['total_informativeness'] / stats['count']
        stats['last_update'] = time.time()
    
    def get_concept_statistics(self) -> Dict[str, Any]:
        """获取概念统计信息"""
        
        if not self.concept_stats:
            return {
                'total_concepts': 0,
                'avg_informativeness': 0.0,
                'informativeness_distribution': {}
            }
        
        # 计算总体统计
        all_informativeness = [
            stats['avg_informativeness'] for stats in self.concept_stats.values()
        ]
        
        # 信息性分布
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {}
        
        for low, high in ranges:
            count = sum(1 for info in all_informativeness if low <= info < high)
            distribution[f"{low}-{high}"] = count
        
        return {
            'total_concepts': len(self.concept_stats),
            'avg_informativeness': sum(all_informativeness) / len(all_informativeness),
            'informativeness_distribution': distribution,
            'max_informativeness': max(all_informativeness),
            'min_informativeness': min(all_informativeness)
        }
    
    def clear_statistics(self):
        """清除统计信息"""
        self.concept_stats.clear()
        self.classification_history.clear()