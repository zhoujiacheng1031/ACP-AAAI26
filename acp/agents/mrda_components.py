# -*- coding: utf-8 -*-
"""
Meta-Relation Discovery Agent (MRDA) components
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .state import ConceptInfo, MetaRelation, ProcessingResult
from .llm_service import BaseLLMService, LLMRequest
from .config import MRDAConfig


@dataclass
class RelationMiningStage:
    """关系挖掘阶段信息"""
    stage_id: int
    query_template: str
    focus_aspect: str
    expected_relations: int


class FineGrainedMetaRelationMiner:
    """细粒度元关系挖掘器"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: MRDAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 定义挖掘阶段
        self.mining_stages = self._init_mining_stages()
    
    def _init_mining_stages(self) -> List[RelationMiningStage]:
        """初始化挖掘阶段"""
        return [
            RelationMiningStage(
                stage_id=1,
                query_template="""分析以下两组概念之间的直接语义关系：

头实体概念：{head_concepts}
尾实体概念：{tail_concepts}
上下文：{context}

请识别概念间的直接关系，重点关注：
1. 语义相似性关系
2. 上下位关系
3. 部分-整体关系

输出格式（每行一个关系）：
关系类型|关系描述|置信度(0-1)|支持证据""",
                focus_aspect="direct_semantic_relations",
                expected_relations=5
            ),
            
            RelationMiningStage(
                stage_id=2,
                query_template="""基于第一阶段发现的关系，进一步分析概念间的间接和隐含关系：

头实体概念：{head_concepts}
尾实体概念：{tail_concepts}
上下文：{context}
已发现关系：{previous_relations}

请识别更深层的关系，重点关注：
1. 功能性关系
2. 因果关系
3. 时空关系
4. 属性关系

输出格式（每行一个关系）：
关系类型|关系描述|置信度(0-1)|支持证据""",
                focus_aspect="indirect_implicit_relations",
                expected_relations=3
            ),
            
            RelationMiningStage(
                stage_id=3,
                query_template="""综合前两阶段的发现，识别概念间的复合和抽象关系：

头实体概念：{head_concepts}
尾实体概念：{tail_concepts}
上下文：{context}
第一阶段关系：{stage1_relations}
第二阶段关系：{stage2_relations}

请识别高层次的关系模式，重点关注：
1. 复合关系（多个简单关系的组合）
2. 抽象关系（概念化的关系）
3. 领域特定关系
4. 上下文相关关系

输出格式（每行一个关系）：
关系类型|关系描述|置信度(0-1)|支持证据""",
                focus_aspect="composite_abstract_relations",
                expected_relations=2
            )
        ]
    
    async def mine_relations(
        self,
        head_concepts: List[ConceptInfo],
        tail_concepts: List[ConceptInfo],
        context: str
    ) -> List[MetaRelation]:
        """执行多阶段关系挖掘"""
        
        start_time = time.time()
        all_relations = []
        stage_results = {}
        
        try:
            self.logger.info(f"开始{len(self.mining_stages)}阶段关系挖掘")
            
            for stage in self.mining_stages:
                self.logger.debug(f"执行第{stage.stage_id}阶段挖掘: {stage.focus_aspect}")
                
                # 创建查询
                query = self._create_stage_query(
                    stage, head_concepts, tail_concepts, context, stage_results
                )
                
                # 执行LLM查询
                response = await self.llm_service.query(
                    prompt=query,
                    task_type="meta_relation_mining",
                    temperature=self.config.mining_temperature,
                    max_tokens=self.config.max_relations_per_stage * 100
                )
                
                # 解析关系
                stage_relations = self._parse_relations(
                    response.content, stage.stage_id, context
                )
                
                # 限制关系数量
                stage_relations = stage_relations[:self.config.max_relations_per_stage]
                
                stage_results[f"stage{stage.stage_id}"] = stage_relations
                all_relations.extend(stage_relations)
                
                self.logger.debug(
                    f"第{stage.stage_id}阶段完成，发现{len(stage_relations)}个关系"
                )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"关系挖掘完成 - 总关系数: {len(all_relations)}, "
                f"处理时间: {processing_time:.2f}s"
            )
            
            return all_relations
            
        except Exception as e:
            error_msg = f"关系挖掘失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def _create_stage_query(
        self,
        stage: RelationMiningStage,
        head_concepts: List[ConceptInfo],
        tail_concepts: List[ConceptInfo],
        context: str,
        stage_results: Dict[str, List[MetaRelation]]
    ) -> str:
        """创建阶段查询"""
        
        # 格式化概念列表
        head_concept_names = [c.name for c in head_concepts]
        tail_concept_names = [c.name for c in tail_concepts]
        
        # 准备查询参数
        query_params = {
            'head_concepts': ', '.join(head_concept_names),
            'tail_concepts': ', '.join(tail_concept_names),
            'context': context
        }
        
        # 添加前一阶段的结果
        if stage.stage_id > 1:
            previous_relations = []
            for i in range(1, stage.stage_id):
                stage_key = f"stage{i}"
                if stage_key in stage_results:
                    for rel in stage_results[stage_key]:
                        previous_relations.append(f"{rel.type}: {rel.description}")
            
            query_params['previous_relations'] = '\n'.join(previous_relations)
        
        # 为第3阶段添加特殊参数
        if stage.stage_id == 3:
            query_params['stage1_relations'] = self._format_stage_relations(
                stage_results.get('stage1', [])
            )
            query_params['stage2_relations'] = self._format_stage_relations(
                stage_results.get('stage2', [])
            )
        
        return stage.query_template.format(**query_params)
    
    def _format_stage_relations(self, relations: List[MetaRelation]) -> str:
        """格式化阶段关系"""
        if not relations:
            return "无"
        
        formatted = []
        for rel in relations:
            formatted.append(f"- {rel.type}: {rel.description} (置信度: {rel.confidence:.2f})")
        
        return '\n'.join(formatted)
    
    def _parse_relations(
        self,
        response_content: str,
        stage_id: int,
        context: str
    ) -> List[MetaRelation]:
        """解析LLM响应中的关系"""
        
        relations = []
        lines = response_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not '|' in line:
                continue
            
            try:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    relation_type = parts[0]
                    description = parts[1]
                    confidence = float(parts[2])
                    evidence = parts[3]
                    
                    # 创建MetaRelation对象
                    meta_relation = MetaRelation(
                        type=relation_type,
                        description=description,
                        confidence=confidence,
                        evidence=evidence,
                        stage=stage_id,
                        metadata={
                            'context': context,
                            'mining_stage': stage_id,
                            'discovery_time': time.time()
                        }
                    )
                    
                    relations.append(meta_relation)
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"解析关系失败: {line}, 错误: {e}")
                continue
        
        return relations
    
    async def mine_relations_batch(
        self,
        concept_pairs: List[Tuple[List[ConceptInfo], List[ConceptInfo]]],
        contexts: List[str]
    ) -> List[List[MetaRelation]]:
        """批量挖掘关系"""
        
        if len(concept_pairs) != len(contexts):
            raise ValueError("概念对数量与上下文数量不匹配")
        
        # 创建异步任务
        tasks = []
        for (head_concepts, tail_concepts), context in zip(concept_pairs, contexts):
            task = self.mine_relations(head_concepts, tail_concepts, context)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"批量挖掘第{i}项失败: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_mining_statistics(
        self,
        relations: List[MetaRelation]
    ) -> Dict[str, Any]:
        """获取挖掘统计信息"""
        
        if not relations:
            return {
                'total_relations': 0,
                'stage_distribution': {},
                'avg_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        # 阶段分布
        stage_distribution = {}
        confidences = []
        
        for relation in relations:
            stage = relation.stage
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            confidences.append(relation.confidence)
        
        # 置信度分布
        confidence_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        confidence_distribution = {}
        
        for low, high in confidence_ranges:
            count = sum(1 for conf in confidences if low <= conf < high)
            confidence_distribution[f"{low}-{high}"] = count
        
        return {
            'total_relations': len(relations),
            'stage_distribution': stage_distribution,
            'avg_confidence': sum(confidences) / len(confidences),
            'confidence_distribution': confidence_distribution,
            'max_confidence': max(confidences),
            'min_confidence': min(confidences)
        }


class SemanticConsistencyVerifier:
    """语义一致性验证器"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: MRDAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def verify_relations(
        self,
        relations: List[MetaRelation],
        head_entity: str,
        tail_entity: str,
        context: str
    ) -> List[MetaRelation]:
        """验证关系的语义一致性"""
        
        if not relations:
            return []
        
        start_time = time.time()
        verified_relations = []
        
        try:
            self.logger.debug(f"开始验证{len(relations)}个关系")
            
            # 批量验证
            verification_tasks = []
            for relation in relations:
                task = self._verify_single_relation(
                    relation, head_entity, tail_entity, context
                )
                verification_tasks.append(task)
            
            # 并发执行验证
            verification_results = await asyncio.gather(
                *verification_tasks, return_exceptions=True
            )
            
            # 处理验证结果
            for relation, result in zip(relations, verification_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"关系验证失败: {relation.type}, 错误: {result}")
                    continue
                
                verification_score, verification_reason = result
                
                # 计算最终分数
                final_score = self._compute_semantic_score(
                    relation, head_entity, tail_entity, verification_score
                )
                
                # 检查是否通过阈值
                if final_score >= self.config.verification_threshold:
                    # 更新关系信息
                    relation.verification_score = final_score
                    if relation.metadata is None:
                        relation.metadata = {}
                    relation.metadata.update({
                        'verification_reason': verification_reason,
                        'verification_time': time.time()
                    })
                    
                    verified_relations.append(relation)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"关系验证完成 - 输入: {len(relations)}, "
                f"通过: {len(verified_relations)}, "
                f"处理时间: {processing_time:.2f}s"
            )
            
            return verified_relations
            
        except Exception as e:
            error_msg = f"关系验证失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    async def _verify_single_relation(
        self,
        relation: MetaRelation,
        head_entity: str,
        tail_entity: str,
        context: str
    ) -> Tuple[float, str]:
        """验证单个关系"""
        
        verification_prompt = f"""请验证以下关系的语义一致性：

关系类型：{relation.type}
关系描述：{relation.description}
头实体：{head_entity}
尾实体：{tail_entity}
上下文：{context}
原始置信度：{relation.confidence}
支持证据：{relation.evidence}

请评估这个关系是否在给定上下文中语义一致和合理。

评估标准：
1. 关系是否符合常识
2. 关系是否与上下文相符
3. 关系是否在实体间成立
4. 关系描述是否准确

输出格式：
验证结果|置信度(0-1)|验证理由"""
        
        try:
            response = await self.llm_service.query(
                prompt=verification_prompt,
                task_type="semantic_verification",
                temperature=0.1  # 使用较低温度确保一致性
            )
            
            # 解析验证结果
            return self._parse_verification_response(response.content)
            
        except Exception as e:
            self.logger.error(f"单个关系验证失败: {e}")
            return 0.0, f"验证失败: {str(e)}"
    
    def _parse_verification_response(self, response_content: str) -> Tuple[float, str]:
        """解析验证响应"""
        
        lines = response_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if '|' in line:
                try:
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) >= 3:
                        result = parts[0].lower()
                        confidence = float(parts[1])
                        reason = parts[2]
                        
                        # 根据验证结果调整置信度
                        if result in ['通过', 'pass', '是', 'yes', 'true']:
                            return confidence, reason
                        else:
                            return 0.0, reason
                            
                except (ValueError, IndexError):
                    continue
        
        # 如果解析失败，返回默认值
        return 0.5, "解析验证结果失败"
    
    def _compute_semantic_score(
        self,
        relation: MetaRelation,
        head_entity: str,
        tail_entity: str,
        verification_score: float
    ) -> float:
        """计算语义分数"""
        
        # 基础分数来自原始置信度
        base_confidence = relation.confidence
        
        # 计算注意力分数（简化实现）
        attention_score = self._compute_attention_score(
            relation, head_entity, tail_entity
        )
        
        # 加权组合
        final_score = (
            self.config.attention_weight * attention_score +
            self.config.confidence_weight * verification_score +
            (1 - self.config.attention_weight - self.config.confidence_weight) * base_confidence
        )
        
        return min(max(final_score, 0.0), 1.0)
    
    def _compute_attention_score(
        self,
        relation: MetaRelation,
        head_entity: str,
        tail_entity: str
    ) -> float:
        """计算注意力分数（简化实现）"""
        
        # 基于关系类型和实体名称的简单相似度计算
        relation_words = set(relation.type.lower().split())
        description_words = set(relation.description.lower().split())
        head_words = set(head_entity.lower().split())
        tail_words = set(tail_entity.lower().split())
        
        # 计算词汇重叠
        all_relation_words = relation_words.union(description_words)
        all_entity_words = head_words.union(tail_words)
        
        if not all_relation_words or not all_entity_words:
            return 0.5
        
        intersection = all_relation_words.intersection(all_entity_words)
        union = all_relation_words.union(all_entity_words)
        
        return len(intersection) / len(union) if union else 0.0


class ContrastiveNegativeSampler:
    """对比负样本生成器"""
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        config: MRDAConfig
    ):
        self.llm_service = llm_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def generate_negatives(
        self,
        positive_relations: List[MetaRelation],
        context: str,
        difficulty: Optional[float] = None
    ) -> List[MetaRelation]:
        """生成对比负样本"""
        
        if not positive_relations:
            return []
        
        difficulty = difficulty or self.config.negative_difficulty
        start_time = time.time()
        
        try:
            self.logger.debug(f"开始生成{len(positive_relations)}个正样本的负样本")
            
            # 计算需要生成的负样本数量
            num_negatives = max(1, int(len(positive_relations) * self.config.negative_ratio))
            
            # 选择要生成负样本的正样本
            selected_positives = positive_relations[:num_negatives]
            
            # 批量生成负样本
            negative_tasks = []
            for positive in selected_positives:
                task = self._generate_single_negative(positive, context, difficulty)
                negative_tasks.append(task)
            
            # 并发执行
            negative_results = await asyncio.gather(
                *negative_tasks, return_exceptions=True
            )
            
            # 处理结果
            negative_relations = []
            for positive, result in zip(selected_positives, negative_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"负样本生成失败: {positive.type}, 错误: {result}")
                    continue
                
                if result:
                    negative_relations.extend(result)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"负样本生成完成 - 正样本: {len(selected_positives)}, "
                f"负样本: {len(negative_relations)}, "
                f"处理时间: {processing_time:.2f}s"
            )
            
            return negative_relations
            
        except Exception as e:
            error_msg = f"负样本生成失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    async def _generate_single_negative(
        self,
        positive_relation: MetaRelation,
        context: str,
        difficulty: float
    ) -> List[MetaRelation]:
        """生成单个正样本的负样本"""
        
        # 根据难度确定相似度范围
        min_sim, max_sim = self.config.similarity_range
        target_similarity = min_sim + (max_sim - min_sim) * difficulty
        
        generation_prompt = f"""基于以下正样本关系，生成语义相似但不同的负样本关系：

正样本关系：
- 类型：{positive_relation.type}
- 描述：{positive_relation.description}
- 置信度：{positive_relation.confidence}
- 证据：{positive_relation.evidence}

上下文：{context}

生成要求：
1. 负样本应与正样本在语义上相似（相似度约{target_similarity:.1f}）
2. 但关系本质上应该是不同的或错误的
3. 生成2-3个不同类型的负样本

负样本类型：
1. 反向关系（关系方向相反）
2. 相似关系（语义相近但不同的关系）
3. 错误关系（看似合理但实际错误的关系）

输出格式（每行一个负样本）：
负样本类型|关系描述|相似度(0-1)|生成理由"""
        
        try:
            response = await self.llm_service.query(
                prompt=generation_prompt,
                task_type="negative_generation",
                temperature=self.config.mining_temperature
            )
            
            # 解析负样本
            return self._parse_negative_samples(
                response.content, positive_relation, context
            )
            
        except Exception as e:
            self.logger.error(f"单个负样本生成失败: {e}")
            return []
    
    def _parse_negative_samples(
        self,
        response_content: str,
        positive_relation: MetaRelation,
        context: str
    ) -> List[MetaRelation]:
        """解析负样本"""
        
        negative_relations = []
        lines = response_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            try:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    negative_type = parts[0]
                    description = parts[1]
                    similarity = float(parts[2])
                    reason = parts[3]
                    
                    # 检查相似度是否在期望范围内
                    min_sim, max_sim = self.config.similarity_range
                    if not (min_sim <= similarity <= max_sim):
                        continue
                    
                    # 创建负样本关系
                    negative_relation = MetaRelation(
                        type=f"NEG_{negative_type}",
                        description=description,
                        confidence=1.0 - similarity,  # 负样本置信度与相似度相反
                        evidence=f"负样本生成: {reason}",
                        verification_score=0.0,  # 负样本验证分数为0
                        stage=positive_relation.stage,
                        metadata={
                            'is_negative': True,
                            'positive_source': positive_relation.type,
                            'similarity_to_positive': similarity,
                            'generation_reason': reason,
                            'context': context,
                            'generation_time': time.time()
                        }
                    )
                    
                    negative_relations.append(negative_relation)
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"解析负样本失败: {line}, 错误: {e}")
                continue
        
        return negative_relations