# -*- coding: utf-8 -*-
"""
Created on 2024-07-31

@author: zhoujiacheng
@file: negative_sampler.py
@purpose: Generate negative samples using LLM with concept consistency
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import logging
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from .concept_retriever import ConceptRetriever
import requests
import random

class NegativeSampler:
    """负样本生成器"""
    
    def __init__(
        self,
        concept_retriever: ConceptRetriever,
        cache_dir: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150
    ):
        self.logger = logging.getLogger(__name__)
        self.concept_retriever = concept_retriever
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 加载环境变量
        load_dotenv()
        
        # API配置
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # 初始化缓存
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = self._load_cache()
        else:
            self.cache = {}
            
    def generate_negative_samples(
        self,
        instance: Dict,
        num_samples: int = 1,
        sample_type: str = 'all'
    ) -> List[Dict]:
        """生成负样本
        
        Args:
            instance: 原始实例
            num_samples: 每种类型生成的样本数量
            sample_type: 生成类型 ('same_domain_same_entity', 
                                'same_domain_diff_entity',
                                'diff_domain' or 'all')
                                
        Returns:
            负样本列表
        """
        # 获取头尾实体的概念
        h_concepts = self.concept_retriever.get_multi_hop_concepts(
            instance['h']['name']
        )
        t_concepts = self.concept_retriever.get_multi_hop_concepts(
            instance['t']['name']
        )
        
        negative_samples = []
        
        if sample_type in ['same_domain_same_entity', 'all']:
            # 生成同域同实体负样本
            samples = self._generate_same_domain_same_entity(
                instance, h_concepts, t_concepts, num_samples
            )
            negative_samples.extend(samples)
            
        if sample_type in ['same_domain_diff_entity', 'all']:
            # 生成同域不同实体负样本
            samples = self._generate_same_domain_diff_entity(
                instance, h_concepts, t_concepts, num_samples
            )
            negative_samples.extend(samples)
            
        if sample_type in ['diff_domain', 'all']:
            # 生成不同域负样本
            samples = self._generate_diff_domain(
                instance, h_concepts, t_concepts, num_samples
            )
            negative_samples.extend(samples)
            
        return negative_samples
        
    def _generate_same_domain_same_entity(
        self,
        instance: Dict,
        h_concepts: List[Dict],
        t_concepts: List[Dict],
        num_samples: int
    ) -> List[Dict]:
        """生成同域同实体负样本"""
        prompt = self._create_prompt(
            instance,
            h_concepts,
            t_concepts,
            "same_domain_same_entity"
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个关系生成助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_samples
            )
            
            samples = []
            for choice in response.choices:
                try:
                    sample = self._parse_llm_response(
                        choice.message.content,
                        instance
                    )
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    self.logger.error(f"解析LLM响应时出错: {str(e)}")
                    continue
                    
            return samples
            
        except Exception as e:
            self.logger.error(f"调用LLM API时出错: {str(e)}")
            return []
            
    def _generate_same_domain_diff_entity(
        self,
        instance: Dict,
        h_concepts: List[Dict],
        t_concepts: List[Dict],
        num_samples: int
    ) -> List[Dict]:
        """生成同域不同实体负样本"""
        prompt = self._create_prompt(
            instance,
            h_concepts,
            t_concepts,
            "same_domain_diff_entity"
        )
        
        try:
            # 获取相似概念的实体
            similar_h_entities = self._get_similar_entities(
                instance['h']['name'],
                h_concepts,
                num_samples
            )
            similar_t_entities = self._get_similar_entities(
                instance['t']['name'],
                t_concepts,
                num_samples
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个关系生成助手。"},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"可以使用的头实体候选: {', '.join(similar_h_entities)}"},
                    {"role": "assistant", "content": f"可以使用的尾实体候选: {', '.join(similar_t_entities)}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_samples
            )
            
            samples = []
            for choice in response.choices:
                try:
                    sample = self._parse_llm_response(
                        choice.message.content,
                        instance,
                        similar_entities={
                            'head': similar_h_entities,
                            'tail': similar_t_entities
                        }
                    )
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    self.logger.error(f"解析LLM响应时出错: {str(e)}")
                    continue
                    
            return samples
            
        except Exception as e:
            self.logger.error(f"调用LLM API时出错: {str(e)}")
            return []
        
    def _generate_diff_domain(
        self,
        instance: Dict,
        h_concepts: List[Dict],
        t_concepts: List[Dict],
        num_samples: int
    ) -> List[Dict]:
        """生成不同域负样本"""
        prompt = self._create_prompt(
            instance,
            h_concepts,
            t_concepts,
            "diff_domain"
        )
        
        try:
            # 获取不同域的概念
            diff_domain_concepts = self._get_diff_domain_concepts(
                h_concepts + t_concepts
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个关系生成助手。"},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"可以使用的不同域概念: {', '.join(diff_domain_concepts)}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_samples
            )
            
            samples = []
            for choice in response.choices:
                try:
                    sample = self._parse_llm_response(
                        choice.message.content,
                        instance,
                        diff_domain_concepts=diff_domain_concepts
                    )
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    self.logger.error(f"解析LLM响应时出错: {str(e)}")
                    continue
                    
            return samples
            
        except Exception as e:
            self.logger.error(f"调用LLM API时出错: {str(e)}")
            return []
        
    def _create_prompt(
        self,
        instance: Dict,
        h_concepts: List[Dict],
        t_concepts: List[Dict],
        sample_type: str
    ) -> str:
        """创建LLM提示"""
        context = " ".join(instance['token'])
        h_concept_str = ", ".join([c['name'] for c in h_concepts[:3]])
        t_concept_str = ", ".join([c['name'] for c in t_concepts[:3]])
        
        if sample_type == "same_domain_same_entity":
            prompt = f"""请生成一个负样本句子,要求:
            1. 使用相同的头实体"{instance['h']['name']}"(概念:{h_concept_str})
            2. 使用相同的尾实体"{instance['t']['name']}"(概念:{t_concept_str})
            3. 表达与原句"{context}"不同的关系
            4. 保持在相同的领域内
            5. 生成的句子要自然流畅
            """
        elif sample_type == "same_domain_diff_entity":
            prompt = f"""请生成一个负样本句子,要求:
            1. 使用不同但概念相似的实体替换原句中的头尾实体
            2. 头实体概念:{h_concept_str}
            3. 尾实体概念:{t_concept_str}
            4. 保持在相同的领域内
            5. 生成的句子要自然流畅
            """
        else:
            prompt = f"""请生成一个负样本句子,要求:
            1. 使用完全不同领域的实体
            2. 与原句"{context}"表达不同类型的关系
            3. 生成的句子要自然流畅
            """
            
        return prompt
        
    def _parse_llm_response(
        self,
        response: str,
        original_instance: Dict,
        similar_entities: Dict = None,
        diff_domain_concepts: List[str] = None
    ) -> Optional[Dict]:
        """改进的LLM响应解析"""
        try:
            # 分词并清理
            tokens = response.strip().split()
            
            # 创建新实例
            instance = original_instance.copy()
            instance['token'] = tokens
            instance['relation'] = 'no_relation'
            
            # 提取头尾实体位置
            h_pos = self._find_entity_position(tokens, instance['h']['name'])
            t_pos = self._find_entity_position(tokens, instance['t']['name'])
            
            if h_pos and t_pos:
                instance['h']['pos'] = h_pos
                instance['t']['pos'] = t_pos
                return instance
            
            return None
            
        except Exception as e:
            self.logger.error(f"解析响应失败: {str(e)}")
            return None
            
    def _find_entity_position(
        self,
        tokens: List[str],
        entity: str
    ) -> Optional[Tuple[int, int]]:
        """查找实体在句子中的位置"""
        entity_tokens = entity.split()
        n = len(tokens)
        m = len(entity_tokens)
        
        for i in range(n - m + 1):
            if tokens[i:i+m] == entity_tokens:
                return (i, i+m-1)
            
        return None
        
    def _load_cache(self) -> Dict:
        """加载缓存"""
        cache_file = os.path.join(self.cache_dir, 'negative_samples_cache.pt')
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        return {}
        
    def _save_cache(self):
        """保存缓存"""
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(self.cache_dir, 'negative_samples_cache.pt')
        torch.save(self.cache, cache_file)
        
    def _get_similar_entities(
        self,
        entity: str,
        concepts: List[Dict],
        num_samples: int
    ) -> List[str]:
        """获取概念相似的实体"""
        try:
            # 使用Neo4j查询相似概念的实体
            concept_names = [c['name'] for c in concepts[:3]]
            query = """
            MATCH (e:Entity)-[r:IS_A]->(c:Concept)
            WHERE c.name IN $concepts AND e.name <> $entity
            RETURN DISTINCT e.name as entity
            LIMIT $limit
            """
            
            results = self.concept_retriever.graph.run(
                query,
                concepts=concept_names,
                entity=entity,
                limit=num_samples * 2
            ).data()
            
            return [r['entity'] for r in results]
            
        except Exception as e:
            self.logger.error(f"获取相似实体时出错: {str(e)}")
            return []
        
    def _get_diff_domain_concepts(
        self,
        concepts: List[Dict]
    ) -> List[str]:
        """获取不同域的概念"""
        try:
            # 获取当前概念的顶层概念
            current_domains = set()
            for concept in concepts:
                hierarchy = self.concept_retriever.get_concept_hierarchy(
                    concept['name']
                )
                current_domains.update(hierarchy['super_concepts'])
            
            # 查询不同域的概念
            query = """
            MATCH (c:Concept)
            WHERE NOT c.name IN $domains
            RETURN c.name as concept
            LIMIT 10
            """
            
            results = self.concept_retriever.graph.run(
                query,
                domains=list(current_domains)
            ).data()
            
            return [r['concept'] for r in results]
            
        except Exception as e:
            self.logger.error(f"获取不同域概念时出错: {str(e)}")
            return [] 
        
    def sample_hard_negative(
        self,
        candidate_instances: List[Dict],
        support_instances: List[Dict],
        hardness: float = 0.5
    ) -> Dict:
        """根据概念嵌入/语义相似度采样难负样本，hardness控制难度(0-1)"""
        # 获取support的概念嵌入均值
        support_concepts = []
        for inst in support_instances:
            h_concepts = self.concept_retriever.get_multi_hop_concepts(inst['h']['name'])
            t_concepts = self.concept_retriever.get_multi_hop_concepts(inst['t']['name'])
            support_concepts.extend(h_concepts + t_concepts)
        if not support_concepts:
            return random.choice(candidate_instances)
        support_emb = self.concept_retriever.get_concept_embeddings(support_concepts).mean(dim=0, keepdim=True)  # [1, dim]
        # 计算每个候选的概念嵌入均值
        scores = []
        for inst in candidate_instances:
            h_concepts = self.concept_retriever.get_multi_hop_concepts(inst['h']['name'])
            t_concepts = self.concept_retriever.get_multi_hop_concepts(inst['t']['name'])
            all_concepts = h_concepts + t_concepts
            if not all_concepts:
                scores.append(-1.0)
                continue
            emb = self.concept_retriever.get_concept_embeddings(all_concepts).mean(dim=0, keepdim=True)
            sim = torch.nn.functional.cosine_similarity(emb, support_emb).item()
            scores.append(sim)
        # 按相似度排序，hardness=1选最难(最相似)，0选最易(最不相似)，0.5选中间
        sorted_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        pos = int(hardness * (len(sorted_idx) - 1))
        idx = sorted_idx[pos]
        return candidate_instances[idx] 