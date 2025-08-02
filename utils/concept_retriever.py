# -*- coding: utf-8 -*-
"""
Created on 2024-07-26

@author: zhoujiacheng
@file: concept_retriever.py
@purpose: Enhanced concept retriever with multi-hop capability
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from py2neo import Graph
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class ConceptRetriever:
    """增强版概念检索器"""
    
    def __init__(
        self,
        max_hops: int = 3,
        embedding_dim: int = 768,
        cache_dir: Optional[str] = None,
        min_confidence: float = 0.5,
        model_name: str = "../model/bert-base-uncased"
    ):
        self.logger = logging.getLogger(__name__)
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.min_confidence = min_confidence
        
        # 初始化BERT模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()  # 设置为评估模式
        
        # 加载环境变量
        load_dotenv()
        
        # 连接Neo4j
        self.graph = Graph(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        
        # 初始化缓存
        self.concept_cache = {}
        self.embedding_cache = {}
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()
            
    def get_concept_embedding(self, concept: str) -> torch.Tensor:
        """获取概念嵌入"""
        # 检查缓存
        if concept in self.embedding_cache:
            return self.embedding_cache[concept]
            
        # 使用BERT编码概念
        inputs = self.tokenizer(
            concept,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        
        # 移动到GPU(如果可用)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.encoder = self.encoder.cuda()
            
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # 使用[CLS]token的输出作为概念表示
            embedding = outputs.last_hidden_state[:, 0, :]
            
        # 移回CPU并转换为numpy
        embedding = embedding.cpu().numpy()
        
        # 更新缓存
        self.embedding_cache[concept] = embedding
        
        return embedding
        
    def get_multi_hop_concepts(
        self, 
        entity: str,
        hop: Optional[int] = None,
        topk: int = 10,
        score_threshold: float = 0.1
    ) -> List[Dict[str, any]]:
        """获取实体的多跳概念，支持topk和score筛选"""
        if hop is None:
            hop = self.max_hops
        cache_key = f"{entity}_{hop}"
        if cache_key in self.concept_cache:
            concepts = self.concept_cache[cache_key]
        else:
        query = """
            MATCH (e:Entity {name: $entity})-[r:IS_A*1..$hop]->(c:Concept)
        RETURN c.name as name, length(r) as hops
        """
        results = self.graph.run(
            query,
            entity=entity,
                hop=hop
        ).data()
        concepts = []
        for result in results:
            concept_name = result['name']
            embedding = self.get_concept_embedding(concept_name)
                # 置信度和score综合
                confidence = 1.0 / (1 + result['hops'])
                score = float(confidence)
            concepts.append({
                'name': concept_name,
                'embedding': embedding,
                'hops': result['hops'],
                    'confidence': confidence,
                    'score': score
            })
        self.concept_cache[cache_key] = concepts
        # 按score筛选
        filtered = [c for c in concepts if c['score'] >= score_threshold]
        filtered = sorted(filtered, key=lambda x: -x['score'])[:topk]
        return filtered
        
    def get_concept_embeddings(
        self,
        concepts: List[Dict[str, any]]
    ) -> torch.Tensor:
        """获取概念嵌入，返回float32张量"""
        embeddings = []
        for concept in concepts:
            concept_name = concept['name']
            if concept_name in self.embedding_cache:
                embedding = self.embedding_cache[concept_name]
            else:
                query = """
                MATCH (c:Concept {name: $concept})
                RETURN c.embedding as embedding
                """
                result = self.graph.run(
                    query,
                    concept=concept_name
                ).data()
                if result:
                    embedding = np.array(result[0]['embedding'])
                    self.embedding_cache[concept_name] = embedding
                else:
                    embedding = np.zeros(self.embedding_dim)
            embeddings.append(embedding)
        if not embeddings:
            return torch.zeros(1, self.embedding_dim, dtype=torch.float32)
        return torch.tensor(np.stack(embeddings), dtype=torch.float32)
        
    def _load_cache(self):
        """从文件加载缓存"""
        concept_cache_file = os.path.join(self.cache_dir, 'concept_cache.pt')
        embedding_cache_file = os.path.join(self.cache_dir, 'embedding_cache.pt')
        
        if os.path.exists(concept_cache_file):
            self.concept_cache = torch.load(concept_cache_file)
            
        if os.path.exists(embedding_cache_file):
            self.embedding_cache = torch.load(embedding_cache_file)
            
    def _save_cache(self):
        """保存缓存到文件"""
        if not self.cache_dir:
            return
            
        concept_cache_file = os.path.join(self.cache_dir, 'concept_cache.pt')
        embedding_cache_file = os.path.join(self.cache_dir, 'embedding_cache.pt')
        
        torch.save(self.concept_cache, concept_cache_file)
        torch.save(self.embedding_cache, embedding_cache_file)

    def get_batch_concepts(
        self,
        entities: List[str],
        max_hops: Optional[int] = None,
        batch_size: int = 32
    ) -> Dict[str, List[Dict[str, any]]]:
        """批量获取实体的概念
        
        Args:
            entities: 实体名称列表
            max_hops: 最大跳数
            batch_size: 批处理大小
            
        Returns:
            实体到概念的映射字典
        """
        results = {}
        for i in tqdm(range(0, len(entities), batch_size)):
            batch = entities[i:i + batch_size]
            for entity in batch:
                results[entity] = self.get_multi_hop_concepts(entity, max_hops)
        return results

    def compute_concept_similarity(
        self,
        concept1: str,
        concept2: str
    ) -> float:
        """计算两个概念的相似度
        
        Args:
            concept1: 第一个概念
            concept2: 第二个概念
            
        Returns:
            相似度分数 (0-1)
        """
        # 获取概念嵌入
        emb1 = self.get_concept_embeddings([{'name': concept1}])
        emb2 = self.get_concept_embeddings([{'name': concept2}])
        
        # 计算余弦相似度
        sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        return sim.item()

    def filter_concepts(
        self,
        concepts: List[Dict[str, any]],
        min_confidence: Optional[float] = None,
        max_hops: Optional[int] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """按置信度、跳数、score筛选概念"""
        filtered = concepts
        if min_confidence is not None:
            filtered = [c for c in filtered if c['confidence'] >= min_confidence]
        if max_hops is not None:
            filtered = [c for c in filtered if c['hops'] <= max_hops]
        if min_score is not None:
            filtered = [c for c in filtered if c['score'] >= min_score]
        if top_k is not None:
            filtered = sorted(filtered, key=lambda x: -x['score'])[:top_k]
        return filtered

    def get_concept_hierarchy(
        self,
        concept: str
    ) -> Dict[str, List[str]]:
        """获取概念的层级结构
        
        Args:
            concept: 概念名称
            
        Returns:
            包含上位和下位概念的字典
        """
        try:
            # 获取上位概念
            super_query = """
            MATCH (c:Concept {name: $concept})-[r:IS_A*1..3]->(sc:Concept)
            RETURN sc.name as concept
            """
            super_concepts = [
                r['concept'] for r in 
                self.graph.run(super_query, concept=concept).data()
            ]
            
            # 获取下位概念
            sub_query = """
            MATCH (c:Concept {name: $concept})<-[r:IS_A*1..3]-(sc:Concept)
            RETURN sc.name as concept
            """
            sub_concepts = [
                r['concept'] for r in
                self.graph.run(sub_query, concept=concept).data()
            ]
            
            return {
                'super_concepts': super_concepts,
                'sub_concepts': sub_concepts
            }
            
        except Exception as e:
            self.logger.error(f"获取概念层级时出错: {str(e)}")
            return {'super_concepts': [], 'sub_concepts': []} 