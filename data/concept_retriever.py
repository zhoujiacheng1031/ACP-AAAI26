# -*- coding: utf-8 -*-
"""
@purpose: Retrieve and process entity concepts from Neo4j
"""

import os
from typing import Dict, List, Optional
import torch
import numpy as np
from py2neo import Graph
from dotenv import load_dotenv
import logging

class ConceptRetriever:
    """概念检索器"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_concepts: int = 5,
        cache_dir: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = embedding_dim
        self.max_concepts = max_concepts
        
        # 加载环境变量
        load_dotenv()
        
        # 连接Neo4j
        self.graph = Graph(
            os.getenv('NEO4J_URI'),
            auth=(
                os.getenv('NEO4J_USER'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        
        # 初始化缓存
        self.concept_cache = {}
        self.embedding_cache = {}
        
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()
            
    def get_concepts(self, entity: str) -> List[str]:
        """获取实体的相关概念
        
        Args:
            entity: 实体名称
            
        Returns:
            概念列表
        """
        # 检查缓存
        if entity in self.concept_cache:
            return self.concept_cache[entity]
            
        # 查询Neo4j
        query = """
        MATCH (e:Entity {name: $entity})-[r:HAS_CONCEPT]->(c:Concept)
        RETURN c.name AS concept, r.weight AS weight
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        
        results = self.graph.run(
            query,
            entity=entity,
            limit=self.max_concepts
        )
        
        concepts = [record['concept'] for record in results]
        
        # 更新缓存
        self.concept_cache[entity] = concepts
        
        if len(self.concept_cache) % 1000 == 0:
            self._save_cache()
            
        return concepts
        
    def get_embeddings(self, concepts: List[str]) -> torch.Tensor:
        """获取概念的嵌入表示
        
        Args:
            concepts: 概念列表
            
        Returns:
            概念嵌入张量 [num_concepts, embedding_dim]
        """
        embeddings = []
        
        for concept in concepts:
            # 检查缓存
            if concept in self.embedding_cache:
                embedding = self.embedding_cache[concept]
            else:
                # 查询Neo4j
                query = """
                MATCH (c:Concept {name: $concept})
                RETURN c.embedding AS embedding
                """
                
                result = self.graph.run(
                    query,
                    concept=concept
                ).data()
                
                if result:
                    embedding = np.array(result[0]['embedding'])
                    self.embedding_cache[concept] = embedding
                else:
                    # 如果找不到嵌入,使用零向量
                    embedding = np.zeros(self.embedding_dim)
                    
            embeddings.append(embedding)
            
        if not embeddings:
            # 如果没有概念,返回零向量
            return torch.zeros(1, self.embedding_dim)
            
        return torch.tensor(embeddings)
        
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
        
    def clear_cache(self):
        """清除缓存"""
        self.concept_cache.clear()
        self.embedding_cache.clear()
        
        if self.cache_dir:
            concept_cache_file = os.path.join(self.cache_dir, 'concept_cache.pt')
            embedding_cache_file = os.path.join(self.cache_dir, 'embedding_cache.pt')
            
            if os.path.exists(concept_cache_file):
                os.remove(concept_cache_file)
            if os.path.exists(embedding_cache_file):
                os.remove(embedding_cache_file) 