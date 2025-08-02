# -*- coding: utf-8 -*-
"""
Created on 2025-01-28

@author: zhoujiacheng
@file: concept_utils.py
@purpose: utilities for handling entity concepts from Neo4j
"""

from py2neo import Graph
from typing import Dict, Optional, List
import json

class ConceptRetriever:
    def __init__(self, 
                 neo4j_uri: str = 'neo4j://115.156.114.150:27687',
                 neo4j_user: str = 'neo4j',
                 neo4j_password: str = 'dasineo4j'):
        """初始化概念检索器"""
        self.graph = Graph(neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.concept_cache = {}  # 用于缓存已查询的概念

    def get_entity_concept(self, entity_str: str, topk: int = 1) -> str:
        """
        获取实体的概念
        
        Args:
            entity_str: 实体字符串
            topk: 返回概率最高的前k个概念
            
        Returns:
            概念字符串，如果没找到则返回空字符串
        """
        # 检查缓存
        cache_key = f"{entity_str}_{topk}"
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]

        try:
            # 构建查询
            query = (
                "MATCH (i:Instance {name:$name})-[r:IS_A]->(c:Concept) "
                "RETURN i.name AS Instance, tofloat(r.probability)/10000 AS `is_a`, "
                "c.name AS Concept ORDER BY `is_a` DESC LIMIT $topk"
            )
            
            # 执行查询
            res = self.graph.run(query, 
                               name=entity_str.lower(), 
                               topk=topk).data()
            
            # 获取结果
            concept = res[0]['Concept'] if res else ""
            
            # 更新缓存
            self.concept_cache[cache_key] = concept
            return concept
            
        except Exception as e:
            print(f"获取概念时出错: {str(e)}")
            return ""

    def get_instance_concepts(self, instance: Dict) -> Dict:
        """
        获取实例中头尾实体的概念
        
        Args:
            instance: 包含头尾实体的实例字典
            
        Returns:
            包含头尾实体概念的字典
        """
        # 获取头实体概念
        h_str = instance['h']['name'].split(' ')[-1].lower()
        h_concept = self.get_entity_concept(h_str)
        
        # 获取尾实体概念
        t_str = instance['t']['name'].split(' ')[-1].lower()
        t_concept = self.get_entity_concept(t_str)
        
        return {
            'head_concept': h_concept,
            'tail_concept': t_concept
        }

    def process_file_concepts(self, 
                            input_file: str, 
                            output_file: str, 
                            save_concepts: bool = True) -> Dict:
        """
        处理文件中所有实例的概念并保存
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            save_concepts: 是否将概念信息保存到输出文件
            
        Returns:
            概念统计字典
        """
        concept_stats = {}  # 用于统计概念频率
        
        with open(input_file, 'r', encoding='utf-8') as f:
            instances = [json.loads(line) for line in f]
        
        processed_instances = []
        for instance in instances:
            # 获取概念
            concepts = self.get_instance_concepts(instance)
            
            # 更新统计
            for concept in concepts.values():
                if concept:
                    concept_stats[concept] = concept_stats.get(concept, 0) + 1
            
            if save_concepts:
                # 将概念添加到实例中
                instance['h']['concept'] = concepts['head_concept']
                instance['t']['concept'] = concepts['tail_concept']
                processed_instances.append(instance)
        
        if save_concepts:
            # 保存处理后的实例
            with open(output_file, 'w', encoding='utf-8') as f:
                for instance in processed_instances:
                    f.write(json.dumps(instance) + '\n')
        
        return concept_stats

    def dump_concept_stats(self, 
                          concept_stats: Dict, 
                          output_file: str,
                          top_n: int = 600):
        """
        将概念统计信息保存到文件
        
        Args:
            concept_stats: 概念统计字典
            output_file: 输出文件路径
            top_n: 保存频率最高的前N个概念
        """
        # 按频率排序
        sorted_concepts = sorted(
            concept_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 创建concept到id的映射
        concept2id = {
            concept: idx 
            for idx, (concept, _) in enumerate(sorted_concepts[:top_n])
            if concept  # 排除空概念
        }
        
        # 保存映射
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concept2id, f, indent=2)

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='输入文件路径')
    parser.add_argument('--output_file', required=True, help='输出文件路径')
    parser.add_argument('--concept2id_file', required=True, help='concept2id输出文件路径')
    parser.add_argument('--top_n', type=int, default=600, help='保存频率最高的前N个概念')
    args = parser.parse_args()

    # 初始化概念检索器
    retriever = ConceptRetriever()
    
    # 处理文件
    concept_stats = retriever.process_file_concepts(
        args.input_file,
        args.output_file
    )
    
    # 保存概念统计
    retriever.dump_concept_stats(
        concept_stats,
        args.concept2id_file,
        args.top_n
    )

if __name__ == "__main__":
    main() 