# -*- coding: utf-8 -*-
"""
Created on 2025-01-23

@author: zhoujiacheng
@file: data_process.py
@purpose: process data for few-shot learning format
"""

import json
import argparse
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from neo4j import GraphDatabase
from transformers import AutoTokenizer
import random

def load_tacred_data(input_file):
    """Load TACRED data from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def convert_tacred_format(tacred_data):
    """Convert TACRED data to the desired format."""
    converted_data = []
    
    for item in tacred_data:
        token = item['token']
        head_start, head_end = item['subj_start'], item['subj_end']
        tail_start, tail_end = item['obj_start'], item['obj_end']
        
        head_entity = {
            "name": item['subj_type'],
            "pos": [head_start, head_end]
        }
        
        tail_entity = {
            "name": item['obj_type'],
            "pos": [tail_start, tail_end]
        }
        
        new_item = {
            "token": token,
            "h": head_entity,
            "t": tail_entity,
            "relation": item['relation']
        }
        
        converted_data.append(new_item)
    
    return converted_data

def save_converted_data(converted_data, output_file):
    """Save converted data to a JSON file with one object per line."""
    with open(output_file, 'w') as f:
        for item in converted_data:
            json.dump(item, f)
            f.write('\n')

def generate_rel2id(converted_data_list):
    """Generate relation to id mapping from all processed data."""
    # Collect all unique relations
    relations = set()
    for data in converted_data_list:
        for item in data:
            relations.add(item['relation'])
    
    # Sort relations alphabetically for consistent ordering
    relations = sorted(list(relations))
    
    # Create relation to id mapping
    rel2id = {rel: idx for idx, rel in enumerate(relations)}
    
    # Save to file
    with open('tacred/rel2id.json', 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, indent=2)

def process_files(input_files, output_files):
    """Process multiple input files and save to corresponding output files."""
    if len(input_files) != len(output_files):
        raise ValueError("Number of input files must match number of output files")
    
    all_converted_data = []
    for input_file, output_file in zip(input_files, output_files):
        print(f"Processing {input_file} -> {output_file}")
        tacred_data = load_tacred_data(input_file)
        converted_data = convert_tacred_format(tacred_data)
        save_converted_data(converted_data, output_file)
        all_converted_data.append(converted_data)
        print(f"Completed processing {input_file}")
    
    # Generate rel2id.json after processing all files
    generate_rel2id(all_converted_data)

def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(description='Process TACRED data files')
    parser.add_argument('--input_file', nargs='+', required=True,
                      help='Input file paths (space-separated)')
    parser.add_argument('--output_file', nargs='+', required=True,
                      help='Output file paths (space-separated)')
    
    args = parser.parse_args()
    process_files(args.input_file, args.output_file)

class DataProcessor:
    """数据预处理器"""
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.neo4j_client = GraphDatabase.driver(
            config['neo4j_uri'],
            auth=(config['neo4j_user'], config['neo4j_password'])
        )
        
    def process_tacred(self, input_file: str, output_file: str):
        """处理TACRED数据集"""
        logging.info(f"Processing TACRED file: {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        processed_data = []
        for item in tqdm(data):
            processed_item = self._process_tacred_item(item)
            if processed_item:
                processed_data.append(processed_item)
                
        # 保存处理后的数据
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        logging.info(f"Processed {len(processed_data)} instances")
        
    def _process_tacred_item(self, item: Dict) -> Dict:
        """处理单个TACRED实例"""
        try:
            # 提取头尾实体
            head_span = (item['h']['pos'][0], item['h']['pos'][1])
            tail_span = (item['t']['pos'][0], item['t']['pos'][1])
            
            # 获取概念
            head_concepts = self._query_concepts(item['h']['id'])
            tail_concepts = self._query_concepts(item['t']['id'])
            
            # 构建处理后的实例
            processed = {
                'tokens': item['token'],
                'head': {
                    'text': ' '.join(item['token'][head_span[0]:head_span[1]]),
                    'type': item['h']['type'],
                    'span': head_span,
                    'id': item['h']['id'],
                    'concepts': head_concepts
                },
                'tail': {
                    'text': ' '.join(item['token'][tail_span[0]:tail_span[1]]),
                    'type': item['t']['type'],
                    'span': tail_span,
                    'id': item['t']['id'],
                    'concepts': tail_concepts
                },
                'relation': item['relation']
            }
            
            return processed
            
        except Exception as e:
            logging.warning(f"Error processing item: {str(e)}")
            return None
            
    def _query_concepts(self, entity_id: str) -> List[str]:
        """查询实体相关概念"""
        with self.neo4j_client.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $id})-[r:HAS_CONCEPT*1..3]->(c:Concept)
                RETURN c.name AS concept, length(r) AS distance
                ORDER BY distance
                LIMIT 10
                """,
                id=entity_id
            )
            return [record['concept'] for record in result]
            
    def merge_data(self, 
                  train_file: str,
                  dev_file: str,
                  test_file: str,
                  output_file: str):
        """合并数据集"""
        merged_data = []
        
        # 读取并合并数据
        for file_path in [train_file, dev_file, test_file]:
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_data.extend(data)
                
        # 保存合并后的数据
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
            
        logging.info(f"Merged {len(merged_data)} instances to {output_file}")
        
    def generate_few_shot(self,
                         input_file: str,
                         output_dir: str,
                         n_way: int = 5,
                         k_shot: int = 5,
                         n_episodes: int = 1000):
        """生成Few-shot数据集"""
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # 按关系分组
        relation_data = {}
        for item in data:
            rel = item['relation']
            if rel not in relation_data:
                relation_data[rel] = []
            relation_data[rel].append(item)
            
        # 生成episodes
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(n_episodes)):
            episode = self._generate_episode(
                relation_data, n_way, k_shot
            )
            
            output_file = os.path.join(output_dir, f"episode_{i}.json")
            with open(output_file, 'w') as f:
                json.dump(episode, f, indent=2)
                
    def _generate_episode(self,
                         relation_data: Dict[str, List],
                         n_way: int,
                         k_shot: int) -> Dict:
        """生成单个Few-shot episode"""
        # 随机选择关系
        selected_relations = random.sample(list(relation_data.keys()), n_way)
        
        # 构建支持集和查询集
        support_set = []
        query_set = []
        
        for rel in selected_relations:
            # 随机选择样本
            samples = random.sample(relation_data[rel], k_shot * 2)
            
            # 分配到支持集和查询集
            support_set.extend(samples[:k_shot])
            query_set.extend(samples[k_shot:])
            
        return {
            'support_set': support_set,
            'query_set': query_set,
            'relations': selected_relations
        }

if __name__ == "__main__":
    main()
