# -*- coding: utf-8 -*-
"""
@purpose: generate negative samples using LLM with concept consistency
"""

import requests
import json
from typing import Dict, List, Optional
from concept_utils import ConceptRetriever

class NegativeSampleGenerator:
    def __init__(self, llm_config: dict):
        """
        初始化负样本生成器
        
        Args:
            llm_config: LLM配置，包含API密钥等信息
        """
        self.llm_config = llm_config
        self.concept_retriever = ConceptRetriever()

    def generate_prompt(self, instance: dict, h_concept: str, t_concept: str) -> str:
        """
        生成用于LLM的提示文本
        
        Args:
            instance: 原始实例
            h_concept: 头实体概念
            t_concept: 尾实体概念
            
        Returns:
            提示文本
        """
        context = " ".join(instance['token'])
        prompt = f"""作为一个关系生成助手，请帮我生成一个负样本。

                原始句子信息:
                - 上下文: {context}
                - 头实体: {instance['h']['name']} (概念: {h_concept})
                - 尾实体: {instance['t']['name']} (概念: {t_concept})
                - 当前关系: {instance['relation']}

                请生成一个新的句子，要求:
                1. 保持头尾实体的概念类型不变
                2. 确保新句子中的关系与原始关系不同
                3. 保持句子的自然性和语法正确性
                4. 生成的句子应该与原始上下文有相似的主题或领域
                5. 生成的句子应该包含隐含的关系信息

                请直接返回生成的句子，不需要任何解释。"""
        return prompt

    def generate_negative_samples(self, 
                                instance: dict, 
                                concepts: dict,
                                num_samples: int = 3) -> List[dict]:
        """
        生成负样本
        
        Args:
            instance: 原始实例
            concepts: 实体对应的概念
            num_samples: 生成的负样本数量
            
        Returns:
            负样本列表
        """
        h_concept = concepts[instance['h']['id']]
        t_concept = concepts[instance['t']['id']]
        
        prompt = self.generate_prompt(instance, h_concept, t_concept)
        responses = self.call_llm_api(prompt, num_samples)
        
        negative_samples = []
        for response in responses:
            if response:
                negative_sample = {
                    'token': response.split(),
                    'h': instance['h'],
                    't': instance['t'],
                    'relation': 'negative',  # 标记为负样本
                    'original_relation': instance['relation']  # 保存原始关系用于评估
                }
                negative_samples.append(negative_sample)
                
        return negative_samples

    def call_llm_api(self, prompt: str, num_samples: int) -> List[Optional[str]]:
        """
        调用LLM API
        
        Args:
            prompt: 提示文本
            num_samples: 生成样本数量
            
        Returns:
            生成的文本列表，失败则返回None
        """
        try:
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a relation generation assistant that creates negative samples while maintaining concept consistency."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "n": num_samples,
                "temperature": 0.7
            }
            
            # 这里需要根据实际使用的API进行调整
            response = requests.post(
                self.llm_config['api_endpoint'],
                headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
                json=payload
            )
            
            if response.status_code == 200:
                results = response.json()
                return [choice['message']['content'] for choice in results['choices']]
            else:
                print(f"API调用失败: {response.status_code}")
                return [None] * num_samples
                
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return [None] * num_samples

    def generate_batch(self, 
                      input_file: str, 
                      output_file: str,
                      samples_per_instance: int = 1):
        """
        批量生成负样本
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            samples_per_instance: 每个实例生成的负样本数量
        """
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            instances = [json.loads(line) for line in f]
        
        # 生成负样本
        negative_samples = []
        for idx, instance in enumerate(instances):
            print(f"处理实例 {idx+1}/{len(instances)}")
            for _ in range(samples_per_instance):
                neg_sample = self.generate_negative_samples(instance, self.concept_retriever.get_instance_concepts(instance))
                if neg_sample:
                    negative_samples.extend(neg_sample)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in negative_samples:
                f.write(json.dumps(sample) + '\n')

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='输入文件路径')
    parser.add_argument('--output_file', required=True, help='输出文件路径')
    parser.add_argument('--api_key', required=True, help='LLM API密钥')
    parser.add_argument('--samples_per_instance', type=int, default=1, 
                       help='每个实例生成的负样本数量')
    args = parser.parse_args()

    # 初始化生成器
    generator = NegativeSampleGenerator(args.api_key)
    
    # 生成负样本
    generator.generate_batch(
        args.input_file,
        args.output_file,
        args.samples_per_instance
    )

if __name__ == "__main__":
    main() 