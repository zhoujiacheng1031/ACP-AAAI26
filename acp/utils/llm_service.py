# -*- coding: utf-8 -*-
"""
LLM service interface and implementations for agent system
"""

import asyncio
import aiohttp
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from .config import LLMServiceConfig
from .state import ProcessingResult


@dataclass
class LLMRequest:
    """LLM请求数据类"""
    prompt: str
    task_type: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    task_type: str
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class LLMServiceError(Exception):
    """LLM服务异常"""
    pass


class LLMRateLimitError(LLMServiceError):
    """LLM速率限制异常"""
    pass


class LLMTimeoutError(LLMServiceError):
    """LLM超时异常"""
    pass


class BaseLLMService(ABC):
    """LLM服务抽象基类"""
    
    def __init__(self, config: LLMServiceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._request_count = 0
        self._last_request_time = 0
        
    @abstractmethod
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """执行实际的LLM请求"""
        pass
    
    async def query(
        self, 
        prompt: str, 
        task_type: str = "general",
        **kwargs
    ) -> LLMResponse:
        """查询LLM"""
        request = LLMRequest(
            prompt=prompt,
            task_type=task_type,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            metadata=kwargs.get('metadata')
        )
        
        # 速率限制检查
        await self._check_rate_limit()
        
        # 重试机制
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = await self._make_request(request)
                response.processing_time = time.time() - start_time
                
                self.logger.debug(
                    f"LLM请求成功 - 任务类型: {task_type}, "
                    f"处理时间: {response.processing_time:.2f}s"
                )
                
                return response
                
            except LLMTimeoutError as e:
                last_error = e
                self.logger.warning(f"LLM请求超时 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                
            except LLMRateLimitError as e:
                last_error = e
                self.logger.warning(f"LLM速率限制 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                
            except Exception as e:
                last_error = e
                self.logger.error(f"LLM请求失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        # 所有重试都失败
        raise LLMServiceError(f"LLM请求失败，已重试{self.config.max_retries}次: {last_error}")
    
    async def batch_query(
        self, 
        requests: List[LLMRequest]
    ) -> List[LLMResponse]:
        """批量查询LLM"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_single_request(request: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.query(
                    request.prompt, 
                    request.task_type,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    metadata=request.metadata
                )
        
        tasks = [process_single_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_rate_limit(self):
        """检查速率限制"""
        current_time = time.time()
        
        # 重置计数器（每分钟）
        if current_time - self._last_request_time > 60:
            self._request_count = 0
            self._last_request_time = current_time
        
        # 检查是否超过速率限制
        if self._request_count >= self.config.rate_limit_per_minute:
            wait_time = 60 - (current_time - self._last_request_time)
            if wait_time > 0:
                self.logger.warning(f"达到速率限制，等待 {wait_time:.1f} 秒")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = time.time()
        
        self._request_count += 1


class SiliconFlowLLMService(BaseLLMService):
    """SiliconFlow LLM服务实现"""
    
    def __init__(self, config: LLMServiceConfig):
        super().__init__(config)
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.get_api_key()}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """执行SiliconFlow API请求"""
        if not self.session:
            raise LLMServiceError("LLM服务未初始化，请使用async with语句")
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._get_system_prompt(request.task_type)
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "stream": False,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "frequency_penalty": self.config.frequency_penalty,
            "n": 1,
            "response_format": {"type": "text"}
        }
        
        try:
            async with self.session.post(self.config.api_url, json=payload) as response:
                if response.status == 429:
                    raise LLMRateLimitError("API速率限制")
                elif response.status == 408:
                    raise LLMTimeoutError("API请求超时")
                elif response.status != 200:
                    error_text = await response.text()
                    raise LLMServiceError(f"API请求失败 (状态码: {response.status}): {error_text}")
                
                result = await response.json()
                content = result['choices'][0]['message']['content']
                
                return LLMResponse(
                    content=content,
                    task_type=request.task_type,
                    confidence=self._estimate_confidence(content),
                    metadata={
                        'model': self.config.model_name,
                        'usage': result.get('usage', {}),
                        'response_id': result.get('id')
                    }
                )
                
        except asyncio.TimeoutError:
            raise LLMTimeoutError("请求超时")
        except aiohttp.ClientError as e:
            raise LLMServiceError(f"网络请求失败: {e}")
    
    def _get_system_prompt(self, task_type: str) -> str:
        """根据任务类型获取系统提示词"""
        prompts = {
            "meta_relation_mining": """你是一个专门发现概念间元关系的助手。
给定两组概念，你需要：
1. 分析概念间可能存在的语义关联
2. 发现潜在的关系模式
3. 输出发现的元关系，格式：关系类型|描述|置信度(0-1)|证据
4. 按关联强度排序输出""",
            
            "semantic_verification": """你是一个语义一致性验证助手。
给定一个关系和相关实体，你需要：
1. 评估关系与实体的语义一致性
2. 计算置信度分数
3. 提供验证理由
4. 输出格式：验证结果|置信度(0-1)|理由""",
            
            "negative_generation": """你是一个负样本生成助手。
给定一个正样本关系，你需要：
1. 生成语义相似但不同的负样本关系
2. 控制相似度在指定范围内
3. 输出格式：负样本关系|相似度(0-1)|生成理由""",
            
            "concept_reasoning": """你是一个概念推理助手。
给定一个概念和上下文，你需要：
1. 进行递归推理，发现相关概念
2. 构建推理链
3. 评估每个概念的重要性
4. 输出格式：推理步骤|相关概念|重要性分数(0-1)|推理依据""",
            
            "concept_selection": """你是一个概念选择助手。
给定一组候选概念和上下文，你需要：
1. 评估概念与上下文的相关性
2. 考虑概念的独特性和重要性
3. 选择最相关的概念
4. 输出格式：概念序号|相关度(0-1)|选择理由""",
            
            "general": """你是一个智能助手，请根据用户的要求提供准确、有用的回答。"""
        }
        
        return prompts.get(task_type, prompts["general"])
    
    def _estimate_confidence(self, content: str) -> float:
        """估算响应置信度"""
        # 简单的置信度估算逻辑
        if not content or len(content.strip()) < 10:
            return 0.1
        
        # 基于内容长度和结构的启发式估算
        lines = content.strip().split('\n')
        structured_lines = sum(1 for line in lines if '|' in line)
        
        if structured_lines > 0:
            structure_score = min(structured_lines / len(lines), 1.0)
        else:
            structure_score = 0.5
        
        length_score = min(len(content) / 500, 1.0)
        
        return (structure_score * 0.6 + length_score * 0.4)


class MockLLMService(BaseLLMService):
    """模拟LLM服务，用于测试"""
    
    def __init__(self, config: LLMServiceConfig):
        super().__init__(config)
        self.responses = self._init_mock_responses()
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """模拟LLM请求"""
        # 模拟处理时间
        await asyncio.sleep(0.1)
        
        response_generator = self.responses.get(
            request.task_type, 
            self.responses["general"]
        )
        
        content = response_generator(request.prompt)
        
        return LLMResponse(
            content=content,
            task_type=request.task_type,
            confidence=0.8,
            metadata={'mock': True}
        )
    
    def _init_mock_responses(self) -> Dict[str, callable]:
        """初始化模拟响应"""
        return {
            "meta_relation_mining": lambda prompt: 
                "相关性|概念间存在语义相关性|0.8|基于概念定义的相似性\n"
                "层次性|概念间存在上下位关系|0.7|基于概念的抽象层次\n"
                "功能性|概念间存在功能关联|0.6|基于概念的功能属性",
            
            "semantic_verification": lambda prompt:
                "通过|0.75|关系与实体语义一致，符合常识",
            
            "negative_generation": lambda prompt:
                "反向关系|0.6|将原关系方向反转\n"
                "相似关系|0.7|选择语义相近但不同的关系",
            
            "concept_reasoning": lambda prompt:
                "步骤1|相关概念A|0.8|基于语义相似性\n"
                "步骤2|相关概念B|0.6|基于功能关联性\n"
                "步骤3|相关概念C|0.4|基于上下文关联",
            
            "concept_selection": lambda prompt:
                "1|0.9|与上下文高度相关\n"
                "3|0.7|具有重要的语义信息\n"
                "5|0.5|提供补充信息",
            
            "general": lambda prompt: f"这是对提示词的模拟响应: {prompt[:50]}..."
        }


class LLMServiceFactory:
    """LLM服务工厂"""
    
    @staticmethod
    def create_service(
        service_type: str, 
        config: LLMServiceConfig
    ) -> BaseLLMService:
        """创建LLM服务实例"""
        if service_type == "siliconflow":
            return SiliconFlowLLMService(config)
        elif service_type == "mock":
            return MockLLMService(config)
        else:
            raise ValueError(f"不支持的LLM服务类型: {service_type}")
    
    @staticmethod
    async def create_async_service(
        service_type: str, 
        config: LLMServiceConfig
    ) -> BaseLLMService:
        """创建异步LLM服务实例"""
        service = LLMServiceFactory.create_service(service_type, config)
        
        if hasattr(service, '__aenter__'):
            await service.__aenter__()
        
        return service


# 便利函数
async def create_llm_service(config: LLMServiceConfig, service_type: str = "siliconflow") -> BaseLLMService:
    """创建LLM服务的便利函数"""
    return await LLMServiceFactory.create_async_service(service_type, config)