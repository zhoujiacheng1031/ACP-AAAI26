# -*- coding: utf-8 -*-
"""
Configuration classes for LangGraph agent system
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


@dataclass
class MRDAConfig:
    """Meta-Relation Discovery Agent配置"""
    
    # 元关系挖掘配置
    num_mining_stages: int = 3
    max_relations_per_stage: int = 10
    mining_temperature: float = 0.1
    
    # 语义一致性验证配置
    verification_threshold: float = 0.5
    attention_weight: float = 0.6
    confidence_weight: float = 0.4
    
    # 负样本生成配置
    negative_difficulty: float = 0.5
    negative_ratio: float = 0.3
    similarity_range: tuple = (0.3, 0.8)
    
    # LLM调用配置
    max_retries: int = 3
    timeout: float = 30.0
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not (1 <= self.num_mining_stages <= 10):
            return False
        if not (0.0 <= self.verification_threshold <= 1.0):
            return False
        if not (0.0 <= self.negative_difficulty <= 1.0):
            return False
        return True


@dataclass
class RCAAConfig:
    """Relevant Concept Alignment Agent配置"""
    
    # 递归推理配置
    max_reasoning_depth: int = 3
    max_concepts_per_path: int = 5
    reasoning_temperature: float = 0.1
    
    # 双重注意力机制配置
    attention_alpha: float = 0.5
    hidden_size: int = 768
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # 隐式信息性建模配置
    informativeness_lambda: float = 0.1
    classification_weight: float = 1.0
    similarity_weight: float = 0.5
    
    # 概念选择配置
    concept_selection_threshold: float = 0.3
    max_selected_concepts: int = 10
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not (1 <= self.max_reasoning_depth <= 5):
            return False
        if not (0.0 <= self.attention_alpha <= 1.0):
            return False
        if not (0.0 <= self.concept_selection_threshold <= 1.0):
            return False
        return True


@dataclass
class LLMServiceConfig:
    """LLM服务配置"""
    
    # API配置
    api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key_env: str = "SILICONFLOW_API_KEY"
    
    # 请求配置
    max_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.7
    top_k: int = 50
    frequency_penalty: float = 0.5
    
    # 重试和超时配置
    max_retries: int = 3
    timeout: float = 30.0
    retry_delay: float = 1.0
    
    # 并发配置
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 60
    
    def get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        return os.getenv(self.api_key_env)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self.api_url or not self.model_name:
            return False
        if not self.get_api_key():
            return False
        if not (0.0 <= self.temperature <= 2.0):
            return False
        return True


@dataclass
class ConceptGraphConfig:
    """概念图配置"""
    
    # 概念查询配置
    max_hop_distance: int = 2
    max_concepts_per_entity: int = 10
    concept_score_threshold: float = 0.1
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600  # 秒
    
    # 概念图路径
    concept_graph_path: Optional[str] = None
    concept_embeddings_path: Optional[str] = None
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not (1 <= self.max_hop_distance <= 5):
            return False
        if not (0.0 <= self.concept_score_threshold <= 1.0):
            return False
        return True


@dataclass
class CacheConfig:
    """缓存配置"""
    
    # 缓存类型
    cache_type: str = "memory"  # memory, redis, file
    
    # 内存缓存配置
    max_memory_size: int = 1000  # MB
    max_items: int = 10000
    
    # 文件缓存配置
    cache_dir: str = "cache"
    max_file_size: int = 100  # MB
    
    # Redis缓存配置（如果使用）
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # TTL配置
    default_ttl: int = 3600  # 秒
    concept_ttl: int = 7200
    llm_response_ttl: int = 1800
    
    def validate(self) -> bool:
        """验证配置有效性"""
        valid_types = ["memory", "redis", "file"]
        if self.cache_type not in valid_types:
            return False
        return True


@dataclass
class AgentConfig:
    """智能体系统总配置"""
    
    # 子配置
    mrda_config: MRDAConfig = field(default_factory=MRDAConfig)
    rcaa_config: RCAAConfig = field(default_factory=RCAAConfig)
    llm_service_config: LLMServiceConfig = field(default_factory=LLMServiceConfig)
    concept_graph_config: ConceptGraphConfig = field(default_factory=ConceptGraphConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    # 全局配置
    enable_parallel_processing: bool = True
    max_batch_size: int = 32
    processing_timeout: float = 300.0  # 秒
    
    # 日志配置
    log_level: str = "INFO"
    enable_detailed_logging: bool = False
    log_file: Optional[str] = None
    
    # 调试配置
    debug_mode: bool = False
    save_intermediate_results: bool = False
    intermediate_results_dir: str = "debug_output"
    
    def validate(self) -> bool:
        """验证所有配置的有效性"""
        configs = [
            self.mrda_config,
            self.rcaa_config,
            self.llm_service_config,
            self.concept_graph_config,
            self.cache_config
        ]
        
        return all(config.validate() for config in configs)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mrda_config': self.mrda_config.__dict__,
            'rcaa_config': self.rcaa_config.__dict__,
            'llm_service_config': self.llm_service_config.__dict__,
            'concept_graph_config': self.concept_graph_config.__dict__,
            'cache_config': self.cache_config.__dict__,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_batch_size': self.max_batch_size,
            'processing_timeout': self.processing_timeout,
            'log_level': self.log_level,
            'enable_detailed_logging': self.enable_detailed_logging,
            'log_file': self.log_file,
            'debug_mode': self.debug_mode,
            'save_intermediate_results': self.save_intermediate_results,
            'intermediate_results_dir': self.intermediate_results_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建配置"""
        mrda_config = MRDAConfig(**config_dict.get('mrda_config', {}))
        rcaa_config = RCAAConfig(**config_dict.get('rcaa_config', {}))
        llm_service_config = LLMServiceConfig(**config_dict.get('llm_service_config', {}))
        concept_graph_config = ConceptGraphConfig(**config_dict.get('concept_graph_config', {}))
        cache_config = CacheConfig(**config_dict.get('cache_config', {}))
        
        return cls(
            mrda_config=mrda_config,
            rcaa_config=rcaa_config,
            llm_service_config=llm_service_config,
            concept_graph_config=concept_graph_config,
            cache_config=cache_config,
            enable_parallel_processing=config_dict.get('enable_parallel_processing', True),
            max_batch_size=config_dict.get('max_batch_size', 32),
            processing_timeout=config_dict.get('processing_timeout', 300.0),
            log_level=config_dict.get('log_level', 'INFO'),
            enable_detailed_logging=config_dict.get('enable_detailed_logging', False),
            log_file=config_dict.get('log_file'),
            debug_mode=config_dict.get('debug_mode', False),
            save_intermediate_results=config_dict.get('save_intermediate_results', False),
            intermediate_results_dir=config_dict.get('intermediate_results_dir', 'debug_output')
        )


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_from_file(config_path: str) -> AgentConfig:
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        return AgentConfig.from_dict(config_dict)
    
    @staticmethod
    def save_to_file(config: AgentConfig, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix == '.json':
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    @staticmethod
    def create_default_config() -> AgentConfig:
        """创建默认配置"""
        return AgentConfig()
    
    @staticmethod
    def merge_configs(base_config: AgentConfig, override_config: Dict[str, Any]) -> AgentConfig:
        """合并配置"""
        base_dict = base_config.to_dict()
        
        # 递归合并字典
        def merge_dict(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = merge_dict(base_dict, override_config)
        return AgentConfig.from_dict(merged_dict)
    
    @staticmethod
    def validate_config(config: AgentConfig) -> List[str]:
        """验证配置并返回错误信息"""
        errors = []
        
        if not config.validate():
            errors.append("配置验证失败")
        
        # 检查API密钥
        if not config.llm_service_config.get_api_key():
            errors.append(f"未找到API密钥环境变量: {config.llm_service_config.api_key_env}")
        
        # 检查缓存目录
        if config.cache_config.cache_type == "file":
            cache_dir = Path(config.cache_config.cache_dir)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"无法创建缓存目录: {e}")
        
        return errors