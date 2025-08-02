# -*- coding: utf-8 -*-
"""
MACAP: Multi-Agent Concept-Aware Prototypical Network
基于多智能体的概念感知原型网络
"""

__version__ = "1.0.0"
__author__ = "MACAP Team"
__description__ = "Multi-Agent Concept-Aware Prototypical Network for Few-Shot Relation Classification"

# Core components
from .agents import (
    AgentGraph, MetaRelationDiscoveryAgent, RelevantConceptAlignmentAgent,
    AgentState, ConceptInfo, MetaRelation, ReasoningPath
)

from .config import (
    AgentConfig, MRDAConfig, RCAAConfig, ConfigManager
)

from .models import (
    ConceptAwarePrototypicalNetwork, AgentEnhancedLAF
)

from .data import (
    FewShotRelationDataset, create_data_loader
)

from .utils import (
    FewShotTrainer, BaseLLMService
)

__all__ = [
    # Agents
    'AgentGraph',
    'MetaRelationDiscoveryAgent',
    'RelevantConceptAlignmentAgent',
    'AgentState',
    'ConceptInfo',
    'MetaRelation',
    'ReasoningPath',
    
    # Configuration
    'AgentConfig',
    'MRDAConfig',
    'RCAAConfig',
    'ConfigManager',
    
    # Models
    'ConceptAwarePrototypicalNetwork',
    'AgentEnhancedLAF',
    
    # Data
    'FewShotRelationDataset',
    'create_data_loader',
    
    # Utils
    'FewShotTrainer',
    'BaseLLMService'
]