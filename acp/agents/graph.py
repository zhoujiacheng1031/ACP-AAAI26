# -*- coding: utf-8 -*-
"""
LangGraph agent graph implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from langgraph import StateGraph, START, END
from langgraph.graph import CompiledGraph

from .state import AgentState, StateManager, StateValidator
from ..config.config import AgentConfig
from ..utils.llm_service import BaseLLMService, create_llm_service
from .concept_query import ConceptQueryNode, create_async_concept_query_node
from .mrda import MetaRelationDiscoveryAgent, MRDAFactory
from .rcaa import RelevantConceptAlignmentAgent, RCAAFactory


class AgentGraph:
    """LangGraph智能体图"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件实例
        self.llm_service: Optional[BaseLLMService] = None
        self.concept_query_node: Optional[ConceptQueryNode] = None
        self.mrda: Optional[MetaRelationDiscoveryAgent] = None
        self.rcaa: Optional[RelevantConceptAlignmentAgent] = None
        
        # LangGraph相关
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[CompiledGraph] = None
        
        # 统计信息
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'node_execution_times': {}
        }
    
    async def initialize(self):
        """异步初始化智能体图"""
        
        start_time = time.time()
        
        try:
            self.logger.info("开始初始化智能体图")
            
            # 1. 初始化LLM服务
            await self._initialize_llm_service()
            
            # 2. 初始化各个组件
            await self._initialize_components()
            
            # 3. 构建LangGraph
            self._build_graph()
            
            # 4. 编译图
            self._compile_graph()
            
            initialization_time = time.time() - start_time
            
            self.logger.info(f"智能体图初始化完成 - 耗时: {initialization_time:.2f}s")
            
        except Exception as e:
            error_msg = f"智能体图初始化失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    async def _initialize_llm_service(self):
        """初始化LLM服务"""
        
        try:
            self.llm_service = await create_llm_service(
                self.config.llm_service_config,
                service_type="siliconflow"
            )
            self.logger.debug("LLM服务初始化成功")
            
        except Exception as e:
            self.logger.error(f"LLM服务初始化失败: {e}")
            raise
    
    async def _initialize_components(self):
        """初始化各个组件"""
        
        try:
            # 初始化概念查询节点
            self.concept_query_node = await create_async_concept_query_node(
                self.config.concept_graph_config
            )
            
            # 初始化MRDA
            self.mrda = await MRDAFactory.create_mrda(
                self.llm_service,
                self.config.mrda_config
            )
            
            # 初始化RCAA
            self.rcaa = await RCAAFactory.create_rcaa(
                self.llm_service,
                self.config.rcaa_config
            )
            
            self.logger.debug("所有组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def _build_graph(self):
        """构建LangGraph"""
        
        try:
            # 创建状态图
            self.graph = StateGraph(AgentState)
            
            # 添加节点
            self.graph.add_node("concept_query", self._concept_query_wrapper)
            self.graph.add_node("mrda", self._mrda_wrapper)
            self.graph.add_node("rcaa", self._rcaa_wrapper)
            
            # 添加边
            self.graph.add_edge(START, "concept_query")
            self.graph.add_edge("concept_query", "mrda")
            self.graph.add_edge("mrda", "rcaa")
            self.graph.add_edge("rcaa", END)
            
            self.logger.debug("LangGraph构建成功")
            
        except Exception as e:
            self.logger.error(f"LangGraph构建失败: {e}")
            raise
    
    def _compile_graph(self):
        """编译图"""
        
        try:
            self.compiled_graph = self.graph.compile()
            self.logger.debug("LangGraph编译成功")
            
        except Exception as e:
            self.logger.error(f"LangGraph编译失败: {e}")
            raise
    
    async def _concept_query_wrapper(self, state: AgentState) -> AgentState:
        """概念查询节点包装器"""
        
        node_start_time = time.time()
        
        try:
            self.logger.debug("执行概念查询节点")
            
            result_state = await self.concept_query_node.process(state)
            
            # 记录执行时间
            execution_time = time.time() - node_start_time
            self._update_node_execution_time("concept_query", execution_time)
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"概念查询节点执行失败: {e}")
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'concept_query_node_error',
                    'message': str(e),
                    'node': 'concept_query'
                }
            })
    
    async def _mrda_wrapper(self, state: AgentState) -> AgentState:
        """MRDA节点包装器"""
        
        node_start_time = time.time()
        
        try:
            self.logger.debug("执行MRDA节点")
            
            result_state = await self.mrda.process(state)
            
            # 记录执行时间
            execution_time = time.time() - node_start_time
            self._update_node_execution_time("mrda", execution_time)
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"MRDA节点执行失败: {e}")
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'mrda_node_error',
                    'message': str(e),
                    'node': 'mrda'
                }
            })
    
    async def _rcaa_wrapper(self, state: AgentState) -> AgentState:
        """RCAA节点包装器"""
        
        node_start_time = time.time()
        
        try:
            self.logger.debug("执行RCAA节点")
            
            result_state = await self.rcaa.process(state)
            
            # 记录执行时间
            execution_time = time.time() - node_start_time
            self._update_node_execution_time("rcaa", execution_time)
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"RCAA节点执行失败: {e}")
            return StateManager.update_state(state, {
                'error_info': {
                    'type': 'rcaa_node_error',
                    'message': str(e),
                    'node': 'rcaa'
                }
            })
    
    def _update_node_execution_time(self, node_name: str, execution_time: float):
        """更新节点执行时间统计"""
        
        if node_name not in self.execution_stats['node_execution_times']:
            self.execution_stats['node_execution_times'][node_name] = {
                'total_time': 0.0,
                'execution_count': 0,
                'avg_time': 0.0
            }
        
        node_stats = self.execution_stats['node_execution_times'][node_name]
        node_stats['total_time'] += execution_time
        node_stats['execution_count'] += 1
        node_stats['avg_time'] = node_stats['total_time'] / node_stats['execution_count']
    
    async def execute(
        self,
        instances: List[Dict[str, Any]],
        n_way: int,
        k_shot: int
    ) -> AgentState:
        """执行智能体图"""
        
        if not self.compiled_graph:
            raise RuntimeError("智能体图未初始化，请先调用initialize()")
        
        execution_start_time = time.time()
        
        try:
            self.logger.info(f"开始执行智能体图 - 实例数: {len(instances)}")
            
            # 创建初始状态
            initial_state = StateManager.create_initial_state(instances, n_way, k_shot)
            
            # 验证初始状态
            if not StateValidator.validate_input_state(initial_state):
                raise ValueError("初始状态验证失败")
            
            # 执行图
            final_state = await self.compiled_graph.ainvoke(initial_state)
            
            execution_time = time.time() - execution_start_time
            
            # 更新统计信息
            self._update_execution_stats(execution_time, success=True)
            
            # 验证最终状态
            if StateValidator.validate_final_state(final_state):
                self.logger.info(f"智能体图执行成功 - 耗时: {execution_time:.2f}s")
            else:
                self.logger.warning("最终状态验证失败，但执行完成")
            
            return final_state
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            self._update_execution_stats(execution_time, success=False)
            
            error_msg = f"智能体图执行失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # 返回错误状态
            error_state = StateManager.create_initial_state(instances, n_way, k_shot)
            return StateManager.update_state(error_state, {
                'error_info': {
                    'type': 'graph_execution_error',
                    'message': error_msg,
                    'execution_time': execution_time
                }
            })
    
    def _update_execution_stats(self, execution_time: float, success: bool):
        """更新执行统计信息"""
        
        self.execution_stats['total_executions'] += 1
        
        if success:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
        
        # 更新平均执行时间
        total_executions = self.execution_stats['total_executions']
        current_avg = self.execution_stats['avg_execution_time']
        self.execution_stats['avg_execution_time'] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
    
    async def execute_batch(
        self,
        batch_instances: List[List[Dict[str, Any]]],
        n_way: int,
        k_shot: int,
        max_concurrent: int = None
    ) -> List[AgentState]:
        """批量执行智能体图"""
        
        if max_concurrent is None:
            max_concurrent = self.config.max_batch_size
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_batch(instances: List[Dict[str, Any]]) -> AgentState:
            async with semaphore:
                return await self.execute(instances, n_way, k_shot)
        
        # 创建任务
        tasks = [execute_single_batch(instances) for instances in batch_instances]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"批量执行第{i}项失败: {result}")
                # 创建错误状态
                error_state = StateManager.create_initial_state(
                    batch_instances[i] if i < len(batch_instances) else [],
                    n_way, k_shot
                )
                error_state = StateManager.update_state(error_state, {
                    'error_info': {
                        'type': 'batch_execution_error',
                        'message': str(result),
                        'batch_index': i
                    }
                })
                processed_results.append(error_state)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        
        stats = self.execution_stats.copy()
        
        # 计算成功率
        total = stats['total_executions']
        if total > 0:
            stats['success_rate'] = stats['successful_executions'] / total
            stats['failure_rate'] = stats['failed_executions'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """获取各组件的统计信息"""
        
        component_stats = {
            'graph_stats': self.get_execution_statistics(),
            'components_initialized': {
                'llm_service': self.llm_service is not None,
                'concept_query_node': self.concept_query_node is not None,
                'mrda': self.mrda is not None,
                'rcaa': self.rcaa is not None
            }
        }
        
        # 添加各组件的统计信息
        if self.mrda:
            component_stats['mrda_stats'] = self.mrda.get_processing_statistics()
        
        if self.rcaa:
            component_stats['rcaa_stats'] = self.rcaa.get_processing_statistics()
        
        return component_stats
    
    def reset_statistics(self):
        """重置统计信息"""
        
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'node_execution_times': {}
        }
        
        # 重置组件统计信息
        if self.mrda:
            self.mrda.reset_statistics()
        
        if self.rcaa:
            self.rcaa.reset_statistics()
    
    async def cleanup(self):
        """清理资源"""
        
        try:
            # 清理LLM服务
            if self.llm_service and hasattr(self.llm_service, '__aexit__'):
                await self.llm_service.__aexit__(None, None, None)
            
            self.logger.info("智能体图资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        # 注意：在析构函数中不能调用异步方法
        # 这里只是记录日志
        if hasattr(self, 'logger'):
            self.logger.debug("AgentGraph对象被销毁")


class AgentGraphFactory:
    """智能体图工厂"""
    
    @staticmethod
    async def create_agent_graph(config: AgentConfig) -> AgentGraph:
        """创建并初始化智能体图"""
        
        # 验证配置
        if not config.validate():
            raise ValueError("配置验证失败")
        
        # 创建智能体图
        agent_graph = AgentGraph(config)
        
        # 初始化
        await agent_graph.initialize()
        
        return agent_graph
    
    @staticmethod
    async def create_agent_graph_from_file(config_path: str) -> AgentGraph:
        """从配置文件创建智能体图"""
        
        from .config import ConfigManager
        
        # 加载配置
        config = ConfigManager.load_from_file(config_path)
        
        # 创建智能体图
        return await AgentGraphFactory.create_agent_graph(config)


# 便利函数
async def create_agent_graph(
    config: Optional[AgentConfig] = None,
    config_path: Optional[str] = None
) -> AgentGraph:
    """创建智能体图的便利函数"""
    
    if config_path:
        return await AgentGraphFactory.create_agent_graph_from_file(config_path)
    elif config:
        return await AgentGraphFactory.create_agent_graph(config)
    else:
        # 使用默认配置
        from .config import AgentConfig
        default_config = AgentConfig()
        return await AgentGraphFactory.create_agent_graph(default_config)


# 上下文管理器
class AgentGraphContext:
    """智能体图上下文管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_graph: Optional[AgentGraph] = None
    
    async def __aenter__(self) -> AgentGraph:
        """异步上下文管理器入口"""
        self.agent_graph = await AgentGraphFactory.create_agent_graph(self.config)
        return self.agent_graph
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.agent_graph:
            await self.agent_graph.cleanup()


# 使用示例函数
async def example_usage():
    """使用示例"""
    
    from .config import AgentConfig
    
    # 创建配置
    config = AgentConfig()
    
    # 使用上下文管理器
    async with AgentGraphContext(config) as agent_graph:
        # 准备测试数据
        test_instances = [
            {
                'h': {'name': 'Elon Musk', 'pos': [0, 2]},
                't': {'name': 'Tesla', 'pos': [5, 6]},
                'token': ['Elon', 'Musk', 'is', 'the', 'CEO', 'of', 'Tesla'],
                'relation': 'CEO_of'
            }
        ]
        
        # 执行智能体图
        result = await agent_graph.execute(test_instances, n_way=2, k_shot=1)
        
        # 检查结果
        if result.get('error_info'):
            print(f"执行失败: {result['error_info']}")
        else:
            print("执行成功")
            print(f"对齐概念数: {len(result.get('aligned_concepts', []))}")
        
        # 获取统计信息
        stats = agent_graph.get_execution_statistics()
        print(f"执行统计: {stats}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())