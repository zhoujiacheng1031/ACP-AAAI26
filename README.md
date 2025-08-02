# MACAP: Multi-Agent Concept-Aware Prototypical Network

基于多智能体的概念感知原型网络，用于小样本关系分类任务。

## 🏗️ 项目结构

```
MACAP/
├── macap/                   # 核心代码包
│   ├── agents/             # 多智能体系统
│   │   ├── graph.py        # LangGraph智能体图
│   │   ├── mrda.py         # 元关系发现智能体
│   │   ├── rcaa.py         # 相关概念对齐智能体
│   │   ├── state.py        # 状态管理
│   │   └── ...
│   ├── config/             # 配置管理
│   │   ├── config.py       # 配置类定义
│   │   ├── agent_config.json # 智能体配置
│   │   └── .env.example    # 环境变量模板
│   ├── data/               # 数据处理
│   │   ├── data_loader.py  # 数据加载器
│   │   ├── few_shot_dataset.py # 小样本数据集
│   │   └── ...
│   ├── models/             # 深度学习模型
│   │   ├── cap.py          # 概念感知原型网络
│   │   ├── laf.py          # 语言感知特征模块
│   │   └── ...
│   └── utils/              # 工具函数
│       ├── llm_service.py  # LLM服务接口
│       ├── train.py        # 训练工具
│       └── ...
├── data/                   # 数据集
│   ├── fewrel/            # FewRel数据集
│   └── tacred/            # TACRED数据集
├── results/                # 实验结果
└── scripts/                # 执行脚本
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp macap/config/.env.example .env
# 编辑 .env 文件，填入必要的API密钥
```

### 2. 数据准备

确保数据集位于正确位置：
- FewRel: `data/fewrel/`
- TACRED: `data/tacred/`

### 3. 训练模型

```bash
# 使用Python模块运行
python -m macap.utils.train --config macap/config/agent_config.json

# 或使用脚本
bash scripts/train.sh
```

### 4. 评估模型

```bash
python -m macap.utils.evaluate --checkpoint results/best_model.pt
```

## 🧠 核心组件

### 多智能体系统 (Agents)

- **MRDA (Meta-Relation Discovery Agent)**: 发现和验证概念间的元关系
- **RCAA (Relevant Concept Alignment Agent)**: 进行递归推理和概念对齐
- **AgentGraph**: 基于LangGraph的智能体协作图

### 深度学习模型 (Models)

- **CAP (Concept-Aware Prototypical Network)**: 概念感知原型网络
- **LAF (Language-Aware Feature)**: 语言感知特征模块
- **NOTA Detection**: 开放世界关系分类

### 数据处理 (Data)

- **FewShotRelationDataset**: 小样本关系分类数据集
- **AgentEnhancedDataLoader**: 智能体增强数据加载器

## 📊 实验结果

训练结果保存在 `results/` 目录中，包括：
- 模型检查点
- 训练日志
- 评估指标
- 配置文件备份

## 🔧 配置说明

### 智能体配置
- MRDA和RCAA参数设置
- LLM服务配置
- 概念图配置

### 模型配置
- 预训练模型设置
- 网络架构参数
- NOTA检测配置

## 🎯 主要特性

- **多智能体协作**: 基于LangGraph的智能体系统
- **概念感知学习**: 利用概念图增强关系分类
- **NOTA检测**: 支持开放世界关系分类
- **异步处理**: 高效的并行处理能力
- **模块化设计**: 易于扩展和定制

## 📖 使用示例

```python
import asyncio
from macap import (
    AgentConfig, create_agent_graph,
    create_cap_model, FewShotTrainer
)

async def main():
    # 创建配置
    config = AgentConfig()
    
    # 创建智能体图
    agent_graph = await create_agent_graph(config)
    
    # 创建模型
    model = create_cap_model(n_way=5, enable_agent_enhancement=True)
    
    # 训练
    trainer = FewShotTrainer(config, model_config, training_config)
    await trainer.initialize()
    await trainer.train()
    
    # 清理
    await agent_graph.cleanup()

asyncio.run(main())
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- LangGraph 框架
- Transformers 库
- PyTorch 生态系统