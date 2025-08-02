# -*- coding: utf-8 -*-
"""
Training script for LangGraph-enhanced few-shot relation classification
"""

import os
import asyncio
import logging
import argparse
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
import json

from ..config.config import AgentConfig, ConfigManager
from ..agents.graph import create_agent_graph
from ..data.data_loader import create_data_loader
from ..models.cap import create_cap_model
from ..agents.state import StateManager


class FewShotTrainer:
    """小样本关系分类训练器"""
    
    def __init__(
        self,
        config: AgentConfig,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        self.config = config
        self.model_config = model_config
        self.training_config = training_config
        self.logger = logging.getLogger(__name__)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.tokenizer = None
        self.agent_graph = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练统计
        self.training_stats = {
            'epoch': 0,
            'global_step': 0,
            'best_accuracy': 0.0,
            'best_f1': 0.0,
            'training_time': 0.0
        }
    
    async def initialize(self):
        """异步初始化"""
        
        self.logger.info("初始化训练器...")
        
        # 1. 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['pretrained_model']
        )
        self.logger.info(f"分词器初始化完成: {self.model_config['pretrained_model']}")
        
        # 2. 初始化智能体图（如果启用）
        if self.training_config.get('use_agent_enhancement', True):
            try:
                self.agent_graph = await create_agent_graph(self.config)
                self.logger.info("智能体图初始化完成")
            except Exception as e:
                self.logger.warning(f"智能体图初始化失败: {e}")
                self.agent_graph = None
        
        # 3. 初始化模型
        self.model = create_cap_model(
            pretrained_model=self.model_config['pretrained_model'],
            n_way=self.training_config['n_way'],
            hidden_size=self.model_config['hidden_size'],
            enable_agent_enhancement=self.agent_graph is not None,
            **self.model_config
        )
        self.model.to(self.device)
        self.logger.info("模型初始化完成")
        
        # 4. 初始化优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # 5. 初始化学习率调度器
        if self.training_config['scheduler_type'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['num_epochs']
            )
        elif self.training_config['scheduler_type'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        
        self.logger.info("训练器初始化完成")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        
        self.logger.info("创建数据加载器...")
        
        # 训练数据加载器
        train_loader = create_data_loader(
            data_path=self.training_config['train_data_path'],
            tokenizer=self.tokenizer,
            agent_graph=self.agent_graph,
            n_way=self.training_config['n_way'],
            k_shot=self.training_config['k_shot'],
            q_query=self.training_config['q_query'],
            batch_size=self.training_config['batch_size'],
            max_length=self.training_config['max_length'],
            nota_ratio=self.training_config['nota_ratio'],
            mode='train',
            use_agent_enhancement=self.training_config.get('use_agent_enhancement', True),
            shuffle=True
        )
        
        # 验证数据加载器
        val_loader = create_data_loader(
            data_path=self.training_config['val_data_path'],
            tokenizer=self.tokenizer,
            agent_graph=self.agent_graph,
            n_way=self.training_config['n_way'],
            k_shot=self.training_config['k_shot'],
            q_query=self.training_config['q_query'],
            batch_size=self.training_config['batch_size'],
            max_length=self.training_config['max_length'],
            nota_ratio=self.training_config['nota_ratio'],
            mode='val',
            use_agent_enhancement=self.training_config.get('use_agent_enhancement', True),
            shuffle=False
        )
        
        self.logger.info(
            f"数据加载器创建完成 - 训练: {len(train_loader)}, 验证: {len(val_loader)}"
        )
        
        return train_loader, val_loader
    
    async def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        epoch_stats = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'separation_loss': 0.0,
            'nota_loss': 0.0,
            'pddm_loss': 0.0,
            'accuracy': 0.0,
            'nota_accuracy': 0.0,
            'num_batches': 0
        }
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.training_config['num_epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 移动数据到设备
                batch = self._move_batch_to_device(batch)
                
                # 前向传播
                outputs = self.model(
                    support_input_ids=batch['support']['input_ids'],
                    support_attention_mask=batch['support']['attention_mask'],
                    support_entity_positions=batch['support']['entity_positions'],
                    support_labels=batch['support']['labels'],
                    query_input_ids=batch['query']['input_ids'],
                    query_attention_mask=batch['query']['attention_mask'],
                    query_entity_positions=batch['query']['entity_positions'],
                    support_agent_results=batch.get('support_agent_results'),
                    query_agent_results=batch.get('query_agent_results')
                )
                
                # 计算损失
                losses = self.model.compute_loss(
                    outputs,
                    batch['query']['labels'],
                    batch['query'].get('nota_labels'),
                    self.training_config.get('loss_weights')
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                
                # 梯度裁剪
                if self.training_config.get('max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['max_grad_norm']
                    )
                
                self.optimizer.step()
                
                # 更新统计信息
                self._update_epoch_stats(epoch_stats, losses, outputs, batch)
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'acc': f"{epoch_stats['accuracy']:.3f}"
                })
                
                # 记录到wandb
                if self.training_config.get('use_wandb') and batch_idx % 10 == 0:
                    self._log_training_step(losses, outputs, batch_idx, epoch)
                
                self.training_stats['global_step'] += 1
                
            except Exception as e:
                self.logger.error(f"训练批次{batch_idx}失败: {e}")
                continue
        
        # 计算平均统计
        for key in epoch_stats:
            if key != 'num_batches':
                epoch_stats[key] /= max(epoch_stats['num_batches'], 1)
        
        return epoch_stats
    
    async def validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        
        self.model.eval()
        val_stats = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'separation_loss': 0.0,
            'nota_loss': 0.0,
            'pddm_loss': 0.0,
            'accuracy': 0.0,
            'nota_accuracy': 0.0,
            'f1_score': 0.0,
            'num_batches': 0
        }
        
        all_predictions = []
        all_labels = []
        all_nota_predictions = []
        all_nota_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                try:
                    # 移动数据到设备
                    batch = self._move_batch_to_device(batch)
                    
                    # 前向传播
                    outputs = self.model(
                        support_input_ids=batch['support']['input_ids'],
                        support_attention_mask=batch['support']['attention_mask'],
                        support_entity_positions=batch['support']['entity_positions'],
                        support_labels=batch['support']['labels'],
                        query_input_ids=batch['query']['input_ids'],
                        query_attention_mask=batch['query']['attention_mask'],
                        query_entity_positions=batch['query']['entity_positions'],
                        support_agent_results=batch.get('support_agent_results'),
                        query_agent_results=batch.get('query_agent_results')
                    )
                    
                    # 计算损失
                    losses = self.model.compute_loss(
                        outputs,
                        batch['query']['labels'],
                        batch['query'].get('nota_labels'),
                        self.training_config.get('loss_weights')
                    )
                    
                    # 更新统计信息
                    self._update_epoch_stats(val_stats, losses, outputs, batch)
                    
                    # 收集预测结果
                    predictions = torch.argmax(outputs['logits'], dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['query']['labels'].cpu().numpy())
                    
                    if 'nota_predictions' in outputs:
                        all_nota_predictions.extend(outputs['nota_predictions'].cpu().numpy())
                        if 'nota_labels' in batch['query']:
                            all_nota_labels.extend(batch['query']['nota_labels'].cpu().numpy())
                    
                except Exception as e:
                    self.logger.error(f"验证批次{batch_idx}失败: {e}")
                    continue
        
        # 计算平均统计
        for key in val_stats:
            if key != 'num_batches':
                val_stats[key] /= max(val_stats['num_batches'], 1)
        
        # 计算F1分数
        if all_predictions and all_labels:
            from sklearn.metrics import f1_score, classification_report
            val_stats['f1_score'] = f1_score(all_labels, all_predictions, average='macro')
            
            # 详细报告（调试模式）
            if self.logger.isEnabledFor(logging.DEBUG):
                report = classification_report(all_labels, all_predictions)
                self.logger.debug(f"分类报告:\n{report}")
        
        return val_stats
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """移动批次数据到设备"""
        
        def move_tensor_dict(tensor_dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in tensor_dict.items()
            }
        
        return {
            'support': move_tensor_dict(batch['support']),
            'query': move_tensor_dict(batch['query']),
            'relations': batch['relations'],
            'support_agent_results': batch.get('support_agent_results'),
            'query_agent_results': batch.get('query_agent_results')
        }
    
    def _update_epoch_stats(
        self,
        stats: Dict[str, float],
        losses: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any]
    ):
        """更新epoch统计信息"""
        
        stats['num_batches'] += 1
        
        # 损失统计
        for loss_name, loss_value in losses.items():
            if loss_name in stats:
                stats[loss_name] += loss_value.item()
        
        # 准确率统计
        predictions = torch.argmax(outputs['logits'], dim=1)
        accuracy = (predictions == batch['query']['labels']).float().mean().item()
        stats['accuracy'] += accuracy
        
        # NOTA准确率统计
        if 'nota_predictions' in outputs and 'nota_labels' in batch['query']:
            nota_accuracy = (
                outputs['nota_predictions'] == batch['query']['nota_labels']
            ).float().mean().item()
            stats['nota_accuracy'] += nota_accuracy
    
    def _log_training_step(
        self,
        losses: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int
    ):
        """记录训练步骤到wandb"""
        
        log_dict = {
            'epoch': epoch,
            'batch': batch_idx,
            'global_step': self.training_stats['global_step']
        }
        
        # 添加损失
        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.item()
        
        # 添加学习率
        log_dict['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        wandb.log(log_dict)
    
    def save_checkpoint(
        self,
        epoch: int,
        val_stats: Dict[str, float],
        is_best: bool = False
    ):
        """保存检查点"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_stats': self.training_stats,
            'val_stats': val_stats,
            'config': {
                'agent_config': self.config.to_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config
            }
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.training_config['output_dir'],
            'checkpoint_latest.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_checkpoint_path = os.path.join(
                self.training_config['output_dir'],
                'checkpoint_best.pt'
            )
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"保存最佳检查点: {best_checkpoint_path}")
    
    async def train(self):
        """主训练循环"""
        
        self.logger.info("开始训练...")
        start_time = time.time()
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        
        # 创建输出目录
        os.makedirs(self.training_config['output_dir'], exist_ok=True)
        
        # 初始化wandb
        if self.training_config.get('use_wandb'):
            wandb.init(
                project=self.training_config.get('wandb_project', 'langgraph-fsrc'),
                config={
                    **self.config.to_dict(),
                    **self.model_config,
                    **self.training_config
                }
            )
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                self.training_stats['epoch'] = epoch
                
                # 训练
                train_stats = await self.train_epoch(train_loader, epoch)
                
                # 验证
                val_stats = await self.validate_epoch(val_loader, epoch)
                
                # 更新学习率
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_stats['accuracy'])
                    else:
                        self.scheduler.step()
                
                # 记录epoch结果
                self.logger.info(
                    f"Epoch {epoch+1}/{self.training_config['num_epochs']} - "
                    f"Train Loss: {train_stats['loss']:.4f}, "
                    f"Train Acc: {train_stats['accuracy']:.4f}, "
                    f"Val Loss: {val_stats['loss']:.4f}, "
                    f"Val Acc: {val_stats['accuracy']:.4f}, "
                    f"Val F1: {val_stats['f1_score']:.4f}"
                )
                
                # 记录到wandb
                if self.training_config.get('use_wandb'):
                    log_dict = {}
                    for key, value in train_stats.items():
                        if key != 'num_batches':
                            log_dict[f'train_epoch/{key}'] = value
                    for key, value in val_stats.items():
                        if key != 'num_batches':
                            log_dict[f'val_epoch/{key}'] = value
                    wandb.log(log_dict)
                
                # 检查是否为最佳模型
                is_best = val_stats['f1_score'] > self.training_stats['best_f1']
                if is_best:
                    self.training_stats['best_accuracy'] = val_stats['accuracy']
                    self.training_stats['best_f1'] = val_stats['f1_score']
                
                # 保存检查点
                self.save_checkpoint(epoch, val_stats, is_best)
                
                # 早停检查
                if self._should_early_stop(val_stats, epoch):
                    self.logger.info("触发早停")
                    break
            
            self.training_stats['training_time'] = time.time() - start_time
            
            self.logger.info(
                f"训练完成 - 最佳准确率: {self.training_stats['best_accuracy']:.4f}, "
                f"最佳F1: {self.training_stats['best_f1']:.4f}, "
                f"训练时间: {self.training_stats['training_time']:.2f}s"
            )
            
        finally:
            # 清理资源
            if self.agent_graph:
                await self.agent_graph.cleanup()
            
            if self.training_config.get('use_wandb'):
                wandb.finish()
    
    def _should_early_stop(self, val_stats: Dict[str, float], epoch: int) -> bool:
        """检查是否应该早停"""
        
        early_stop_patience = self.training_config.get('early_stop_patience')
        if not early_stop_patience:
            return False
        
        # 简单的早停逻辑（可以改进）
        if not hasattr(self, '_best_val_loss'):
            self._best_val_loss = float('inf')
            self._patience_counter = 0
        
        if val_stats['loss'] < self._best_val_loss:
            self._best_val_loss = val_stats['loss']
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        
        return self._patience_counter >= early_stop_patience


async def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='LangGraph增强的小样本关系分类训练')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model_config', type=str, help='模型配置文件路径')
    parser.add_argument('--training_config', type=str, help='训练配置文件路径')
    
    # 覆盖参数
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--train_data', type=str, help='训练数据路径')
    parser.add_argument('--val_data', type=str, help='验证数据路径')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--use_wandb', action='store_true', help='使用wandb记录')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        agent_config = ConfigManager.load_from_file(args.config)
        
        # 加载模型配置
        if args.model_config:
            with open(args.model_config, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
        else:
            model_config = {
                'pretrained_model': 'bert-base-uncased',
                'hidden_size': 768,
                'dropout': 0.1,
                'use_pddm': True,
                'use_nota_detection': True,
                'nota_detection_method': 'nddm',
                'nota_threshold': 0.5,
                'margin': 1.0
            }
        
        # 加载训练配置
        if args.training_config:
            with open(args.training_config, 'r', encoding='utf-8') as f:
                training_config = json.load(f)
        else:
            training_config = {
                'train_data_path': 'data/fewrel/train.json',
                'val_data_path': 'data/fewrel/dev.json',
                'output_dir': 'outputs',
                'n_way': 5,
                'k_shot': 1,
                'q_query': 5,
                'batch_size': 4,
                'max_length': 128,
                'nota_ratio': 0.3,
                'num_epochs': 10,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'scheduler_type': 'cosine',
                'max_grad_norm': 1.0,
                'early_stop_patience': 5,
                'use_agent_enhancement': True,
                'use_wandb': False,
                'wandb_project': 'langgraph-fsrc',
                'loss_weights': {
                    'classification': 1.0,
                    'separation': 0.5,
                    'nota': 0.5,
                    'pddm': 0.3
                }
            }
        
        # 应用命令行覆盖
        if args.output_dir:
            training_config['output_dir'] = args.output_dir
        if args.train_data:
            training_config['train_data_path'] = args.train_data
        if args.val_data:
            training_config['val_data_path'] = args.val_data
        if args.batch_size:
            training_config['batch_size'] = args.batch_size
        if args.learning_rate:
            training_config['learning_rate'] = args.learning_rate
        if args.num_epochs:
            training_config['num_epochs'] = args.num_epochs
        if args.use_wandb:
            training_config['use_wandb'] = True
        
        # 创建训练器
        trainer = FewShotTrainer(agent_config, model_config, training_config)
        
        # 初始化
        await trainer.initialize()
        
        # 开始训练
        await trainer.train()
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())