# -*- coding: utf-8 -*-
"""
@purpose: Evaluation metrics for few-shot relation classification
"""

import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

logger = logging.getLogger(__name__)

def compute_metrics(
    preds: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    nota_preds: Union[List, np.ndarray] = None,
    nota_labels: Union[List, np.ndarray] = None
) -> Dict[str, float]:
    """计算评估指标
    
    Args:
        preds: 关系分类预测
        labels: 真实标签
        nota_preds: NOTA预测 (可选)
        nota_labels: NOTA标签 (可选)
        
    Returns:
        包含各项指标的字典
    """
    metrics = {}
    
    try:
        # 转换为numpy数组
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
            
        # 计算关系分类指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='macro'  # 使用宏平均
        )
        
        accuracy = accuracy_score(labels, preds)
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        # 计算NOTA检测指标(如果提供)
        if nota_preds is not None and nota_labels is not None:
            if not isinstance(nota_preds, np.ndarray):
                nota_preds = np.array(nota_preds)
            if not isinstance(nota_labels, np.ndarray):
                nota_labels = np.array(nota_labels)
                
            nota_precision, nota_recall, nota_f1, _ = precision_recall_fscore_support(
                nota_labels,
                nota_preds,
                average='binary'  # 二分类
            )
            
            nota_accuracy = accuracy_score(nota_labels, nota_preds)
            
            metrics.update({
                'nota_precision': nota_precision,
                'nota_recall': nota_recall,
                'nota_f1': nota_f1,
                'nota_accuracy': nota_accuracy
            })
            
        return metrics
        
    except Exception as e:
        logger.error(f"计算指标时出错: {str(e)}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }

def compute_confusion_matrix(
    preds: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    num_classes: int
) -> np.ndarray:
    """计算混淆矩阵
    
    Args:
        preds: 预测
        labels: 标签
        num_classes: 类别数
        
    Returns:
        混淆矩阵
    """
    try:
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
            
        confusion_matrix = np.zeros((num_classes, num_classes))
        for i in range(len(preds)):
            confusion_matrix[labels[i]][preds[i]] += 1
            
        return confusion_matrix
        
    except Exception as e:
        logger.error(f"计算混淆矩阵时出错: {str(e)}")
        return np.zeros((num_classes, num_classes))

def compute_per_class_metrics(
    confusion_matrix: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """计算每个类别的指标
    
    Args:
        confusion_matrix: 混淆矩阵
        
    Returns:
        每个类别的指标字典
    """
    try:
        num_classes = confusion_matrix.shape[0]
        per_class_metrics = {}
        
        for i in range(num_classes):
            tp = confusion_matrix[i][i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        return per_class_metrics
        
    except Exception as e:
        logger.error(f"计算每类指标时出错: {str(e)}")
        return {} 