import json
import re
from typing import Optional, List, Dict
import os

def get_rel2data(in_file, rel2idx):
    rel2rules = []
    max_id = -1
    for rel in rel2idx:
        if rel2idx[rel] > max_id:
            max_id = rel2idx[rel]
    for i in range(max_id+1):
        rel2rules.append([])
    with open(in_file, 'rb', encoding='utf-8') as inf:
        for line in inf:
            if(len(line) != 0):
                obj = json.loads(line)
                rel_str = obj["relation"]
                idx = rel2idx[rel_str]
                rel2rules[idx].append(obj)

    return rel2rules

def get_id2rel(rel2id):
    id2rel = {}
    for rel in rel2id:
        id2rel[rel2id[rel]] = rel
    return id2rel

def get_rel2id(rel2id_file):
    with open (rel2id_file, 'r', encoding='utf-8') as relf:
        rel2id = json.load(relf)

    return rel2id

def simple_obj(obj):
    if 'mask_0_emb' in obj:  del (obj['mask_0_emb'])
    if 'mask_1_emb' in obj:  del (obj['mask_1_emb'])
    if 'mask_2_emb' in obj:  del (obj['mask_2_emb'])
    # if 'mask_0' in obj: del(obj['mask_0'])
    # if 'mask_1' in obj: del (obj['mask_1'])
    # if 'mask_2' in obj: del (obj['mask_2'])
    if 'is_act' in obj: obj['is_act'] = 1
    return obj

def simple_rule_obj(obj):
    if 'mask_0_emb' in obj:  del (obj['mask_0_emb'])
    if 'mask_1_emb' in obj:  del (obj['mask_1_emb'])
    if 'mask_2_emb' in obj:  del (obj['mask_2_emb'])
    return obj

def get_data_list(data_file):
    with open(data_file, 'r', encoding='utf-8') as inf:
        data_list = []
        for line in inf:
            line = line.strip()
            if len(line) == 0:
                continue
            obj = json.loads(line)
            data_list.append(obj)

    return data_list

def isSameObj(obj1, obj2):
    sent1 = " ".join(obj1['token'])
    sent2 = " ".join(obj2['token'])
    h1 = obj1['h']['name'].lower()
    h2 = obj2['h']['name'].lower()
    t1 = obj1['t']['name'].lower()
    t2 = obj2['t']['name'].lower()
    if sent1.lower() == sent2.lower() and h1 == h2 and t1 == t2:
        return True
    else:
        return False

def clean_text(text: str) -> str:
    """清理文本
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
        
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 标准化标点符号
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", '"', text)
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    return text

def normalize_entity(text: str) -> str:
    """标准化实体文本
    
    Args:
        text: 实体文本
        
    Returns:
        标准化后的实体文本
    """
    if not text:
        return ""
        
    # 转换为小写
    text = text.lower()
    
    # 移除括号内容
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 移除特殊字符
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 移除多余空白
    text = ' '.join(text.split())
    
    return text

def extract_spans(text: str, entity: str) -> List[List[int]]:
    """提取实体在文本中的位置
    
    Args:
        text: 原文本
        entity: 实体文本
        
    Returns:
        实体位置列表 [[start, end], ...]
    """
    spans = []
    start = 0
    while True:
        start = text.lower().find(entity.lower(), start)
        if start == -1:
            break
        spans.append([start, start + len(entity)])
        start += 1
    return spans

def get_context_window(text: str, 
                      span: List[int], 
                      window_size: int = 50) -> str:
    """获取实体周围的上下文窗口
    
    Args:
        text: 原文本
        span: 实体位置 [start, end]
        window_size: 窗口大小
        
    Returns:
        上下文文本
    """
    start, end = span
    context_start = max(0, start - window_size)
    context_end = min(len(text), end + window_size)
    return text[context_start:context_end]

def check_entity_overlap(span1: List[int], 
                        span2: List[int]) -> bool:
    """检查两个实体是否重叠
    
    Args:
        span1: 第一个实体的位置 [start, end]
        span2: 第二个实体的位置 [start, end]
        
    Returns:
        是否重叠
    """
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])

def load_dataset(data_dir: str, split: str) -> List[Dict]:
    """加载处理后的数据集
    
    Args:
        data_dir: 数据集目录路径
        split: 数据集划分(train/dev/test)
        
    Returns:
        数据集实例列表
    """
    file_path = os.path.join(data_dir, f'{split}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_rel2id(data_dir: str) -> Dict[str, int]:
    """加载关系ID映射
    
    Args:
        data_dir: 数据集目录路径
        
    Returns:
        关系到ID的映射字典
    """
    file_path = os.path.join(data_dir, 'rel2id.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_entity_span(tokens: List[str], entity: Dict) -> str:
    """获取实体文本
    
    Args:
        tokens: 句子分词列表
        entity: 实体信息字典
        
    Returns:
        实体文本
    """
    start, end = entity['pos']
    return ' '.join(tokens[start:end+1])

def get_context_window(
    text: List[str],
    span: List[int],
    window_size: int = 5
) -> List[str]:
    """获取实体周围的上下文窗口
    
    Args:
        text: 句子分词列表
        span: 实体位置 [start, end]
        window_size: 窗口大小
        
    Returns:
        上下文分词列表
    """
    start, end = span
    context_start = max(0, start - window_size)
    context_end = min(len(text), end + window_size)
    return text[context_start:context_end]