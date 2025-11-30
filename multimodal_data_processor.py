"""
多模态数据处理模块
支持时间序列、文本、图像等多种数据模态的融合处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，多模态功能将受限")


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, max_length: int = 512, use_bert: bool = False):
        """
        初始化文本处理器
        
        参数:
            max_length: 最大文本长度
            use_bert: 是否使用BERT（需要安装transformers库）
        """
        self.max_length = max_length
        self.use_bert = use_bert
        
        # 尝试加载BERT（如果可用）
        self.bert_model = None
        self.bert_tokenizer = None
        if use_bert:
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = 'bert-base-chinese'  # 中文BERT
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_model = AutoModel.from_pretrained(model_name)
                print(f"✅ BERT模型加载成功: {model_name}")
            except ImportError:
                print("⚠️  transformers库未安装，将使用简单的文本处理")
                self.use_bert = False
            except Exception as e:
                print(f"⚠️  BERT加载失败: {e}")
                self.use_bert = False
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        提取文本特征
        
        参数:
            text: 文本字符串
        
        返回:
            特征向量
        """
        if self.use_bert and self.bert_model and self.bert_tokenizer:
            # 使用BERT提取特征
            inputs = self.bert_tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS]标记的嵌入
                features = outputs.last_hidden_state[:, 0, :].numpy()
            
            return features[0]
        else:
            # 简单的文本特征提取（词频、长度等）
            features = np.array([
                len(text),  # 文本长度
                text.count(' '),  # 空格数
                text.count('。'),  # 句号数
                text.count('，'),  # 逗号数
                text.count('！'),  # 感叹号数
                text.count('？'),  # 问号数
            ])
            # 归一化
            features = features / (np.max(features) + 1e-8)
            return features
    
    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        情感分析（简化版）
        
        参数:
            text: 文本字符串
        
        返回:
            情感得分字典
        """
        # 简单的情感词词典
        positive_words = ['涨', '升', '好', '优', '强', '利好', '增长', '上升', '盈利', '收益']
        negative_words = ['跌', '降', '坏', '差', '弱', '利空', '下降', '亏损', '损失', '风险']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.5  # 中性
        else:
            sentiment_score = positive_count / total
        
        return {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'is_positive': sentiment_score > 0.6,
            'is_negative': sentiment_score < 0.4
        }


class TimeSeriesFeatureExtractor:
    """时间序列特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        pass
    
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取统计特征
        
        参数:
            data: 时间序列数据
        
        返回:
            特征向量
        """
        features = np.array([
            np.mean(data),  # 均值
            np.std(data),   # 标准差
            np.min(data),   # 最小值
            np.max(data),   # 最大值
            np.median(data),  # 中位数
            np.percentile(data, 25),  # 第一四分位数
            np.percentile(data, 75),  # 第三四分位数
            (np.max(data) - np.min(data)) / (np.mean(data) + 1e-8),  # 波动率
        ])
        
        # 归一化
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return features
    
    def extract_trend_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取趋势特征
        
        参数:
            data: 时间序列数据
        
        返回:
            特征向量
        """
        if len(data) < 2:
            return np.zeros(3)
        
        # 计算趋势
        linear_trend = np.polyfit(range(len(data)), data, 1)[0]  # 线性趋势斜率
        
        # 计算变化率
        change_rate = (data[-1] - data[0]) / (data[0] + 1e-8)
        
        # 计算波动性
        volatility = np.std(np.diff(data)) / (np.mean(np.abs(data)) + 1e-8)
        
        features = np.array([linear_trend, change_rate, volatility])
        return features
    
    def extract_all_features(self, data: np.ndarray) -> np.ndarray:
        """提取所有特征"""
        stat_features = self.extract_statistical_features(data)
        trend_features = self.extract_trend_features(data)
        return np.concatenate([stat_features, trend_features])


class MultimodalFusion:
    """多模态融合模块"""
    
    def __init__(self,
                 time_series_dim: int = 11,
                 text_dim: int = 768,
                 fusion_method: str = 'attention'):
        """
        初始化多模态融合模块
        
        参数:
            time_series_dim: 时间序列特征维度
            text_dim: 文本特征维度
            fusion_method: 融合方法 ('attention', 'concatenate', 'weighted')
        """
        self.time_series_dim = time_series_dim
        self.text_dim = text_dim
        self.fusion_method = fusion_method
        
        if TORCH_AVAILABLE and fusion_method == 'attention':
            # 注意力融合层
            self.attention_weights = nn.Sequential(
                nn.Linear(time_series_dim + text_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # 两个模态的注意力权重
                nn.Softmax(dim=-1)
            )
        else:
            self.attention_weights = None
    
    def fuse(self, time_series_features: np.ndarray, 
             text_features: np.ndarray) -> np.ndarray:
        """
        融合多模态特征
        
        参数:
            time_series_features: 时间序列特征
            text_features: 文本特征
        
        返回:
            融合后的特征
        """
        if self.fusion_method == 'concatenate':
            # 简单拼接
            return np.concatenate([time_series_features, text_features])
        
        elif self.fusion_method == 'weighted':
            # 加权融合（固定权重）
            alpha = 0.7  # 时间序列权重
            beta = 0.3   # 文本权重
            
            # 归一化（分别归一化）
            ts_norm = time_series_features / (np.linalg.norm(time_series_features) + 1e-8)
            text_norm = text_features / (np.linalg.norm(text_features) + 1e-8)
            
            # 始终使用拼接，因为形状通常不匹配
            return np.concatenate([alpha * ts_norm, beta * text_norm])
        
        elif self.fusion_method == 'attention' and self.attention_weights:
            # 注意力融合
            try:
                combined = np.concatenate([time_series_features, text_features])
                combined_tensor = torch.FloatTensor(combined).unsqueeze(0)
                
                with torch.no_grad():
                    weights = self.attention_weights(combined_tensor)[0].numpy()
                
                ts_weight = weights[0]
                text_weight = weights[1]
                
                # 如果形状不匹配，使用加权拼接而不是加权相加
                if time_series_features.shape == text_features.shape:
                    return ts_weight * time_series_features + text_weight * text_features
                else:
                    # 形状不匹配时，使用加权拼接
                    return np.concatenate([ts_weight * time_series_features, text_weight * text_features])
            except Exception:
                # 如果注意力融合失败，使用简单拼接
                return np.concatenate([time_series_features, text_features])
        
        else:
            # 默认：拼接
            return np.concatenate([time_series_features, text_features])


class MultimodalDataProcessor:
    """多模态数据处理器（主类）"""
    
    def __init__(self,
                 text_max_length: int = 512,
                 use_bert: bool = False,
                 fusion_method: str = 'attention'):
        """
        初始化多模态数据处理器
        
        参数:
            text_max_length: 文本最大长度
            use_bert: 是否使用BERT
            fusion_method: 融合方法
        """
        self.text_processor = TextProcessor(
            max_length=text_max_length,
            use_bert=use_bert
        )
        
        self.ts_extractor = TimeSeriesFeatureExtractor()
        
        # 获取特征维度
        ts_dim = len(self.ts_extractor.extract_all_features(np.array([1, 2, 3, 4, 5])))
        text_dim = 768 if use_bert else 6  # BERT: 768, 简单方法: 6
        
        self.fusion = MultimodalFusion(
            time_series_dim=ts_dim,
            text_dim=text_dim,
            fusion_method=fusion_method
        )
    
    def process(self,
                time_series_data: np.ndarray,
                text_data: Optional[str] = None,
                additional_features: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        处理多模态数据
        
        参数:
            time_series_data: 时间序列数据
            text_data: 文本数据（可选）
            additional_features: 额外特征（可选）
        
        返回:
            处理后的特征字典
        """
        # 提取时间序列特征
        ts_features = self.ts_extractor.extract_all_features(time_series_data)
        ts_features = np.asarray(ts_features, dtype=np.float32)
        
        # 提取文本特征
        if text_data:
            text_features = self.text_processor.extract_features(text_data)
            sentiment = self.text_processor.sentiment_analysis(text_data)
        else:
            text_features = np.zeros(self.fusion.text_dim, dtype=np.float32)
            sentiment = {'sentiment_score': 0.5}
        
        # 确保文本特征是numpy数组
        text_features = np.asarray(text_features, dtype=np.float32)
        
        # 处理标量或0维数组
        if text_features.ndim == 0:
            text_features = np.array([text_features])
        elif text_features.ndim > 1:
            text_features = text_features.flatten()
        
        # 确保特征维度匹配
        if text_features.shape[0] != self.fusion.text_dim:
            # 如果文本特征维度不匹配，进行调整
            if text_features.shape[0] > self.fusion.text_dim:
                text_features = text_features[:self.fusion.text_dim]
            else:
                # 用零填充
                padding = np.zeros(self.fusion.text_dim - text_features.shape[0], dtype=np.float32)
                text_features = np.concatenate([text_features, padding])
        
        # 融合特征
        try:
            fused_features = self.fusion.fuse(ts_features, text_features)
            # 确保返回的是numpy数组
            if not isinstance(fused_features, np.ndarray):
                fused_features = np.array(fused_features, dtype=np.float32)
        except Exception as e:
            # 如果融合失败，使用简单拼接（静默失败，不频繁打印）
            fused_features = np.concatenate([ts_features, text_features])
        
        return {
            'time_series_features': ts_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'sentiment': sentiment
        }
    
    def batch_process(self,
                     time_series_batch: List[np.ndarray],
                     text_batch: Optional[List[str]] = None) -> List[Dict[str, np.ndarray]]:
        """
        批量处理
        
        参数:
            time_series_batch: 时间序列数据列表
            text_batch: 文本数据列表（可选）
        
        返回:
            处理结果列表
        """
        results = []
        text_batch = text_batch or [None] * len(time_series_batch)
        
        for ts_data, text_data in zip(time_series_batch, text_batch):
            result = self.process(ts_data, text_data)
            results.append(result)
        
        return results

