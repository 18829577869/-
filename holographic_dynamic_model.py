"""
全息动态模型模块
集成多种数据源、模型和策略的综合动态系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import json
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HolographicMemory:
    """全息记忆模块（存储和检索历史信息）"""
    
    def __init__(self, memory_size: int = 1000, decay_rate: float = 0.95):
        """
        初始化全息记忆
        
        参数:
            memory_size: 记忆容量
            decay_rate: 衰减率（越旧的信息权重越小）
        """
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.memories = []
        self.weights = []
    
    def store(self, data: Dict, timestamp: Optional[datetime] = None):
        """
        存储记忆
        
        参数:
            data: 数据字典
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.memories.append({
            'data': data,
            'timestamp': timestamp
        })
        
        # 计算权重（越新权重越大）
        age = (datetime.now() - timestamp).total_seconds() / 3600  # 小时
        weight = np.exp(-age * (1 - self.decay_rate))
        self.weights.append(weight)
        
        # 保持记忆容量
        if len(self.memories) > self.memory_size:
            self.memories.pop(0)
            self.weights.pop(0)
    
    def retrieve(self, 
                 query: Optional[Dict] = None,
                 top_k: int = 10,
                 use_weights: bool = True) -> List[Dict]:
        """
        检索记忆
        
        参数:
            query: 查询条件（可选）
            top_k: 返回前K个相关记忆
            use_weights: 是否使用权重
        
        返回:
            相关记忆列表
        """
        if len(self.memories) == 0:
            return []
        
        # 简化版：返回最近的记忆
        if query is None:
            if use_weights:
                # 按权重排序
                indices = np.argsort(self.weights)[::-1]
            else:
                # 按时间排序（最新的在前）
                indices = list(range(len(self.memories)))[::-1]
            
            top_k = min(top_k, len(indices))
            return [self.memories[i] for i in indices[:top_k]]
        
        # 如果有查询，可以进行更复杂的检索（这里简化处理）
        return self.retrieve(query=None, top_k=top_k, use_weights=use_weights)
    
    def get_statistics(self) -> Dict:
        """获取记忆统计"""
        if len(self.memories) == 0:
            return {}
        
        return {
            'total_memories': len(self.memories),
            'oldest_memory': min(m['timestamp'] for m in self.memories),
            'newest_memory': max(m['timestamp'] for m in self.memories),
            'average_weight': np.mean(self.weights) if self.weights else 0
        }


class TextAnalyzer:
    """文本分析器（增强版）"""
    
    def __init__(self):
        """初始化文本分析器"""
        # 关键词词典
        self.positive_keywords = [
            '涨', '升', '好', '优', '强', '利好', '增长', '上升', '盈利', '收益',
            '突破', '反弹', '上涨', '利好', '推荐', '买入', '看涨'
        ]
        
        self.negative_keywords = [
            '跌', '降', '坏', '差', '弱', '利空', '下降', '亏损', '损失', '风险',
            '下跌', '破位', '看跌', '卖出', '谨慎', '危险', '警告'
        ]
        
        self.neutral_keywords = [
            '维持', '稳定', '持平', '观望', '中性', '不变'
        ]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        情感分析
        
        参数:
            text: 文本
        
        返回:
            情感分析结果
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_keywords if word in text_lower)
        
        total = positive_count + negative_count + neutral_count
        
        if total == 0:
            sentiment_score = 0.5  # 中性
            confidence = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total
            sentiment_score = (sentiment_score + 1) / 2  # 归一化到[0,1]
            confidence = total / (total + 5)  # 置信度
        
        # 判断情感倾向
        if sentiment_score > 0.6:
            sentiment_label = 'positive'
        elif sentiment_score < 0.4:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': float(sentiment_score),
            'sentiment_label': sentiment_label,
            'confidence': float(confidence),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        提取关键词（简化版）
        
        参数:
            text: 文本
            top_n: 返回前N个关键词
        
        返回:
            关键词列表
        """
        # 简单实现：提取所有关键词
        all_keywords = self.positive_keywords + self.negative_keywords + self.neutral_keywords
        found_keywords = [kw for kw in all_keywords if kw in text.lower()]
        return found_keywords[:top_n]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取实体（简化版）
        
        参数:
            text: 文本
        
        返回:
            实体字典
        """
        # 简单实现：识别数字、百分比等
        import re
        
        entities = {
            'numbers': re.findall(r'\d+\.?\d*', text),
            'percentages': re.findall(r'\d+\.?\d*%', text),
            'prices': re.findall(r'[\d,]+\.?\d*元', text),
        }
        
        return entities
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        批量分析
        
        参数:
            texts: 文本列表
        
        返回:
            分析结果列表
        """
        return [self.analyze_sentiment(text) for text in texts]


class HolographicDynamicModel:
    """全息动态模型（主类）"""
    
    def __init__(self,
                 memory_size: int = 1000,
                 enable_text_analysis: bool = True,
                 enable_memory: bool = True):
        """
        初始化全息动态模型
        
        参数:
            memory_size: 记忆容量
            enable_text_analysis: 是否启用文本分析
            enable_memory: 是否启用记忆功能
        """
        self.enable_text_analysis = enable_text_analysis
        self.enable_memory = enable_memory
        
        # 初始化组件
        if enable_memory:
            self.memory = HolographicMemory(memory_size=memory_size)
        else:
            self.memory = None
        
        if enable_text_analysis:
            self.text_analyzer = TextAnalyzer()
        else:
            self.text_analyzer = None
        
        # 状态记录
        self.state_history = []
        self.prediction_history = []
    
    def process(self,
                time_series_data: np.ndarray,
                text_data: Optional[str] = None,
                technical_indicators: Optional[Dict] = None,
                market_intelligence: Optional[Dict] = None) -> Dict[str, Any]:
        """
        处理综合数据
        
        参数:
            time_series_data: 时间序列数据
            text_data: 文本数据
            technical_indicators: 技术指标
            market_intelligence: 市场情报
        
        返回:
            处理结果
        """
        result = {
            'timestamp': datetime.now(),
            'time_series_stats': self._calculate_time_series_stats(time_series_data),
            'text_analysis': None,
            'technical_indicators': technical_indicators,
            'market_intelligence': market_intelligence,
            'memory_influence': None,
            'comprehensive_signal': None
        }
        
        # 文本分析
        if text_data and self.text_analyzer:
            text_analysis = self.text_analyzer.analyze_sentiment(text_data)
            result['text_analysis'] = text_analysis
            
            # 提取关键词和实体
            keywords = self.text_analyzer.extract_keywords(text_data)
            entities = self.text_analyzer.extract_entities(text_data)
            result['text_analysis']['keywords'] = keywords
            result['text_analysis']['entities'] = entities
        
        # 记忆检索
        if self.memory:
            # 检索相关记忆
            relevant_memories = self.memory.retrieve(top_k=5)
            result['memory_influence'] = self._analyze_memory_influence(relevant_memories)
            
            # 存储当前状态到记忆
            memory_data = {
                'time_series_stats': result['time_series_stats'],
                'text_analysis': result['text_analysis'],
                'technical_indicators': technical_indicators
            }
            self.memory.store(memory_data)
        
        # 生成综合信号
        result['comprehensive_signal'] = self._generate_comprehensive_signal(result)
        
        # 记录历史
        self.state_history.append(result)
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
        
        return result
    
    def _calculate_time_series_stats(self, data: np.ndarray) -> Dict:
        """计算时间序列统计"""
        if len(data) == 0:
            return {}
        
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'trend': float((data[-1] - data[0]) / (data[0] + 1e-8) * 100),
            'volatility': float(np.std(data) / (np.mean(data) + 1e-8) * 100)
        }
    
    def _analyze_memory_influence(self, memories: List[Dict]) -> Dict:
        """分析记忆影响"""
        if len(memories) == 0:
            return {'influence_score': 0.0, 'similar_patterns': 0}
        
        # 简化版：计算平均影响
        influence_scores = []
        for mem in memories:
            # 可以根据记忆内容计算影响分数
            score = 0.5  # 占位符
            influence_scores.append(score)
        
        return {
            'influence_score': float(np.mean(influence_scores)),
            'similar_patterns': len(memories),
            'memory_count': len(memories)
        }
    
    def _generate_comprehensive_signal(self, result: Dict) -> Dict:
        """生成综合信号"""
        signals = []
        weights = []
        
        # 时间序列趋势
        ts_stats = result.get('time_series_stats', {})
        if 'trend' in ts_stats:
            trend = ts_stats['trend']
            if trend > 2:
                signals.append('buy')
                weights.append(0.3)
            elif trend < -2:
                signals.append('sell')
                weights.append(0.3)
            else:
                signals.append('hold')
                weights.append(0.2)
        
        # 文本情感
        text_analysis = result.get('text_analysis')
        if text_analysis:
            sentiment = text_analysis.get('sentiment_score', 0.5)
            if sentiment > 0.6:
                signals.append('buy')
                weights.append(0.3)
            elif sentiment < 0.4:
                signals.append('sell')
                weights.append(0.2)
            else:
                signals.append('hold')
                weights.append(0.1)
        
        # 技术指标（如果有）
        indicators = result.get('technical_indicators')
        if indicators:
            # 可以根据具体指标计算信号
            signals.append('hold')
            weights.append(0.2)
        
        # 综合信号
        buy_weight = sum(w for s, w in zip(signals, weights) if s == 'buy')
        sell_weight = sum(w for s, w in zip(signals, weights) if s == 'sell')
        hold_weight = sum(w for s, w in zip(signals, weights) if s == 'hold')
        
        if buy_weight > sell_weight and buy_weight > hold_weight:
            final_signal = 'buy'
            confidence = buy_weight
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            final_signal = 'sell'
            confidence = sell_weight
        else:
            final_signal = 'hold'
            confidence = hold_weight
        
        return {
            'signal': final_signal,
            'confidence': float(confidence),
            'buy_weight': float(buy_weight),
            'sell_weight': float(sell_weight),
            'hold_weight': float(hold_weight)
        }
    
    def get_state_summary(self) -> Dict:
        """获取状态摘要"""
        if len(self.state_history) == 0:
            return {}
        
        recent_states = self.state_history[-10:]
        
        signals = [s.get('comprehensive_signal', {}).get('signal', 'hold') 
                  for s in recent_states]
        
        signal_counts = {
            'buy': signals.count('buy'),
            'sell': signals.count('sell'),
            'hold': signals.count('hold')
        }
        
        return {
            'total_states': len(self.state_history),
            'recent_signals': signal_counts,
            'memory_stats': self.memory.get_statistics() if self.memory else {},
            'latest_signal': self.state_history[-1].get('comprehensive_signal', {}) if self.state_history else {}
        }

