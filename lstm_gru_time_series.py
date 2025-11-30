"""
LSTM/GRU 时间序列处理模块
支持时间序列预测、特征提取和序列分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，LSTM/GRU功能将不可用。请安装: pip install torch")


class LSTMTimesSeriesPredictor(nn.Module):
    """LSTM时间序列预测器"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2, use_bidirectional: bool = False):
        """
        初始化LSTM模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率
            use_bidirectional: 是否使用双向LSTM
        """
        super(LSTMTimesSeriesPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.num_directions = 2 if use_bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        返回:
            输出张量 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        if self.use_bidirectional:
            # 双向LSTM：合并前向和后向的隐藏状态
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # 单向LSTM：使用最后一层的隐藏状态
            out = h_n[-1]
        
        # 全连接层
        output = self.fc(out)
        
        return output


class GRUTimeSeriesPredictor(nn.Module):
    """GRU时间序列预测器"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2, use_bidirectional: bool = False):
        """
        初始化GRU模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: GRU层数
            output_size: 输出维度
            dropout: Dropout比率
            use_bidirectional: 是否使用双向GRU
        """
        super(GRUTimeSeriesPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.num_directions = 2 if use_bidirectional else 1
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        返回:
            输出张量 (batch_size, output_size)
        """
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        
        # 使用最后一个时间步的输出
        if self.use_bidirectional:
            # 双向GRU：合并前向和后向的隐藏状态
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # 单向GRU：使用最后一层的隐藏状态
            out = h_n[-1]
        
        # 全连接层
        output = self.fc(out)
        
        return output


class AttentionMechanism(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, hidden_size: int, attention_dim: Optional[int] = None):
        """
        初始化注意力机制
        
        参数:
            hidden_size: 隐藏层大小
            attention_dim: 注意力维度（如果为None，则使用hidden_size）
        """
        super(AttentionMechanism, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim or hidden_size
        
        # 注意力权重计算层
        self.attention_weights = nn.Linear(hidden_size, self.attention_dim)
        self.attention_context = nn.Linear(self.attention_dim, 1)
        
    def forward(self, lstm_out):
        """
        计算注意力权重
        
        参数:
            lstm_out: LSTM/GRU输出 (batch_size, seq_len, hidden_size)
        
        返回:
            attention_output: 加权后的输出 (batch_size, hidden_size)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # 计算注意力分数
        attention_scores = self.attention_weights(lstm_out)  # (batch_size, seq_len, attention_dim)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.attention_context(attention_scores)  # (batch_size, seq_len, 1)
        
        # 归一化注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # 加权求和
        attention_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        return attention_output, attention_weights.squeeze(-1)


class LSTMAttentionPredictor(nn.Module):
    """带注意力机制的LSTM预测器"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2, use_bidirectional: bool = False):
        """
        初始化带注意力机制的LSTM模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率
            use_bidirectional: 是否使用双向LSTM
        """
        super(LSTMAttentionPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.num_directions = 2 if use_bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        
        # 注意力机制
        self.attention = AttentionMechanism(hidden_size * self.num_directions)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x):
        """
        前向传播（带注意力机制）
        
        参数:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        返回:
            output: 输出张量 (batch_size, output_size)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * num_directions)
        
        # 注意力机制
        attention_output, attention_weights = self.attention(lstm_out)
        
        # 全连接层
        output = self.fc(attention_output)
        
        return output, attention_weights


class TimeSeriesProcessor:
    """时间序列处理器（封装LSTM/GRU功能）"""
    
    def __init__(self, 
                 model_type: str = 'lstm',  # 'lstm', 'gru', 'lstm_attention'
                 seq_length: int = 60,
                 input_size: int = 1,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 use_bidirectional: bool = False,
                 use_gpu: bool = False):
        """
        初始化时间序列处理器
        
        参数:
            model_type: 模型类型 ('lstm', 'gru', 'lstm_attention')
            seq_length: 序列长度（输入时间步数）
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: 层数
            output_size: 输出维度
            dropout: Dropout比率
            use_bidirectional: 是否使用双向
            use_gpu: 是否使用GPU
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用时间序列处理功能")
        
        self.model_type = model_type
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.use_bidirectional = use_bidirectional
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        if model_type == 'lstm':
            self.model = LSTMTimesSeriesPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ).to(self.device)
        elif model_type == 'gru':
            self.model = GRUTimeSeriesPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ).to(self.device)
        elif model_type == 'lstm_attention':
            self.model = LSTMAttentionPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练状态
        self.is_trained = False
        self.training_history = []
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列训练数据
        
        参数:
            data: 一维数组（时间序列数据）
        
        返回:
            X: 输入序列 (num_samples, seq_length, input_size)
            y: 输出序列 (num_samples, output_size)
        """
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            # 输入序列
            seq = data[i:i + self.seq_length]
            X.append(seq)
            # 输出（下一个时间步的值）
            target = data[i + self.seq_length]
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # 重塑为3D (samples, time_steps, features)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], self.input_size))
        
        if y.ndim == 1:
            y = y.reshape((y.shape[0], self.output_size))
        
        return X, y
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        数据归一化
        
        参数:
            data: 原始数据
            method: 归一化方法 ('minmax', 'zscore')
        
        返回:
            normalized_data: 归一化后的数据
            params: 归一化参数（用于反归一化）
        """
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val == 0:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - min_val) / (max_val - min_val)
            params = {'method': 'minmax', 'min': min_val, 'max': max_val}
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - mean) / std
            params = {'method': 'zscore', 'mean': mean, 'std': std}
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return normalized, params
    
    def denormalize(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """
        反归一化
        
        参数:
            data: 归一化后的数据
            params: 归一化参数
        
        返回:
            denormalized_data: 反归一化后的数据
        """
        if params['method'] == 'minmax':
            return data * (params['max'] - params['min']) + params['min']
        elif params['method'] == 'zscore':
            return data * params['std'] + params['mean']
        else:
            raise ValueError(f"不支持的归一化方法: {params['method']}")
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              validation_split: float = 0.2,
              verbose: bool = True) -> Dict:
        """
        训练模型
        
        参数:
            X_train: 训练输入 (num_samples, seq_length, input_size)
            y_train: 训练标签 (num_samples, output_size)
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例
            verbose: 是否显示训练过程
        
        返回:
            history: 训练历史
        """
        self.model.train()
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # 划分训练集和验证集
        val_size = int(len(X_tensor) * validation_split)
        X_val = X_tensor[-val_size:]
        y_val = y_tensor[-val_size:]
        X_train_tensor = X_tensor[:-val_size]
        y_train_tensor = y_tensor[:-val_size]
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练
            train_losses = []
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                if isinstance(output, tuple):
                    output = output[0]  # 如果有注意力机制，取第一个输出
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                if isinstance(val_output, tuple):
                    val_output = val_output[0]
                val_loss = criterion(val_output, y_val).item()
            self.model.train()
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        self.training_history.append(history)
        
        return history
    
    def predict(self, X: np.ndarray, return_attention: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测
        
        参数:
            X: 输入数据 (num_samples, seq_length, input_size)
            return_attention: 是否返回注意力权重（仅对attention模型有效）
        
        返回:
            predictions: 预测结果
            attention_weights: 注意力权重（如果return_attention=True）
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            
            if isinstance(output, tuple):
                predictions = output[0].cpu().numpy()
                attention_weights = output[1].cpu().numpy()
                if return_attention:
                    return predictions, attention_weights
                else:
                    return predictions
            else:
                return output.cpu().numpy()
    
    def predict_next(self, sequence: np.ndarray) -> float:
        """
        预测下一个时间步的值
        
        参数:
            sequence: 输入序列 (seq_length,)
        
        返回:
            预测值
        """
        if len(sequence) < self.seq_length:
            raise ValueError(f"序列长度必须至少为 {self.seq_length}")
        
        # 取最后seq_length个值
        seq = sequence[-self.seq_length:]
        X = seq.reshape(1, self.seq_length, self.input_size)
        
        prediction = self.predict(X)
        return prediction[0, 0] if prediction.ndim > 1 else prediction[0]
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取时间序列特征（使用LSTM/GRU的隐藏状态）
        
        参数:
            data: 时间序列数据
        
        返回:
            features: 提取的特征向量
        """
        self.model.eval()
        
        # 创建序列
        X, _ = self.create_sequences(data)
        
        if len(X) == 0:
            return np.array([])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # 获取LSTM/GRU的隐藏状态
            if hasattr(self.model, 'lstm'):
                lstm_out, (h_n, _) = self.model.lstm(X_tensor)
            elif hasattr(self.model, 'gru'):
                lstm_out, h_n = self.model.gru(X_tensor)
            else:
                raise ValueError("模型不支持特征提取")
            
            # 使用最后一个时间步的隐藏状态作为特征
            if self.use_bidirectional:
                features = torch.cat([h_n[-2], h_n[-1]], dim=1).cpu().numpy()
            else:
                features = h_n[-1].cpu().numpy()
        
        return features

