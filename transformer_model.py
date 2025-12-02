"""
Transformer模型模块
用于时间序列预测的Transformer架构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，Transformer功能将不可用。请安装: pip install torch")


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (seq_len, batch_size, d_model)
        
        返回:
            添加位置编码后的张量 (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTimeSeriesPredictor(nn.Module):
    """Transformer时间序列预测器"""
    
    def __init__(self,
                 input_size: int = 1,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 output_size: int = 1,
                 max_seq_len: int = 100):
        """
        初始化Transformer模型
        
        参数:
            input_size: 输入特征维度
            d_model: 模型维度（必须能被nhead整除）
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            output_size: 输出维度
            max_seq_len: 最大序列长度
        """
        super(TransformerTimeSeriesPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # 输入投影层（将输入特征映射到d_model维度）
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器-解码器
        # 使用batch_first=True以获得更好的推理性能并消除警告
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 改为True以消除警告
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 改为True以消除警告
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码（防止看到未来信息）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None):
        """
        前向传播
        
        参数:
            src: 源序列 (batch_size, seq_len, input_size) - 使用batch_first格式
            tgt: 目标序列（用于训练，可选）(batch_size, tgt_len, input_size)
        
        返回:
            预测结果 (batch_size, output_size) 或 (batch_size, tgt_len, output_size)
        """
        # 确保输入是(batch_size, seq_len, input_size)格式
        if src.dim() == 2:
            src = src.unsqueeze(0)  # 添加batch维度
        if src.dim() == 3 and src.size(2) != self.input_size:
            # 如果格式是(seq_len, batch_size, input_size)，转换为(batch_size, seq_len, input_size)
            if src.size(0) != self.d_model and src.size(0) > src.size(1):
                src = src.transpose(0, 1)
        
        batch_size = src.size(0)
        src_len = src.size(1)
        
        # 输入投影 (batch_size, seq_len, input_size) -> (batch_size, seq_len, d_model)
        src = self.input_projection(src)
        
        # 位置编码需要(seq_len, batch_size, d_model)格式
        src_transposed = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src_transposed = self.pos_encoder(src_transposed)
        src = src_transposed.transpose(0, 1)  # 转回(batch_size, seq_len, d_model)
        
        # Transformer编码器 (batch_first=True)
        memory = self.transformer_encoder(src)  # (batch_size, seq_len, d_model)
        
        # 如果没有目标序列（推理模式），使用源序列的最后一部分作为目标
        if tgt is None:
            # 生成单步预测，取最后一个时间步
            tgt = src[:, -1:, :]  # (batch_size, 1, d_model)
            tgt_len = 1
        else:
            # 确保tgt是(batch_size, tgt_len, input_size)格式
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(0)
            if tgt.dim() == 3 and tgt.size(2) != self.input_size:
                if tgt.size(0) != self.d_model and tgt.size(0) > tgt.size(1):
                    tgt = tgt.transpose(0, 1)
            tgt = self.input_projection(tgt)  # (batch_size, tgt_len, d_model)
            # 位置编码
            tgt_transposed = tgt.transpose(0, 1)  # (tgt_len, batch_size, d_model)
            tgt_transposed = self.pos_encoder(tgt_transposed)
            tgt = tgt_transposed.transpose(0, 1)  # (batch_size, tgt_len, d_model)
            tgt_len = tgt.size(1)
        
        # 生成掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(src.device)
        
        # Transformer解码器 (batch_first=True)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # (batch_size, tgt_len, d_model)
        
        # 输出投影
        output = self.output_projection(output)  # (batch_size, tgt_len, output_size)
        
        # 如果只有一步预测，压缩维度
        if output.size(1) == 1:
            output = output.squeeze(1)  # (batch_size, output_size)
        
        return output


class TransformerPredictor:
    """Transformer预测器封装类"""
    
    def __init__(self,
                 input_size: int = 1,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 output_size: int = 1,
                 max_seq_len: int = 100,
                 use_gpu: bool = False):
        """
        初始化Transformer预测器
        
        参数:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            output_size: 输出维度
            max_seq_len: 最大序列长度
            use_gpu: 是否使用GPU
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用Transformer功能")
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.model = TransformerTimeSeriesPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_size=output_size,
            max_seq_len=max_seq_len
        ).to(self.device)
        
        self.input_size = input_size
        self.max_seq_len = max_seq_len
        self.is_trained = False
        self.training_history = []
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """数据归一化"""
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
        """反归一化"""
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
            X_train: 训练输入 (num_samples, seq_len, input_size)
            y_train: 训练标签 (num_samples, output_size)
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例
            verbose: 是否显示训练过程
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练
            train_losses = []
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, y_val).item()
            self.model.train()
            
            scheduler.step()
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        self.training_history.append(history)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 输入数据 (num_samples, seq_len, input_size)
        
        返回:
            预测结果
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            return output.cpu().numpy()
    
    def predict_next(self, sequence: np.ndarray) -> float:
        """
        预测下一个时间步的值
        
        参数:
            sequence: 输入序列 (seq_len,)
        
        返回:
            预测值
        """
        if len(sequence) < self.max_seq_len:
            # 如果序列长度不足，用前面的值填充
            padding = np.full(self.max_seq_len - len(sequence), sequence[0])
            sequence = np.concatenate([padding, sequence])
        
        # 取最后max_seq_len个值
        seq = sequence[-self.max_seq_len:]
        X = seq.reshape(1, self.max_seq_len, self.input_size)
        
        prediction = self.predict(X)
        return prediction[0, 0] if prediction.ndim > 1 else prediction[0]
    
    def extract_attention_weights(self, X: np.ndarray) -> List[np.ndarray]:
        """
        提取注意力权重（需要修改模型以支持）
        
        参数:
            X: 输入数据
        
        返回:
            注意力权重列表
        """
        # 注意：这需要修改Transformer模型以返回注意力权重
        # 当前版本返回空列表
        return []

