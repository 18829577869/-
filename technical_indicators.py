"""
技术指标计算模块
支持 KDJ、OBV、RSI、MACD 等多种技术指标
参数可配置，支持动态调整
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TechnicalIndicators:
    """技术指标计算类，支持多种指标和参数调整"""
    
    def __init__(self, 
                 # KDJ 参数
                 kdj_period: int = 9,
                 kdj_slow_period: int = 3,
                 kdj_fast_period: int = 3,
                 # RSI 参数
                 rsi_period: int = 14,
                 # MACD 参数
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 # OBV 参数
                 obv_smooth_period: int = 20,
                 # MA 参数
                 ma_periods: list = [5, 10, 20, 60]):
        """
        初始化技术指标计算器
        
        参数:
            kdj_period: KDJ指标的周期（默认9）
            kdj_slow_period: KDJ慢速平滑周期（默认3）
            kdj_fast_period: KDJ快速平滑周期（默认3）
            rsi_period: RSI指标周期（默认14）
            macd_fast: MACD快线周期（默认12）
            macd_slow: MACD慢线周期（默认26）
            macd_signal: MACD信号线周期（默认9）
            obv_smooth_period: OBV平滑周期（默认20）
            ma_periods: 移动平均线周期列表（默认[5,10,20,60]）
        """
        self.kdj_period = kdj_period
        self.kdj_slow_period = kdj_slow_period
        self.kdj_fast_period = kdj_fast_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.obv_smooth_period = obv_smooth_period
        self.ma_periods = ma_periods
    
    def calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """
        计算 KDJ 指标
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        
        返回:
            包含 K、D、J 的字典
        """
        # 计算 RSV（未成熟随机值）
        low_min = low.rolling(window=self.kdj_period, min_periods=1).min()
        high_max = high.rolling(window=self.kdj_period, min_periods=1).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        
        # 计算 K 值（快速指标）
        k = rsv.ewm(alpha=1/self.kdj_fast_period, adjust=False).mean()
        
        # 计算 D 值（慢速指标）
        d = k.ewm(alpha=1/self.kdj_slow_period, adjust=False).mean()
        
        # 计算 J 值
        j = 3 * k - 2 * d
        
        return {
            'K': k,
            'D': d,
            'J': j,
            'RSV': rsv
        }
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        计算 OBV（能量潮）指标
        
        参数:
            close: 收盘价序列
            volume: 成交量序列
        
        返回:
            包含 OBV 和 OBV均线的字典
        """
        # 计算价格变化方向
        price_change = close.diff()
        direction = price_change.apply(lambda x: 1 if x >= 0 else -1)
        
        # 计算 OBV
        obv = (volume * direction).cumsum()
        
        # OBV 平滑（移动平均）
        obv_ma = obv.rolling(window=self.obv_smooth_period, min_periods=1).mean()
        
        return {
            'OBV': obv,
            'OBV_MA': obv_ma,
            'OBV_Ratio': obv / (obv_ma + 1e-10)  # OBV 比率
        }
    
    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        """
        计算 RSI（相对强弱指标）
        
        参数:
            close: 收盘价序列
        
        返回:
            RSI 序列
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        计算 MACD 指标
        
        参数:
            close: 收盘价序列
        
        返回:
            包含 MACD、信号线、柱状图的字典
        """
        # 计算快线和慢线
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        # DIF（差离值）
        dif = ema_fast - ema_slow
        
        # DEA（信号线）
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        
        # MACD柱状图
        histogram = (dif - dea) * 2
        
        return {
            'DIF': dif,
            'DEA': dea,
            'MACD': histogram
        }
    
    def calculate_ma(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        计算移动平均线
        
        参数:
            close: 收盘价序列
        
        返回:
            包含各周期MA的字典
        """
        ma_dict = {}
        for period in self.ma_periods:
            ma_dict[f'MA{period}'] = close.rolling(window=period, min_periods=1).mean()
        return ma_dict
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        参数:
            df: 包含 open, high, low, close, volume 列的 DataFrame
        
        返回:
            添加了所有技术指标的 DataFrame
        """
        result_df = df.copy()
        
        # 数据清理：确保所有数值列都是数字类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in result_df.columns:
                # 转换为数值类型，无法转换的设为NaN
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                # 用前一个有效值填充NaN，如果没有则用0填充
                result_df[col] = result_df[col].ffill().fillna(0)
        
        # 确保必需的列存在
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result_df.columns:
                if col == 'high':
                    result_df[col] = result_df['close']  # 如果没有high，用close代替
                elif col == 'low':
                    result_df[col] = result_df['close']  # 如果没有low，用close代替
                elif col == 'volume':
                    result_df[col] = 0  # 如果没有volume，设为0
            else:
                # 再次确保数据类型正确
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                result_df[col] = result_df[col].ffill().fillna(0)
        
        # 计算 KDJ
        kdj = self.calculate_kdj(result_df['high'], result_df['low'], result_df['close'])
        result_df['KDJ_K'] = kdj['K']
        result_df['KDJ_D'] = kdj['D']
        result_df['KDJ_J'] = kdj['J']
        result_df['KDJ_RSV'] = kdj['RSV']
        
        # 计算 OBV
        obv = self.calculate_obv(result_df['close'], result_df['volume'])
        result_df['OBV'] = obv['OBV']
        result_df['OBV_MA'] = obv['OBV_MA']
        result_df['OBV_Ratio'] = obv['OBV_Ratio']
        
        # 计算 RSI
        result_df['RSI'] = self.calculate_rsi(result_df['close'])
        
        # 计算 MACD
        macd = self.calculate_macd(result_df['close'])
        result_df['MACD_DIF'] = macd['DIF']
        result_df['MACD_DEA'] = macd['DEA']
        result_df['MACD'] = macd['MACD']
        
        # 计算移动平均线
        ma_dict = self.calculate_ma(result_df['close'])
        for key, value in ma_dict.items():
            result_df[key] = value
        
        # 填充缺失值（修复FutureWarning：避免类型降级）
        # 分离数值列和非数值列
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = result_df.select_dtypes(exclude=[np.number]).columns
        
        # 处理数值列：先转换为float，再填充
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                # 使用astype避免类型降级警告
                filled = result_df[col].bfill().fillna(0)
                result_df[col] = filled.astype(float, copy=False)
        
        # 处理非数值列（保留date和time列）
        for col in non_numeric_cols:
            if col not in ['date', 'time'] and col in result_df.columns:
                result_df[col] = result_df[col].bfill().fillna('')
        
        return result_df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """
        获取技术指标摘要（最新值）
        
        参数:
            df: 包含技术指标的 DataFrame
        
        返回:
            技术指标摘要字典
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        summary = {
            'KDJ': {
                'K': float(latest.get('KDJ_K', 0)),
                'D': float(latest.get('KDJ_D', 0)),
                'J': float(latest.get('KDJ_J', 0))
            },
            'OBV': {
                'OBV': float(latest.get('OBV', 0)),
                'OBV_MA': float(latest.get('OBV_MA', 0)),
                'OBV_Ratio': float(latest.get('OBV_Ratio', 1.0))
            },
            'RSI': float(latest.get('RSI', 50)),
            'MACD': {
                'DIF': float(latest.get('MACD_DIF', 0)),
                'DEA': float(latest.get('MACD_DEA', 0)),
                'MACD': float(latest.get('MACD', 0))
            }
        }
        
        # 添加移动平均线
        for period in self.ma_periods:
            key = f'MA{period}'
            if key in df.columns:
                summary[key] = float(latest[key])
        
        return summary

