"""
V8 - LLM 增强的股票交易环境
集成市场情报分析，提供更全面的市场感知能力
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Optional
from llm_market_intelligence import MarketIntelligenceAgent


class StockTradingEnvV8(gym.Env):
    """
    V8 环境 - 基于 V7，新增 LLM 市场情报增强
    
    新增特征：
    1. 宏观经济评分
    2. 市场情绪评分
    3. 风险等级
    4. 政策影响评分
    5. 突发事件影响
    6. 资金流向评分
    7. 国际联动系数
    8. VIX 恐慌指数
    
    观察空间扩展：
    - 原有技术指标: 21 维
    - LLM 市场情报: 8 维
    - 总计: 29 维
    """
    
    def __init__(
        self,
        data_file: str,
        initial_balance: float = 100000,
        llm_provider: str = "deepseek",
        llm_api_key: Optional[str] = None,
        enable_llm_cache: bool = True,
        llm_weight: float = 0.3  # LLM 信号在决策中的权重
    ):
        super().__init__()
        
        # === 加载股票数据 ===
        self.df = pd.read_csv(data_file)
        self.stock_name = data_file.split('/')[-1].replace('.csv', '')
        
        # 数据预处理
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 处理交易状态
        if 'tradestatus' in self.df.columns:
            self.df['tradestatus'] = pd.to_numeric(self.df['tradestatus'], errors='coerce')
            self.df = self.df[self.df['tradestatus'] == 1]
        
        # 数值列处理
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                       'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['close']).reset_index(drop=True)
        
        if len(self.df) < 100:
            raise ValueError(f"数据不足: {len(self.df)} 条 < 100")
        
        # === 计算技术指标 ===
        self._calculate_technical_indicators()
        
        # === 初始化 LLM 市场情报代理 ===
        self.llm_agent = MarketIntelligenceAgent(
            provider=llm_provider,
            api_key=llm_api_key,
            enable_cache=enable_llm_cache
        )
        self.llm_weight = llm_weight
        
        # 预加载所有日期的市场情报（训练时使用）
        self._preload_market_intelligence()
        
        # === 环境参数 ===
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.cost_basis = 0
        
        # 风险管理
        self.peak = initial_balance
        self.max_drawdown = 0
        self.prev_net_worth = initial_balance
        
        # 交易统计
        self.trade_count = 0
        self.win_trades = 0
        self.total_trades = 0
        self.risk_events = 0
        
        # === 定义动作空间（7个离散动作）===
        self.action_space = gym.spaces.Discrete(7)
        # 0: 持有
        # 1-3: 买入 25%, 50%, 100%
        # 4-6: 卖出 25%, 50%, 100%
        
        # === 定义观察空间（29 维）===
        # 21 维技术指标 + 8 维 LLM 情报
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32
        )
        
        print(f"[V8环境] {self.stock_name}")
        print(f"  数据: {len(self.df)} 天")
        print(f"  LLM: {llm_provider}")
        print(f"  初始资金: {initial_balance:,.0f} 元")
    
    def _calculate_technical_indicators(self):
        """计算技术指标（与 V7 一致）"""
        # 移动平均
        self.df['ma5'] = self.df['close'].rolling(window=5, min_periods=1).mean()
        self.df['ma10'] = self.df['close'].rolling(window=10, min_periods=1).mean()
        self.df['ma20'] = self.df['close'].rolling(window=20, min_periods=1).mean()
        
        # 波动率
        self.df['volatility_5'] = self.df['close'].rolling(window=5, min_periods=1).std()
        self.df['volatility_20'] = self.df['close'].rolling(window=20, min_periods=1).std()
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        
        # 布林带
        self.df['bb_upper'] = self.df['ma20'] + 2 * self.df['volatility_20']
        self.df['bb_lower'] = self.df['ma20'] - 2 * self.df['volatility_20']
        
        # 填充缺失值
        self.df = self.df.bfill()
        self.df = self.df.fillna(0)
    
    def _preload_market_intelligence(self):
        """预加载所有交易日的市场情报"""
        print(f"[V8] 预加载市场情报...")
        
        self.market_intelligence_cache = {}
        unique_dates = self.df['date'].dt.strftime('%Y-%m-%d').unique()
        
        for date_str in unique_dates:
            intelligence = self.llm_agent.get_market_intelligence(date_str)
            self.market_intelligence_cache[date_str] = intelligence
        
        print(f"[V8] 已加载 {len(self.market_intelligence_cache)} 天的市场情报")
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察状态
        
        Returns:
            29维特征向量:
            - [0-20]: 技术指标 (21维)
            - [21-28]: LLM 市场情报 (8维)
        """
        row = self.df.iloc[self.current_step]
        
        # === 技术指标特征 (21维) ===
        price = row['close']
        
        tech_features = [
            row['open'] / price - 1,
            row['high'] / price - 1,
            row['low'] / price - 1,
            row['preclose'] / price - 1 if row['preclose'] > 0 else 0,
            row['volume'] / 1e8,  # 归一化
            row['amount'] / 1e9,
            row['turn'] / 100,
            row['pctChg'] / 100,
            row['ma5'] / price - 1,
            row['ma10'] / price - 1,
            row['ma20'] / price - 1,
            row['volatility_5'] / price,
            row['volatility_20'] / price,
            row['rsi'] / 100 - 0.5,  # 归一化到 [-0.5, 0.5]
            row['macd'] / price,
            row['macd_signal'] / price,
            (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'] + 1e-10),
            # 持仓信息
            self.shares_held * price / (self.initial_balance + 1e-10),
            self.balance / (self.initial_balance + 1e-10),
            (self.net_worth - self.peak) / (self.peak + 1e-10),  # 当前回撤
            self.max_drawdown
        ]
        
        # === LLM 市场情报特征 (8维) ===
        date_str = row['date'].strftime('%Y-%m-%d')
        intelligence = self.market_intelligence_cache.get(date_str)
        
        if intelligence:
            llm_features = self.llm_agent.get_feature_vector(intelligence)
        else:
            # 如果没有缓存，实时获取
            intelligence = self.llm_agent.get_market_intelligence(date_str)
            llm_features = self.llm_agent.get_feature_vector(intelligence)
        
        # 合并特征
        obs = np.array(tech_features + llm_features, dtype=np.float32)
        
        # 处理异常值
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.cost_basis = 0
        
        self.peak = self.initial_balance
        self.max_drawdown = 0
        self.prev_net_worth = self.initial_balance
        
        self.trade_count = 0
        self.win_trades = 0
        self.total_trades = 0
        self.risk_events = 0
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """
        执行动作
        
        Args:
            action: 0持有, 1-3买入(25%/50%/100%), 4-6卖出(25%/50%/100%)
        """
        # 确保 action 是整数（PPO.predict 可能返回数组）
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        row = self.df.iloc[self.current_step]
        price = row['close']
        date_str = row['date'].strftime('%Y-%m-%d')
        
        # 获取市场情报
        intelligence = self.market_intelligence_cache.get(date_str)
        
        # === 执行交易 ===
        trade_executed = False
        action_type = "hold"
        
        if action in [1, 2, 3]:  # 买入
            buy_ratios = {1: 0.25, 2: 0.50, 3: 1.00}
            ratio = buy_ratios[action]
            
            if self.balance > 1000:
                # 考虑 LLM 风险评估
                risk_level = intelligence.get('risk_level', 0.5) if intelligence else 0.5
                
                # 高风险时降低买入比例
                if risk_level > 0.7:
                    ratio *= 0.5
                elif risk_level > 0.5:
                    ratio *= 0.8
                
                buy_amount = self.balance * ratio
                
                # 交易成本
                commission = buy_amount * 0.0003
                transfer_fee = buy_amount * 0.00002
                total_cost = buy_amount + commission + transfer_fee
                
                if total_cost <= self.balance:
                    shares = buy_amount / price
                    self.shares_held += shares
                    self.balance -= total_cost
                    self.cost_basis = (self.cost_basis * (self.shares_held - shares) + buy_amount) / self.shares_held
                    trade_executed = True
                    action_type = f"buy_{int(ratio*100)}%"
                    self.trade_count += 1
        
        elif action in [4, 5, 6]:  # 卖出
            sell_ratios = {4: 0.25, 5: 0.50, 6: 1.00}
            ratio = sell_ratios[action]
            
            if self.shares_held > 0.01:
                sell_shares = self.shares_held * ratio
                sell_amount = sell_shares * price
                
                # 交易成本
                commission = sell_amount * 0.0003
                transfer_fee = sell_amount * 0.00002
                stamp_duty = sell_amount * 0.001
                total_cost = commission + transfer_fee + stamp_duty
                
                self.balance += (sell_amount - total_cost)
                
                # 统计盈亏
                if price > self.cost_basis:
                    self.win_trades += 1
                self.total_trades += 1
                
                self.shares_held -= sell_shares
                trade_executed = True
                action_type = f"sell_{int(ratio*100)}%"
                self.trade_count += 1
        
        # === 更新净值 ===
        self.net_worth = self.balance + self.shares_held * price
        
        # === 计算奖励 ===
        reward = self._calculate_reward_v8(intelligence, trade_executed, action_type)
        
        # === 更新统计 ===
        if self.net_worth > self.peak:
            self.peak = self.net_worth
        
        current_drawdown = (self.peak - self.net_worth) / self.peak
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.prev_net_worth = self.net_worth
        self.current_step += 1
        
        # === 判断是否结束 ===
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "action": action_type,
            "trade_executed": trade_executed,
            "date": date_str,
            "market_risk": intelligence.get('risk_level', 0.5) if intelligence else 0.5,
            "market_sentiment": intelligence.get('market_sentiment_score', 0) if intelligence else 0
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _calculate_reward_v8(
        self,
        intelligence: Optional[Dict],
        trade_executed: bool,
        action_type: str
    ) -> float:
        """
        V8 奖励函数 - 集成 LLM 市场情报（修复版）
        
        奖励组成：
        1. 净值变化奖励（主要驱动力）
        2. 回撤惩罚（风险控制）
        3. 持仓奖励（鼓励持有资产）
        4. 交易激励（鼓励交易）
        5. LLM 情报增强（市场环境匹配）
        """
        reward = 0.0
        price = self.df.iloc[self.current_step]['close']
        position_value = self.shares_held * price
        position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        
        # 1. 净值变化奖励（核心驱动）
        net_worth_change = self.net_worth - self.prev_net_worth
        reward += net_worth_change / 100  # 放大系数，让收益更明显
        
        # 2. 回撤惩罚（重要！）
        current_drawdown = (self.peak - self.net_worth) / self.peak
        if current_drawdown > 0.25:
            reward -= 10.0
            self.risk_events += 1
        elif current_drawdown > 0.15:
            reward -= 3.0
            self.risk_events += 1
        elif current_drawdown > 0.05:
            reward -= 0.5
        
        # 3. 持仓奖励（核心！鼓励持有资产）
        if position_ratio > 0.5:
            reward += 0.2  # 持仓超过 50% 给予奖励
        elif position_ratio > 0.3:
            reward += 0.1  # 持仓超过 30% 给予小奖励
        elif position_ratio < 0.1:
            reward -= 0.5  # 空仓或极低仓位惩罚（增强！）
        
        # 4. 交易激励（鼓励执行交易）
        if trade_executed:
            if "buy" in action_type:
                reward += 0.3  # 买入给予较大奖励（增强！）
            elif "sell" in action_type:
                reward += 0.1  # 卖出给予小奖励
        
        # 5. LLM 情报增强（市场环境匹配）
        if intelligence and self.llm_weight > 0:
            market_sentiment = intelligence.get('market_sentiment_score', 0)
            risk_level = intelligence.get('risk_level', 0.5)
            macro_score = intelligence.get('macro_economic_score', 0)
            
            # 5.1 买入时机匹配
            if "buy" in action_type and trade_executed:
                # 在积极市场环境买入 → 额外奖励
                if market_sentiment > 0.2 and risk_level < 0.6:
                    reward += 0.2 * self.llm_weight
                # 在高风险环境买入 → 轻微惩罚
                elif risk_level > 0.7:
                    reward -= 0.1 * self.llm_weight
            
            # 5.2 卖出时机匹配
            elif "sell" in action_type and trade_executed:
                # 在高风险环境卖出 → 额外奖励
                if risk_level > 0.6 or market_sentiment < -0.2:
                    reward += 0.2 * self.llm_weight
            
            # 5.3 持仓水平与市场环境匹配
            if position_ratio > 0.6:
                # 好环境高仓位 → 奖励
                if macro_score > 0.2 and market_sentiment > 0.1:
                    reward += 0.15 * self.llm_weight
                # 差环境高仓位 → 惩罚
                elif macro_score < -0.2 or market_sentiment < -0.3:
                    reward -= 0.2 * self.llm_weight
            
            # 5.4 突发事件应对
            emergency_impact = intelligence.get('emergency_impact_score', 0)
            if emergency_impact < -0.5:
                # 突发负面事件时，低仓位 → 奖励
                if position_ratio < 0.3:
                    reward += 0.2 * self.llm_weight
                # 突发负面事件时，高仓位 → 惩罚
                elif position_ratio > 0.7:
                    reward -= 0.3 * self.llm_weight
        
        # 6. 盈利里程碑奖励
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        if total_return > 0.10:
            reward += 2.0
        elif total_return > 0.05:
            reward += 1.0
        elif total_return > 0:
            reward += 0.3
        
        return float(reward)
    
    def render(self):
        """打印当前状态"""
        row = self.df.iloc[self.current_step]
        date_str = row['date'].strftime('%Y-%m-%d')
        intelligence = self.market_intelligence_cache.get(date_str, {})
        
        dd = (self.peak - self.net_worth) / self.peak * 100
        return_pct = (self.net_worth - self.initial_balance) / self.initial_balance * 100
        
        sentiment = intelligence.get('market_sentiment_score', 0)
        risk = intelligence.get('risk_level', 0.5)
        
        print(f"日期:{date_str} | 净值:{self.net_worth:>10,.0f} | "
              f"收益:{return_pct:+6.2f}% | 回撤:{dd:5.2f}% | "
              f"情绪:{sentiment:+.2f} | 风险:{risk:.2f}")


# 测试代码
if __name__ == "__main__":
    print("=== V8 环境测试 ===\n")
    
    # 测试环境初始化
    env = StockTradingEnvV8(
        data_file="stockdata_v7/train/sh.600036.招商银行.csv",
        llm_provider="deepseek",
        enable_llm_cache=True
    )
    
    obs, info = env.reset()
    print(f"观察空间维度: {obs.shape}")
    print(f"前 21 维（技术指标）: {obs[:21]}")
    print(f"后 8 维（LLM情报）: {obs[21:]}\n")
    
    # 测试几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        print(f"  动作: {info['action']} | 奖励: {reward:+.3f}\n")
        
        if done:
            break
    
    print("测试完成！")

