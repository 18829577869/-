# stock_env_v9.py - V9混合策略版
# -*- coding: utf-8 -*-
"""
V9 核心设计：
1. 主策略：V7技术指标 + 风险管理（90%权重）
2. 辅助策略：LLM实时事件检测（10%权重）
3. LLM仅在检测到重大事件时介入
4. 保持V7的差异化风险策略
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import os
from llm_market_intelligence import MarketIntelligenceAgent

class StockTradingEnvV9(gym.Env):
    def __init__(self, data_file, initial_balance=100000, 
                 commission_rate=0.00025,
                 min_commission=5, 
                 transfer_fee_rate=0.00001, 
                 stamp_duty_rate=0.0005,
                 min_trade_unit=100, 
                 slippage_rate=0.001, 
                 history_window=5,
                 llm_provider="deepseek",
                 enable_llm_cache=True,
                 llm_weight=0.05):  # LLM权重降低到5%
        super().__init__()
        
        # LLM智能代理（仅用于事件检测）
        self.llm_provider = llm_provider
        self.llm_weight = llm_weight
        self.llm_agent = None
        self.llm_intelligence_cache = {}
        
        if enable_llm_cache:
            try:
                self.llm_agent = MarketIntelligenceAgent(
                    provider=llm_provider,
                    enable_cache=True
                )
                self._preload_llm_intelligence(data_file)
            except Exception as e:
                print(f"[警告] LLM初始化失败: {e}")
                print("[提示] 将仅使用技术指标策略")
        
        # 读取数据
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 识别标的类别（V6/V7策略）
        self.stock_info = self._identify_stock_type(data_file)
        
        # 基础特征
        base_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                        'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        self.df[base_columns] = self.df[base_columns].apply(pd.to_numeric, errors='coerce')
        
        # 计算技术指标和风险指标（V7核心）
        self._add_technical_indicators()
        self._add_risk_indicators()
        
        # V7观测特征
        self.obs_columns = base_columns + [
            'MA5', 'MA20', 'RSI', 'MACD', 'Volume_Ratio',
            'Volatility', 'Volume_Anomaly', 'Consecutive_Down', 
            'Amplitude', 'Gap', 'ATR'
        ]
        
        # 填充缺失值
        self.df = self.df.ffill().bfill()
        self.df = self.df.dropna().reset_index(drop=True)
        
        if len(self.df) < history_window + 50:
            raise ValueError(f"数据不足")
        
        # 交易参数
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_trade_unit = min_trade_unit
        self.slippage_rate = slippage_rate
        self.history_window = history_window
        
        # 根据标的类别设置风险参数（V7差异化策略）
        self._set_risk_params()
        
        # 状态变量
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.peak_net_worth = initial_balance
        self.prev_net_worth = initial_balance
        
        # 统计变量
        self.trade_history = []
        self.net_worth_history = []
        self.daily_returns = []
        self.risk_events = []
        self.risk_event_count = 0
        
        # 归一化参数
        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()
        
        # 动作空间（7个离散动作）
        self.action_space = gym.spaces.Discrete(7)
        # 0=持有, 1=买25%, 2=买50%, 3=买100%, 4=卖25%, 5=卖50%, 6=卖100%
        
        # 观测空间：V7技术特征 + LLM事件信号（轻量级）
        # V7特征: history_window * len(obs_columns) + 6(持仓/风险)
        # LLM特征: 3个关键事件指标（emergency, policy, risk_level）
        # 注意：始终包含LLM特征以保持观测空间一致
        v7_feature_size = self.history_window * len(self.obs_columns) + 6
        llm_feature_size = 3  # 始终为3，即使没有LLM agent也用0填充
        
        obs_shape = (v7_feature_size + llm_feature_size,)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
        
        llm_status = "已加载" if (self.llm_agent and len(self.llm_intelligence_cache) > 0) else "未加载"
        print(f"[V9环境] {os.path.basename(data_file)}")
        print(f"  数据: {len(self.df)} 天")
        print(f"  主策略: V7技术指标 ({100-int(llm_weight*100)}%)")
        print(f"  辅助策略: LLM事件检测 ({int(llm_weight*100)}%) - {llm_status}")
        print(f"  初始资金: {initial_balance:,.0f} 元")
    
    def _preload_llm_intelligence(self, data_file):
        """预加载LLM市场情报"""
        if not self.llm_agent:
            return
        
        try:
            # 读取数据文件的日期范围
            df_temp = pd.read_csv(data_file)
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            dates = df_temp['date'].dt.strftime('%Y-%m-%d').unique()
            
            print(f"[V9] 预加载LLM事件检测...")
            
            # 批量加载
            for date_str in dates:
                intel = self.llm_agent.get_market_intelligence(date_str, force_refresh=False)
                if intel:
                    self.llm_intelligence_cache[date_str] = intel
            
            print(f"[V9] 已加载 {len(self.llm_intelligence_cache)} 天的事件数据")
            
        except Exception as e:
            print(f"[警告] LLM数据预加载失败: {e}")
    
    def _get_llm_event_signal(self, current_date):
        """获取LLM事件信号（仅关键指标）"""
        if not self.llm_agent:
            return [0.0, 0.0, 0.0]
        
        date_str = current_date.strftime('%Y-%m-%d')
        
        if date_str in self.llm_intelligence_cache:
            intel = self.llm_intelligence_cache[date_str]
            return [
                intel.get('emergency_impact_score', 0.0),  # 突发事件
                intel.get('policy_impact_score', 0.0),     # 政策影响
                intel.get('risk_level', 0.5) - 0.5         # 风险等级(归一化)
            ]
        
        return [0.0, 0.0, 0.0]
    
    def _identify_stock_type(self, data_file):
        """识别标的类型（继承V7）"""
        filename = os.path.basename(data_file)
        
        # 银行股（保守型）
        if any(bank in filename for bank in ['银行', '招商', '工商', '建设', '成都']):
            return {
                'category': 'bank',
                'risk_level': 'conservative',
                'volatility': 'low'
            }
        
        # 保险股（稳健型）
        if any(ins in filename for ins in ['平安', '保险']):
            return {
                'category': 'insurance',
                'risk_level': 'moderate',
                'volatility': 'medium'
            }
        
        # 白酒股（高波动）
        if any(liquor in filename for liquor in ['五粮液', '茅台', '泸州']):
            return {
                'category': 'liquor',
                'risk_level': 'aggressive',
                'volatility': 'high'
            }
        
        # 默认（中等）
        return {
            'category': 'other',
            'risk_level': 'moderate',
            'volatility': 'medium'
        }
    
    def _set_risk_params(self):
        """根据标的类别设置风险参数（继承V7差异化策略）"""
        if self.stock_info['risk_level'] == 'conservative':
            # 银行股：保守策略
            self.max_position_pct = 0.8
            self.risk_threshold = 3
            self.max_drawdown_tolerance = 0.15
        elif self.stock_info['risk_level'] == 'aggressive':
            # 高波动股：激进策略
            self.max_position_pct = 0.6
            self.risk_threshold = 4
            self.max_drawdown_tolerance = 0.25
        else:
            # 默认：平衡策略
            self.max_position_pct = 0.7
            self.risk_threshold = 3.5
            self.max_drawdown_tolerance = 0.20
    
    def _add_technical_indicators(self):
        """添加技术指标（继承V7）"""
        self.df['MA5'] = self.df['close'].rolling(window=5, min_periods=1).mean()
        self.df['MA20'] = self.df['close'].rolling(window=20, min_periods=1).mean()
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        
        # 成交量比率
        self.df['Volume_Ratio'] = self.df['volume'] / (
            self.df['volume'].rolling(window=20, min_periods=1).mean() + 1e-10
        )
    
    def _add_risk_indicators(self):
        """添加风险指标（继承V7）"""
        # 波动率
        self.df['Volatility'] = self.df['close'].pct_change().rolling(
            window=20, min_periods=1
        ).std()
        
        # 成交量异常
        vol_mean = self.df['volume'].rolling(window=20, min_periods=1).mean()
        vol_std = self.df['volume'].rolling(window=20, min_periods=1).std()
        self.df['Volume_Anomaly'] = (self.df['volume'] - vol_mean) / (vol_std + 1e-10)
        
        # 连续下跌天数
        self.df['Consecutive_Down'] = (
            self.df['close'].diff() < 0
        ).astype(int).groupby(
            (self.df['close'].diff() >= 0).cumsum()
        ).cumsum()
        
        # 振幅
        self.df['Amplitude'] = (
            (self.df['high'] - self.df['low']) / self.df['close']
        )
        
        # 跳空缺口
        self.df['Gap'] = (
            (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)
        ).abs()
        
        # ATR
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
    
    def _calculate_risk_level(self, current_idx):
        """计算当前风险等级（继承V7）"""
        row = self.df.iloc[current_idx]
        
        risk_score = 0
        
        if row['Volatility'] > 0.03:
            risk_score += 1
        if abs(row['Volume_Anomaly']) > 2:
            risk_score += 1
        if row['Consecutive_Down'] >= 3:
            risk_score += 1
        if row['Amplitude'] > 0.05:
            risk_score += 1
        if abs(row['Gap']) > 0.02:
            risk_score += 1
        if row['RSI'] > 70 or row['RSI'] < 30:
            risk_score += 1
        
        return risk_score
    
    def _normalize_obs(self, obs):
        """归一化观测值"""
        obs_range = self.obs_max - self.obs_min
        obs_range = obs_range.replace(0, 1)
        normalized = (obs - self.obs_min) / obs_range
        return np.clip(normalized * 2 - 1, -1, 1)
    
    def _get_observation(self):
        """获取观测值：V7技术特征 + LLM事件信号"""
        if self.current_step < self.history_window:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # V7技术特征
        history_data = []
        for i in range(self.history_window):
            idx = self.current_step - self.history_window + i
            obs_row = self.df.loc[idx, self.obs_columns].values
            normalized = self._normalize_obs(pd.Series(obs_row, index=self.obs_columns))
            history_data.append(normalized.values)
        
        history_flat = np.concatenate(history_data).astype(np.float32)
        
        # 持仓信息
        position_pct = self.shares_held * self.df.loc[self.current_step, 'close'] / self.net_worth
        cash_pct = self.balance / self.net_worth
        profit_pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        
        # 风险等级
        risk_level = self._calculate_risk_level(self.current_step)
        risk_score = risk_level / 6.0
        
        # 标的类别（one-hot）
        category_encoding = 0.5
        if self.stock_info['category'] == 'bank':
            category_encoding = -1.0
        elif self.stock_info['category'] == 'liquor':
            category_encoding = 1.0
        
        portfolio_info = np.array([
            np.clip(float(position_pct), 0, 1),
            np.clip(float(cash_pct), 0, 1),
            np.clip(float(profit_pct), -1, 1),
            np.clip(float(drawdown), 0, 1),
            np.clip(float(risk_score), 0, 1),
            float(category_encoding)
        ], dtype=np.float32)
        
        # LLM事件信号（轻量级）- 始终添加以保持观测空间一致
        if self.llm_agent and len(self.llm_intelligence_cache) > 0:
            current_date = self.df.loc[self.current_step, 'date']
            llm_signal = self._get_llm_event_signal(current_date)
        else:
            llm_signal = [0.0, 0.0, 0.0]
        
        llm_features = np.array(llm_signal, dtype=np.float32)
        
        # 组合观测（始终包含LLM特征以保持一致性）
        full_observation = np.concatenate([history_flat, portfolio_info, llm_features])
        
        return full_observation.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = self.history_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        self.trade_history = []
        self.net_worth_history = [self.initial_balance]
        self.daily_returns = []
        self.risk_events = []
        self.risk_event_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """执行动作（V7策略 + LLM事件调整）"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # 当前价格
        current_price = self.df.loc[self.current_step, 'close']
        
        # 风险检查
        risk_level = self._calculate_risk_level(self.current_step)
        if risk_level >= self.risk_threshold:
            self.risk_events.append(self.current_step)
            self.risk_event_count += 1
        
        # LLM事件检查（仅在检测到重大事件时介入）
        llm_emergency_warning = False
        if self.llm_agent:
            current_date = self.df.loc[self.current_step, 'date']
            llm_signal = self._get_llm_event_signal(current_date)
            # 突发事件得分 < -0.3 或风险等级 > 0.3 时触发预警
            if llm_signal[0] < -0.3 or llm_signal[2] > 0.3:
                llm_emergency_warning = True
        
        # 转换动作为整数
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # 执行交易
        buy_ratios = {0: 0, 1: 0.25, 2: 0.5, 3: 1.0}
        sell_ratios = {0: 0, 4: 0.25, 5: 0.5, 6: 1.0}
        
        if action in buy_ratios and buy_ratios[action] > 0:
            # 买入逻辑（考虑风险和LLM预警）
            if llm_emergency_warning or risk_level >= self.risk_threshold:
                # 事件预警时减少买入
                buy_ratio = buy_ratios[action] * 0.3
            else:
                buy_ratio = buy_ratios[action]
            
            max_position_value = self.net_worth * self.max_position_pct
            current_position_value = self.shares_held * current_price
            available_position = max(0, max_position_value - current_position_value)
            
            buy_value = min(self.balance * buy_ratio, available_position)
            if buy_value > self.min_commission:
                shares_to_buy = int(buy_value / current_price / self.min_trade_unit) * self.min_trade_unit
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.slippage_rate)
                    commission = max(cost * self.commission_rate, self.min_commission)
                    total_cost = cost + commission
                    
                    if total_cost <= self.balance:
                        self.balance -= total_cost
                        self.shares_held += shares_to_buy
                        self.trade_history.append({
                            'step': self.current_step,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': total_cost
                        })
        
        elif action in sell_ratios and sell_ratios[action] > 0:
            # 卖出逻辑（LLM紧急预警时优先卖出）
            if llm_emergency_warning:
                sell_ratio = min(sell_ratios[action] * 1.5, 1.0)  # 紧急情况加大卖出
            else:
                sell_ratio = sell_ratios[action]
            
            shares_to_sell = int(self.shares_held * sell_ratio / self.min_trade_unit) * self.min_trade_unit
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.slippage_rate)
                commission = max(revenue * self.commission_rate, self.min_commission)
                stamp_duty = revenue * self.stamp_duty_rate
                transfer_fee = shares_to_sell * self.transfer_fee_rate
                total_cost = commission + stamp_duty + transfer_fee
                
                self.balance += (revenue - total_cost)
                self.shares_held -= shares_to_sell
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue - total_cost
                })
        
        # 更新状态
        self.current_step += 1
        next_price = self.df.loc[self.current_step, 'close']
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * next_price
        
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        self.net_worth_history.append(self.net_worth)
        
        # 计算收益（V7策略为主）
        reward = self._calculate_reward_v9(action, llm_emergency_warning)
        
        # 检查是否结束
        done = self.current_step >= len(self.df) - 1
        terminated = done
        truncated = False
        
        # 计算最终指标
        if done:
            self._calculate_final_metrics()
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _calculate_reward_v9(self, action, llm_emergency_warning):
        """
        V9奖励函数：V7策略为主 + LLM事件调整
        """
        # 基础收益
        net_worth_change = self.net_worth - self.prev_net_worth
        profit_reward = net_worth_change / self.initial_balance * 10
        
        # 持仓奖励（鼓励持有优质资产）
        position_value = self.shares_held * self.df.loc[self.current_step, 'close']
        if position_value > 0:
            hold_reward = 0.1
        else:
            hold_reward = -0.05  # 轻微惩罚空仓
        
        # 交易奖励（鼓励适度交易）
        if action != 0:
            trade_reward = 0.2
        else:
            trade_reward = 0
        
        # 回撤惩罚（V7风险管理）
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        if drawdown > self.max_drawdown_tolerance:
            drawdown_penalty = -5.0 * (drawdown - self.max_drawdown_tolerance)
        elif drawdown > 0.1:
            drawdown_penalty = -1.0 * drawdown
        else:
            drawdown_penalty = 0
        
        # LLM事件调整（辅助信号）
        llm_bonus = 0
        if self.llm_agent and llm_emergency_warning:
            # 紧急事件时，奖励保守操作
            if action == 0:  # 持有现金
                llm_bonus = 0.3
            elif action in [5, 6]:  # 卖出
                llm_bonus = 0.5
            elif action in [1, 2, 3]:  # 买入（惩罚）
                llm_bonus = -0.5
        
        # 总奖励（V7为主 90% + LLM辅助 10%）
        reward = (
            profit_reward + 
            hold_reward + 
            trade_reward + 
            drawdown_penalty +
            llm_bonus * self.llm_weight * 10  # LLM权重缩放
        )
        
        return reward
    
    def _calculate_final_metrics(self):
        """计算最终指标"""
        # 最大回撤
        peak = self.net_worth_history[0]
        max_dd = 0
        for nw in self.net_worth_history:
            if nw > peak:
                peak = nw
            dd = (peak - nw) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        self.max_drawdown = max_dd * 100
        
        # 夏普比率
        if len(self.net_worth_history) > 1:
            returns = np.diff(self.net_worth_history) / self.net_worth_history[:-1]
            self.sharpe_ratio = (
                np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            )
        else:
            self.sharpe_ratio = 0
    
    def render(self, mode='human'):
        """渲染（可选）"""
        pass

