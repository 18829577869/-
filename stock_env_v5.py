# stock_env_v5.py - 增强风险感知版本
# -*- coding: utf-8 -*-
"""
V5 版本核心改进：
1. 增加6个风险感知指标
2. 风险预警机制
3. 动态风险调整奖励
4. 智能止损机制
"""
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=10000, commission_rate=0.00025,
                 min_commission=5, transfer_fee_rate=0.00001, stamp_duty_rate=0.0005,
                 min_trade_unit=100, slippage_rate=0.001, history_window=5):
        super().__init__()
        
        # 读取并预处理数据
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 基础特征列
        base_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                        'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        self.df[base_columns] = self.df[base_columns].apply(pd.to_numeric, errors='coerce')
        
        # 计算技术指标和风险指标
        self._add_technical_indicators()
        self._add_risk_indicators()  # 新增：风险指标
        
        # 最终特征列（增加6个风险指标）
        self.obs_columns = base_columns + [
            'MA5', 'MA20', 'RSI', 'MACD', 'Volume_Ratio',
            # 风险指标
            'Volatility', 'Volume_Anomaly', 'Consecutive_Down', 
            'Amplitude', 'Gap', 'ATR'
        ]
        
        # 删除NaN行
        self.df = self.df.dropna().reset_index(drop=True)
        
        if len(self.df) < history_window + 50:
            raise ValueError(f"文件 {data_file} 数据不足")
        
        # 交易参数
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_trade_unit = min_trade_unit
        self.slippage_rate = slippage_rate
        self.history_window = history_window
        
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
        self.risk_events = []  # 新增：风险事件记录
        
        # 归一化参数
        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()
        
        # 离散动作空间
        self.action_space = gym.spaces.Discrete(7)
        
        # 观测空间：历史窗口 × 特征数(24) + 持仓信息(4) + 风险等级(1)
        obs_shape = (self.history_window * len(self.obs_columns) + 5,)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
    
    def _add_technical_indicators(self):
        """计算技术指标"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']
        
        # 移动平均线
        self.df['MA5'] = close.rolling(window=5).mean()
        self.df['MA20'] = close.rolling(window=20).mean()
        
        # RSI指标
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        
        # 成交量比率
        ma_volume = volume.rolling(window=20).mean()
        self.df['Volume_Ratio'] = volume / (ma_volume + 1e-8)
    
    def _add_risk_indicators(self):
        """计算风险指标（V5新增）"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_price = self.df['open']
        preclose = self.df['preclose']
        volume = self.df['volume']
        
        # 1. 波动率（价格波动剧烈程度）
        returns = close.pct_change()
        self.df['Volatility'] = returns.rolling(20).std() * 100
        
        # 2. 成交量异常（放量/缩量）
        volume_ma = volume.rolling(20).mean()
        self.df['Volume_Anomaly'] = volume / (volume_ma + 1e-8)
        
        # 3. 连续下跌天数（负值表示连续上涨）
        price_change = close.diff()
        self.df['Consecutive_Down'] = (price_change < 0).astype(int).rolling(5, min_periods=1).sum()
        
        # 4. 振幅（日内波动幅度）
        self.df['Amplitude'] = (high - low) / (close + 1e-8) * 100
        
        # 5. 跳空缺口（相对于前收盘）
        self.df['Gap'] = ((open_price - preclose) / (preclose + 1e-8) * 100).fillna(0)
        
        # 6. ATR（真实波动幅度，风险度量）
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(14).mean() / (close + 1e-8) * 100
    
    def _assess_risk_level(self):
        """评估当前风险等级（0-4）"""
        current = self.df.iloc[self.current_step]
        risk_score = 0
        warnings = []
        
        # 1. 波动率风险
        if current['Volatility'] > self.df['Volatility'].quantile(0.9):
            risk_score += 1
            warnings.append('HIGH_VOLATILITY')
        
        # 2. 巨量成交（可能变盘）
        if current['Volume_Anomaly'] > 2.0:
            risk_score += 1
            warnings.append('HUGE_VOLUME')
        
        # 3. 连续下跌
        if current['Consecutive_Down'] >= 3:
            risk_score += 1
            warnings.append('CONSECUTIVE_DROP')
        
        # 4. RSI极端值
        if current['RSI'] > 80:
            risk_score += 1
            warnings.append('OVERBOUGHT')
        elif current['RSI'] < 20:
            warnings.append('OVERSOLD')  # 超卖不算风险，可能是机会
        
        # 5. 大幅跳空（可能有重大消息）
        if abs(current['Gap']) > 3:  # 3%以上跳空
            risk_score += 1
            warnings.append('BIG_GAP')
        
        # 6. ATR异常（波动突然放大）
        if current['ATR'] > self.df['ATR'].quantile(0.85):
            risk_score += 1
            warnings.append('HIGH_ATR')
        
        if len(warnings) > 0:
            self.risk_events.append({
                'step': self.current_step,
                'risk_level': risk_score,
                'warnings': warnings
            })
        
        return risk_score, warnings
    
    def reset(self, *, seed=None, options=None):
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
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取观测（包含风险等级）"""
        # 1. 历史窗口的市场数据
        historical_data = []
        for i in range(self.history_window):
            step = self.current_step - self.history_window + i + 1
            row = self.df.iloc[step][self.obs_columns]
            normalized = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
            historical_data.extend(normalized.values)
        
        # 2. 当前持仓信息
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        
        position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 1
        profit_ratio = (self.net_worth - self.initial_balance) / self.initial_balance
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        # 3. 当前风险等级（归一化到0-1）
        risk_level, _ = self._assess_risk_level()
        risk_normalized = risk_level / 6.0  # 最大风险分6分
        
        position_info = [
            np.clip(position_ratio, 0, 1),
            np.clip(cash_ratio, 0, 1),
            np.clip(profit_ratio, -1, 1),
            np.clip(drawdown, 0, 1),
            np.clip(risk_normalized, 0, 1)  # 新增：风险等级
        ]
        
        obs = np.array(historical_data + position_info, dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行动作"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        risk_level, warnings = self._assess_risk_level()
        
        # 根据风险等级动态调整动作（高风险时限制买入）
        if risk_level >= 3 and action in [1, 2, 3]:  # 高风险时限制买入
            action = 0  # 强制改为持有
        
        total_fee = 0
        if action == 1:
            total_fee = self._execute_buy(current_price, 0.25)
        elif action == 2:
            total_fee = self._execute_buy(current_price, 0.50)
        elif action == 3:
            total_fee = self._execute_buy(current_price, 1.0)
        elif action == 4:
            total_fee = self._execute_sell(current_price, 0.25)
        elif action == 5:
            total_fee = self._execute_sell(current_price, 0.50)
        elif action == 6:
            total_fee = self._execute_sell(current_price, 1.0)
        
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        # 计算奖励（整合风险信息）
        reward = self._calculate_reward_with_risk(total_fee, action, risk_level, warnings)
        
        daily_return = (self.net_worth / self.prev_net_worth - 1) if self.prev_net_worth > 0 else 0
        self.daily_returns.append(daily_return)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 智能止损（V5改进）
        if self.net_worth < self.initial_balance * 0.5:
            done = True
            reward -= 50
        
        return self._get_obs(), float(reward), done, truncated, {}
    
    def _execute_buy(self, price, amount):
        """执行买入"""
        if self.balance < 100:
            return 0
        
        adjusted_price = price * (1 + self.slippage_rate)
        cost = self.balance * amount
        
        if cost < 100:
            return 0
        
        shares = int(cost / (adjusted_price * self.min_trade_unit)) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        actual_cost = shares * adjusted_price
        transfer_fee = actual_cost * self.transfer_fee_rate
        commission = max(self.min_commission, actual_cost * self.commission_rate)
        total_fee = transfer_fee + commission
        total_cost = actual_cost + total_fee
        
        if total_cost <= self.balance:
            self.balance -= total_cost
            self.shares_held += shares
            self.trade_history.append({
                'step': self.current_step,
                'action': 'BUY',
                'shares': shares,
                'price': adjusted_price,
                'fee': total_fee
            })
            return total_fee
        
        return 0
    
    def _execute_sell(self, price, amount):
        """执行卖出"""
        if self.shares_held < self.min_trade_unit:
            return 0
        
        adjusted_price = price * (1 - self.slippage_rate)
        shares = int(self.shares_held * amount / self.min_trade_unit) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        revenue = shares * adjusted_price
        transfer_fee = revenue * self.transfer_fee_rate
        commission = max(self.min_commission, revenue * self.commission_rate)
        stamp_duty = revenue * self.stamp_duty_rate
        total_fee = transfer_fee + commission + stamp_duty
        
        self.balance += revenue - total_fee
        self.shares_held -= shares
        self.trade_history.append({
            'step': self.current_step,
            'action': 'SELL',
            'shares': shares,
            'price': adjusted_price,
            'fee': total_fee
        })
        
        return total_fee
    
    def _calculate_reward_with_risk(self, transaction_fee, action, risk_level, warnings):
        """
        V5核心：整合风险的奖励函数
        """
        # 1. 净值变化奖励
        net_worth_change = self.net_worth - self.prev_net_worth
        return_reward = net_worth_change / 100
        
        # 2. 回撤惩罚
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        if drawdown > 0.30:
            drawdown_penalty = -5.0
        elif drawdown > 0.20:
            drawdown_penalty = -2.0
        elif drawdown > 0.10:
            drawdown_penalty = -0.5
        else:
            drawdown_penalty = 0
        
        # 3. 持仓奖励/惩罚（根据风险动态调整）
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        if risk_level >= 3:  # 高风险时
            if position_ratio > 0.5:
                position_bonus = -0.2  # 高风险高仓位惩罚
            elif position_ratio > 0:
                position_bonus = -0.1
            else:
                position_bonus = 0.1  # 空仓奖励
        else:  # 低风险时
            if position_ratio > 0.5:
                position_bonus = 0.1  # 鼓励持仓
            elif position_ratio > 0:
                position_bonus = 0.05
            else:
                position_bonus = -0.1  # 空仓惩罚
        
        # 4. 交易奖励
        if action != 0:
            trade_bonus = 0.05
        else:
            trade_bonus = 0
        
        # 5. 风险应对奖励（新增）
        risk_response_bonus = 0
        if risk_level >= 3:  # 高风险情况
            if action in [4, 5, 6]:  # 卖出操作
                risk_response_bonus = 0.3  # 高风险时卖出给予奖励
            elif action in [1, 2, 3]:  # 买入操作
                risk_response_bonus = -0.3  # 高风险时买入惩罚
        
        # 6. 盈利奖励
        if self.net_worth > self.initial_balance * 1.05:
            profit_bonus = 1.0
        elif self.net_worth > self.initial_balance:
            profit_bonus = 0.5
        else:
            profit_bonus = 0
        
        # 总奖励
        total_reward = (return_reward + drawdown_penalty + position_bonus + 
                       trade_bonus + risk_response_bonus + profit_bonus)
        
        return total_reward
    
    def render(self):
        """显示当前状态"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth * 100
        risk_level, warnings = self._assess_risk_level()
        
        date = self.df.iloc[self.current_step]['date'].strftime('%Y-%m-%d')
        
        risk_str = f"[风险{risk_level}]" if risk_level > 0 else ""
        warning_str = f" {','.join(warnings[:2])}" if warnings else ""
        
        print(f"{date} {risk_str}{warning_str} | 净值:{self.net_worth:8.0f} | "
              f"收益:{profit:+7.0f}({profit_pct:+6.2f}%) | "
              f"持仓:{self.shares_held:6.0f}股 | 回撤:{drawdown:5.2f}%")
    
    def get_stats(self):
        """获取统计指标"""
        if len(self.daily_returns) < 2:
            return {}
        
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        
        peak = self.initial_balance
        max_dd = 0
        for nw in self.net_worth_history:
            if nw > peak:
                peak = nw
            dd = (peak - nw) / peak
            if dd > max_dd:
                max_dd = dd
        
        daily_returns_array = np.array(self.daily_returns)
        if len(daily_returns_array) > 0 and daily_returns_array.std() > 0:
            sharpe_ratio = (daily_returns_array.mean() / daily_returns_array.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        num_trades = len(self.trade_history)
        win_days = sum(1 for r in daily_returns_array if r > 0)
        win_rate = win_days / len(daily_returns_array) if len(daily_returns_array) > 0 else 0
        
        stats = {
            'final_net_worth': self.net_worth,
            'total_return': total_return * 100,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'total_days': len(self.daily_returns),
            'risk_events': len(self.risk_events)  # 新增：风险事件数
        }
        
        return stats



