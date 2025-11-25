# stock_env_v4_final_fixed.py - 修复"不交易"问题的版本
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    修复版本改进：
    1. 改进动作空间为离散动作
    2. 调整奖励函数，鼓励交易和盈利
    3. 增加交易奖励，避免过度保守
    """
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
        
        # 计算技术指标
        self._add_technical_indicators()
        
        # 最终特征列
        self.obs_columns = base_columns + ['MA5', 'MA20', 'RSI', 'MACD', 'Volume_Ratio']
        
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
        
        # 归一化参数
        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()
        
        # 🔧 修复1：改为离散动作空间
        # 动作: 0=持有, 1=买入25%, 2=买入50%, 3=买入100%, 4=卖出25%, 5=卖出50%, 6=卖出100%
        self.action_space = gym.spaces.Discrete(7)
        
        # 观测空间保持不变
        obs_shape = (self.history_window * len(self.obs_columns) + 4,)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
    
    def _add_technical_indicators(self):
        """计算技术指标"""
        close = self.df['close']
        volume = self.df['volume']
        
        self.df['MA5'] = close.rolling(window=5).mean()
        self.df['MA20'] = close.rolling(window=20).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        
        ma_volume = volume.rolling(window=20).mean()
        self.df['Volume_Ratio'] = volume / (ma_volume + 1e-8)
    
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
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取观测"""
        historical_data = []
        for i in range(self.history_window):
            step = self.current_step - self.history_window + i + 1
            row = self.df.iloc[step][self.obs_columns]
            normalized = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
            historical_data.extend(normalized.values)
        
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        
        position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 1
        profit_ratio = (self.net_worth - self.initial_balance) / self.initial_balance
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        position_info = [
            np.clip(position_ratio, 0, 1),
            np.clip(cash_ratio, 0, 1),
            np.clip(profit_ratio, -1, 1),
            np.clip(drawdown, 0, 1)
        ]
        
        obs = np.array(historical_data + position_info, dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行动作"""
        # 🔧 修复2：解析离散动作
        # 0=持有, 1=买入25%, 2=买入50%, 3=买入100%, 4=卖出25%, 5=卖出50%, 6=卖出100%
        current_price = float(self.df.iloc[self.current_step]['close'])
        
        total_fee = 0
        if action == 1:  # 买入25%
            total_fee = self._execute_buy(current_price, 0.25)
        elif action == 2:  # 买入50%
            total_fee = self._execute_buy(current_price, 0.50)
        elif action == 3:  # 买入100%
            total_fee = self._execute_buy(current_price, 1.0)
        elif action == 4:  # 卖出25%
            total_fee = self._execute_sell(current_price, 0.25)
        elif action == 5:  # 卖出50%
            total_fee = self._execute_sell(current_price, 0.50)
        elif action == 6:  # 卖出100%
            total_fee = self._execute_sell(current_price, 1.0)
        # action == 0 时持有，不做任何操作
        
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        # 🔧 修复3：改进奖励函数
        reward = self._calculate_reward_fixed(total_fee, action)
        
        daily_return = (self.net_worth / self.prev_net_worth - 1) if self.prev_net_worth > 0 else 0
        self.daily_returns.append(daily_return)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        if self.net_worth < self.initial_balance * 0.5:
            done = True
            reward -= 50  # 爆仓重罚
        
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
    
    def _calculate_reward_fixed(self, transaction_fee, action):
        """
        🔧 修复版奖励函数
        主要改进：
        1. 增加基础奖励，避免初始奖励为0
        2. 减少回撤惩罚强度
        3. 增加交易奖励，鼓励探索
        4. 根据持仓状态给予奖励
        """
        # 1. 净值变化奖励（核心）
        net_worth_change = self.net_worth - self.prev_net_worth
        return_reward = net_worth_change / 100  # 缩小尺度
        
        # 2. 回撤惩罚（减弱）
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        if drawdown > 0.30:
            drawdown_penalty = -5.0  # 从-15降到-5
        elif drawdown > 0.20:
            drawdown_penalty = -2.0  # 从-5降到-2
        elif drawdown > 0.10:
            drawdown_penalty = -0.5  # 从-1降到-0.5
        else:
            drawdown_penalty = 0
        
        # 3. 持仓奖励（新增）- 鼓励持有股票而不是一直空仓
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        if position_ratio > 0.5:  # 持仓超过50%
            position_bonus = 0.1
        elif position_ratio > 0:  # 有持仓
            position_bonus = 0.05
        else:  # 空仓
            position_bonus = -0.1  # 惩罚一直空仓
        
        # 4. 交易奖励（新增）- 鼓励尝试交易
        if action != 0:  # 如果不是持有动作
            trade_bonus = 0.05  # 小奖励鼓励交易
        else:
            trade_bonus = 0
        
        # 5. 盈利奖励
        if self.net_worth > self.initial_balance * 1.05:  # 盈利超5%
            profit_bonus = 1.0
        elif self.net_worth > self.initial_balance:
            profit_bonus = 0.5
        else:
            profit_bonus = 0
        
        # 总奖励
        total_reward = return_reward + drawdown_penalty + position_bonus + trade_bonus + profit_bonus
        
        return total_reward
    
    def render(self):
        """显示当前状态"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth * 100
        
        date = self.df.iloc[self.current_step]['date'].strftime('%Y-%m-%d')
        
        print(f"日期:{date} | 净值:{self.net_worth:8.0f} | 收益:{profit:+7.0f}({profit_pct:+6.2f}%) | "
              f"持仓:{self.shares_held:6.0f}股({position_value:8.0f}元) | 回撤:{drawdown:5.2f}%")
    
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
            'total_days': len(self.daily_returns)
        }
        
        return stats



