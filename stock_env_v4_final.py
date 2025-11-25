# stock_env_v4_final.py - 终极优化版本
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    优化改进：
    1. 增加持仓信息到观测空间
    2. 增加历史窗口（过去5天数据）
    3. 增加技术指标（MA5、MA20、RSI）
    4. 统一的奖励函数（收益+风险平衡）
    5. 完善的统计指标
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
        
        # 最终特征列（包含技术指标）
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
        
        # 动作空间：[action_type, amount]
        # action_type: 0=卖出, 1=持有, 2=买入 (离散化为3档)
        # amount: 0~1 连续值，表示买入/卖出的仓位比例
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, 1]), 
            dtype=np.float32
        )
        
        # 观测空间：历史窗口 × 特征数 + 持仓信息(4维)
        # 持仓信息: [持仓比例, 现金比例, 当前收益率, 当前回撤]
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
        
        # 移动平均线
        self.df['MA5'] = close.rolling(window=5).mean()
        self.df['MA20'] = close.rolling(window=20).mean()
        
        # RSI指标
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标（简化版：快线-慢线）
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        
        # 成交量比率（相对于20日均量）
        ma_volume = volume.rolling(window=20).mean()
        self.df['Volume_Ratio'] = volume / (ma_volume + 1e-8)
    
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 从history_window开始，确保有足够的历史数据
        self.current_step = self.history_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        # 清空统计
        self.trade_history = []
        self.net_worth_history = [self.initial_balance]
        self.daily_returns = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取观测（包含历史窗口+持仓信息）"""
        # 1. 历史窗口的市场数据
        historical_data = []
        for i in range(self.history_window):
            step = self.current_step - self.history_window + i + 1
            row = self.df.iloc[step][self.obs_columns]
            # 归一化到[-1, 1]
            normalized = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
            historical_data.extend(normalized.values)
        
        # 2. 当前持仓信息（4维）
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
        
        # 拼接
        obs = np.array(historical_data + position_info, dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行动作"""
        # 解析动作
        action_type = int(np.clip(np.round(action[0]), 0, 2))  # 0=卖, 1=持有, 2=买
        amount = np.clip(action[1], 0, 1)
        
        current_price = float(self.df.iloc[self.current_step]['close'])
        
        # 执行交易
        total_fee = 0
        if action_type == 2:  # 买入
            total_fee = self._execute_buy(current_price, amount)
        elif action_type == 0:  # 卖出
            total_fee = self._execute_sell(current_price, amount)
        # action_type == 1 时持有，不做任何操作
        
        # 更新净值
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        # 更新峰值
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        # 计算奖励
        reward = self._calculate_reward(total_fee)
        
        # 计算日收益率
        daily_return = (self.net_worth / self.prev_net_worth - 1) if self.prev_net_worth > 0 else 0
        self.daily_returns.append(daily_return)
        
        # 检查是否结束
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 如果亏损超过50%，提前结束（防止爆仓）
        if self.net_worth < self.initial_balance * 0.5:
            done = True
            reward -= 20  # 爆仓重罚
        
        return self._get_obs(), float(reward), done, truncated, {}
    
    def _execute_buy(self, price, amount):
        """执行买入操作"""
        if self.balance < 100:
            return 0
        
        adjusted_price = price * (1 + self.slippage_rate)  # 滑点
        cost = self.balance * amount
        
        if cost < 100:
            return 0
        
        # 计算能买多少股（必须是100的整数倍）
        shares = int(cost / (adjusted_price * self.min_trade_unit)) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        # 计算费用
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
        """执行卖出操作"""
        if self.shares_held < self.min_trade_unit:
            return 0
        
        adjusted_price = price * (1 - self.slippage_rate)  # 滑点
        
        # 计算卖出股数（必须是100的整数倍）
        shares = int(self.shares_held * amount / self.min_trade_unit) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        # 计算费用
        revenue = shares * adjusted_price
        transfer_fee = revenue * self.transfer_fee_rate
        commission = max(self.min_commission, revenue * self.commission_rate)
        stamp_duty = revenue * self.stamp_duty_rate  # 印花税只在卖出时收取
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
    
    def _calculate_reward(self, transaction_fee):
        """
        计算奖励（核心改进）
        平衡收益和风险：
        1. 日收益率奖励（主要）
        2. 回撤惩罚（防止大亏）
        3. 交易成本惩罚（防止频繁交易）
        """
        # 1. 日收益率（放大100倍便于学习）
        daily_return = (self.net_worth / self.prev_net_worth - 1) * 100 if self.prev_net_worth > 0 else 0
        return_reward = daily_return
        
        # 2. 回撤惩罚（渐进式）
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        if drawdown > 0.30:  # 回撤超30% → 重罚
            drawdown_penalty = -15.0
        elif drawdown > 0.20:  # 回撤超20% → 中罚
            drawdown_penalty = -5.0
        elif drawdown > 0.10:  # 回撤超10% → 轻罚
            drawdown_penalty = -1.0
        else:
            drawdown_penalty = 0
        
        # 3. 交易成本惩罚（鼓励减少交易频率）
        fee_penalty = -transaction_fee / 100
        
        # 4. 持续盈利奖励（鼓励稳定增长）
        if self.net_worth > self.initial_balance * 1.1:  # 盈利超10%
            stability_bonus = 0.5
        else:
            stability_bonus = 0
        
        # 总奖励
        total_reward = return_reward + drawdown_penalty + fee_penalty + stability_bonus
        
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
        
        # 最大回撤
        peak = self.initial_balance
        max_dd = 0
        for nw in self.net_worth_history:
            if nw > peak:
                peak = nw
            dd = (peak - nw) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 夏普比率（年化，假设252个交易日）
        daily_returns_array = np.array(self.daily_returns)
        if len(daily_returns_array) > 0 and daily_returns_array.std() > 0:
            sharpe_ratio = (daily_returns_array.mean() / daily_returns_array.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 交易次数
        num_trades = len(self.trade_history)
        
        # 胜率（正收益天数比例）
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



