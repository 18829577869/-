# stock_env_v4.py
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=10000, commission_rate=0.00025,
                 min_commission=5, transfer_fee_rate=0.00001, stamp_duty_rate=0.0005,
                 min_trade_unit=100, slippage_rate=0.001):
        super().__init__()
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        self.obs_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                            'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        self.df[self.obs_columns] = self.df[self.obs_columns].apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna().reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"文件 {data_file} 无有效数据")

        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_trade_unit = min_trade_unit
        self.slippage_rate = slippage_rate

        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()

        self.action_space = gym.spaces.Box(low=np.array([1, 0]), high=np.array([3, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.obs_columns),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step][self.obs_columns]
        obs = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
        return obs.values.astype(np.float32)

    def step(self, action):
        action_type = int(np.clip(action[0], 1, 3))   # 1=buy, 2=sell, 3=hold
        amount = np.clip(action[1], 0, 1)
        current_price = float(self.df.iloc[self.current_step]['close'])

        slippage = current_price * self.slippage_rate
        commission = transfer_fee = stamp_duty = 0

        if action_type == 1:  # Buy
            adjusted_price = current_price + slippage
            cost = self.balance * amount
            if cost > 0:
                shares = (cost // (adjusted_price * self.min_trade_unit)) * self.min_trade_unit / adjusted_price
                if shares > 0:
                    transfer_fee = cost * self.transfer_fee_rate
                    commission = max(self.min_commission, cost * self.commission_rate)
                    total_cost = cost + transfer_fee + commission
                    self.balance -= total_cost
                    self.shares_held += shares

        elif action_type == 2:  # Sell
            adjusted_price = current_price - slippage
            shares = (self.shares_held * amount // self.min_trade_unit) * self.min_trade_unit
            if shares > 0:
                revenue = shares * adjusted_price
                transfer_fee = revenue * self.transfer_fee_rate
                commission = max(self.min_commission, revenue * self.commission_rate)
                stamp_duty = revenue * self.stamp_duty_rate
                total_fee = transfer_fee + commission + stamp_duty
                self.balance += revenue - total_fee
                self.shares_held -= shares

        self.net_worth = self.balance + self.shares_held * current_price
        
        # V4 核心：必须除以 5000！！！
        reward = (self.net_worth - self.initial_balance) / 5000

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        print(f"Step: {self.current_step:4d} | NetWorth: {self.net_worth:9.2f} | Shares: {self.shares_held:8.0f} | Cash: {self.balance:9.2f}")