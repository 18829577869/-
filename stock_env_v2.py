# stock_env_v2.py (缩放奖励)
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=10000, commission=0.001):
        super(StockTradingEnv, self).__init__()
        self.df = pd.read_csv(data_file)
        print("Loaded CSV shape:", self.df.shape)  # Debug: Check initial load
        print("Columns:", self.df.columns.tolist())  # Debug: Verify column names
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Convert tradestatus to numeric and filter
        self.df['tradestatus'] = pd.to_numeric(self.df['tradestatus'], errors='coerce')
        print("Tradestatus unique values:", self.df['tradestatus'].unique())  # Debug: Check types/values
        self.df = self.df[self.df['tradestatus'] == 1]
        
        self.obs_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        self.df[self.obs_columns] = self.df[self.obs_columns].apply(pd.to_numeric, errors='coerce')
        print("After to_numeric shape:", self.df.shape)  # Debug: Before dropna
        
        self.df = self.df.dropna().reset_index(drop=True)
        print("After dropna shape:", self.df.shape)  # Debug: Final shape
        
        if len(self.df) == 0:
            raise ValueError("No valid trading days in data. Check CSV for missing data, NaNs, or incorrect types.")
        
        self.initial_balance = initial_balance
        self.commission = commission
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        
        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()
        
        self.action_space = gym.spaces.Box(low=np.array([1, 0]), high=np.array([3, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.obs_columns),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Ignore seed and options if not used (your env is deterministic)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        obs = self._get_observation()
        info = {}  # Empty info dict for Gym API compatibility
        return obs, info

    def _get_observation(self):
        row = self.df.iloc[self.current_step][self.obs_columns]
        obs = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
        return obs.values.astype(np.float32)

    def step(self, action):
        action_type = int(np.clip(action[0], 1, 3))
        amount = np.clip(action[1], 0, 1)
        current_price = float(self.df.iloc[self.current_step]['close'])
        
        if action_type == 1:  # Buy
            cost = self.balance * amount
            shares_bought = cost / current_price
            commission_fee = cost * self.commission
            self.shares_held += shares_bought
            self.balance -= (cost + commission_fee)
        elif action_type == 2:  # Sell
            shares_sold = self.shares_held * amount
            revenue = shares_sold * current_price
            commission_fee = revenue * self.commission
            self.shares_held -= shares_sold
            self.balance += (revenue - commission_fee)
        
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.initial_balance
        reward = reward / 1000 if reward > 0 else -0.1  # 缩放正奖励，负惩罚温和化
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        return self._get_observation(), reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}, Net Worth: {self.net_worth:.2f}")