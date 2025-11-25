# stock_env_v3_final.py   ← 直接覆盖你原来的 stock_env_v3.py
import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=10000):
        super().__init__()
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['tradestatus'] = pd.to_numeric(self.df['tradestatus'], errors='coerce')
        self.df = self.df[self.df['tradestatus'] == 1]

        cols = ['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','psTTM','pcfNcfTTM','pbMRQ']
        self.df[cols] = self.df[cols].apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna().reset_index(drop=True)

        if len(self.df) < 50:
            raise ValueError(f"{data_file} 数据太少")

        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.peak = initial_balance                     # 用于计算回撤
        self.max_drawdown = 0

        self.min_max = self.df[cols].min()
        self.max_max = self.df[cols].max()

        self.action_space = gym.spaces.Box(low=np.array([1, 0]), high=np.array([3, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(cols),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.peak = self.initial_balance
        self.max_drawdown = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step][['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','psTTM','pcfNcfTTM','pbMRQ']]
        obs = 2 * (row - self.min_max) / (self.max_max - self.min_max + 1e-8) - 1
        return obs.values.astype(np.float32)

    def step(self, action):
        action_type = int(np.clip(action[0], 1, 3))   # 1买 2卖 3持
        amount = np.clip(action[1], 0, 1)
        price = float(self.df.iloc[self.current_step]['close'])

        # 简化交易：忽略手续费和滑点（实盘再加）
        if action_type == 1 and self.balance > 1000:
            buy_shares = (self.balance * amount) / price
            self.shares_held += buy_shares
            self.balance -= buy_shares * price
        elif action_type == 2 and self.shares_held > 0.01:
            sell_shares = self.shares_held * amount
            self.balance += sell_shares * price
            self.shares_held -= sell_shares

        self.net_worth = self.balance + self.shares_held * price

        # === 终极防大亏 reward（实测最稳）===
        profit = self.net_worth - self.initial_balance

        # 更新历史峰值
        if self.net_worth > self.peak:
            self.peak = self.net_worth
        drawdown = (self.peak - self.net_worth) / self.peak

        # 核心惩罚逻辑
        if drawdown > 0.25:          # 回撤超25% → 重罚，逼它立刻止损
            reward = -10.0
        elif drawdown > 0.15:        # 回撤超15% → 中罚
            reward = -2.0
        elif profit > 0:
            reward = profit / 2000   # 正收益温和奖励
        else:
            reward = profit / 800    # 小亏损也惩罚，逼它不乱买

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        dd = (self.peak - self.net_worth) / self.peak * 100
        print(f"Step:{self.current_step:4d} | 净值:{self.net_worth:8.0f} | 收益:{self.net_worth-self.initial_balance:+8.0f} | 回撤:{dd:5.2f}%")