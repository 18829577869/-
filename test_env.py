from stock_env import StockTradingEnv
import numpy as np

# 用训练集测试
env = StockTradingEnv('stockdata/train/sh.600000.浦发银行.csv')
obs = env.reset()
print("Initial obs:", obs)  # 应为 [-1,1] 间的数组

# 模拟一步 (e.g., buy 50%)
action = np.array([1, 0.5])
next_obs, reward, done, truncated, info = env.step(action)
env.render()  # 打印状态
print("Reward:", reward)