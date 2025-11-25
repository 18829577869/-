# train_v2.py (多股训练、增 timesteps)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env_v2 import StockTradingEnv
import random

# 多股训练文件列表
stock_files = [
    'stockdata/train/sh.600000.浦发银行.csv',
    'stockdata/train/sh.600036.招商银行.csv',
    'stockdata/train/sz.002083.孚日股份.csv',
    'stockdata/train/sz.001389.广合科技.csv',
    'stockdata/train/sh.600418.江淮汽车.csv'
]

def make_env():
    file = random.choice(stock_files)  # 随机选股
    return StockTradingEnv(file)

# 训练：4 环境并行
train_env = DummyVecEnv([make_env for _ in range(4)])
model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=500000)  # 增到 500000
model.save("ppo_stock_model_v2.zip")

# 测试（示例用浦发测试，可循环多股）
test_env = StockTradingEnv('stockdata/test/sh.600000.浦发银行.csv')
obs, _ = test_env.reset()
done, truncated = False, False
while not (done or truncated):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = test_env.step(action)
    test_env.render()
print("Final Net Worth:", test_env.net_worth)