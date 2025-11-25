# train_v3.py (优化：增加环境数到8，timesteps到1000000，添加TensorBoard日志)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env_v3 import StockTradingEnv
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

# 训练：8 环境并行
train_env = DummyVecEnv([make_env for _ in range(8)])
model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, batch_size=128, tensorboard_log="logs/")  # 增 batch_size，添加日志
model.learn(total_timesteps=1000000)  # 增到 1000000
model.save("ppo_stock_model_v3.zip")

# 测试（循环多股测试）
test_files = [f.replace('train', 'test') for f in stock_files]
for test_file in test_files:
    test_env = StockTradingEnv(test_file)
    obs, _ = test_env.reset()
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = test_env.step(action)
        test_env.render()
    print(f"{test_file} Final Net Worth:", test_env.net_worth)