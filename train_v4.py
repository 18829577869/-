# train_v4.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env_v4 import StockTradingEnv
import random
import os

# 只用你已经成功下载的 3 只最优质 ETF
stock_files = [
    'stockdata/train/159966.SZ.创蓝筹.csv',
    'stockdata/train/159876.SZ.有色基金.csv',
    'stockdata/train/159928.SZ.消费ETF.csv'
]

test_files = [
    'stockdata/test/159966.SZ.创蓝筹.csv',
    'stockdata/test/159876.SZ.有色基金.csv',
    'stockdata/test/159928.SZ.消费ETF.csv'
]

def make_env():
    file = random.choice(stock_files)
    return StockTradingEnv(file)

print("开始训练 V4 模型（200万步 + reward/5000）...")
train_env = DummyVecEnv([make_env for _ in range(16)])

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./logs_v4/"
)

model.learn(total_timesteps=2_000_000)
model.save("ppo_stock_model_v4.zip")
print("训练完成！模型已保存为 ppo_stock_model_v4.zip")

# 测试 2025 年真实数据
print("\n开始在 2025 年数据上回测...")
for test_file in test_files:
    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        continue
    env = StockTradingEnv(test_file)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
    name = test_file.split('/')[-1]
    print(f"\n→ {name} 2025年最终净值: {env.net_worth:,.2f} 元（本金 10,000）\n{'='*60}\n")