# train_v3_final.py   ← 直接运行这个
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env_v3_final import StockTradingEnv
import random

stock_files = [
    'stockdata/train/sh.600000.浦发银行.csv',
    'stockdata/train/sh.600036.招商银行.csv',
    'stockdata/train/sz.002083.孚日股份.csv',
    'stockdata/train/sz.001389.广合科技.csv',
    'stockdata/train/sh.600418.江淮汽车.csv'
]

def make_env():
    return StockTradingEnv(random.choice(stock_files))

print("开始训练【V3终极防大亏版】，5只股票通用，绝不爆仓！")
env = DummyVecEnv([make_env for _ in range(8)])

model = PPO("MlpPolicy", env, verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./v3_final_logs/")

model.learn(total_timesteps=1_500_000)
model.save("ppo_v3_no_explosion.zip")
print("训练完成！模型已保存：ppo_v3_no_explosion.zip")

# 自动测试全部5只
test_files = [f.replace("train","test") for f in stock_files]
for f in test_files:
    env = StockTradingEnv(f)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        env.render()
    name = f.split('/')[-1].split('.')[1]
    print(f"→ {name} 2025年最终净值: {env.net_worth:,.0f} 元 | 最大回撤: {(env.peak-env.net_worth)/env.peak*100:5.2f}%\n")