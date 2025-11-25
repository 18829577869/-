from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env import StockTradingEnv

# 训练
train_env = DummyVecEnv([lambda: StockTradingEnv('stockdata/train/sh.600000.浦发银行.csv')])
model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=100000)  # 测试时可减到 10000 加速
model.save("ppo_stock_model.zip")

# 测试
test_env = StockTradingEnv('stockdata/test/sh.600000.浦发银行.csv')
obs, _ = test_env.reset()  # 修改：忽略 info
done, truncated = False, False
while not (done or truncated):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = test_env.step(action)
    test_env.render()
print("Final Net Worth:", test_env.net_worth)