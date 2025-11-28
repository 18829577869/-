import gymnasium as gym  # 使用 Gymnasium 替换 Gym 以避免警告
from stable_baselines3 import PPO

# 加载模型（替换为实际文件路径）
model = PPO.load("ppo_stock_v7.zip")

print("模型加载成功！")