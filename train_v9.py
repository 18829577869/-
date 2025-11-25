"""
V9训练脚本 - 混合策略版
主策略：V7技术指标（90%）
辅助策略：LLM事件检测（10%）
"""

import os
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stock_env_v9 import StockTradingEnvV9
import random
import numpy as np

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# === 配置参数 ===
STOCK_FILES = [
    "stockdata_v7/train/sh.600036.招商银行.csv",
    "stockdata_v7/train/sh.601838.成都银行.csv",
    "stockdata_v7/train/sh.601318.中国平安.csv",
    "stockdata_v7/train/sh.601939.建设银行.csv",
    "stockdata_v7/train/sh.601398.工商银行.csv",
    "stockdata_v7/train/sz.000858.五粮液.csv",
]

STOCK_FILES = [f for f in STOCK_FILES if os.path.exists(f)]

TOTAL_TIMESTEPS = 3_000_000  # 300万步（V7训练量）
N_ENVS = 8
SAVE_FREQ = 100_000

# PPO 超参数（V7优化后的参数）
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # 适度探索
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": "./logs_v9/"
}


def make_env(stock_files, rank, seed=0):
    def _init():
        stock_file = random.choice(stock_files)
        env = StockTradingEnvV9(
            data_file=stock_file,
            initial_balance=100000,
            llm_provider="deepseek",
            enable_llm_cache=True,
            llm_weight=0.05  # LLM权重5%
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("="*70)
    print("V9混合策略训练")
    print("="*70)
    print(f"\n配置:")
    print(f"  股票数量: {len(STOCK_FILES)}")
    print(f"  总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  并行环境数: {N_ENVS}")
    print(f"  主策略: V7技术指标 (95%)")
    print(f"  辅助策略: LLM事件检测 (5%)")
    print(f"\n股票列表:")
    for f in STOCK_FILES:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # 创建目录
    os.makedirs("./models_v9", exist_ok=True)
    os.makedirs("./logs_v9", exist_ok=True)
    
    # 创建并行环境
    env_fns = [make_env(STOCK_FILES, i, seed=42) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    
    # 创建模型
    model = PPO("MlpPolicy", env, **PPO_PARAMS)
    
    # 训练
    print(f"开始训练 {TOTAL_TIMESTEPS:,} 步...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=None,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[中断] 训练被用户中止")
    
    train_time = time.time() - start_time
    
    # 保存模型
    model_path = "ppo_stock_v9.zip"
    model.save(model_path)
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"训练时长: {train_time/60:.1f} 分钟")
    print(f"最终模型: {model_path}")
    print(f"检查点: ./models_v9/")
    print(f"日志: ./logs_v9/")
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir=./logs_v9/")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    main()

