"""
V8 修复版训练脚本
修复奖励函数，解决"不交易"问题
"""

import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stock_env_v8 import StockTradingEnvV8
import random
import numpy as np

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

TOTAL_TIMESTEPS = 5_000_000  # 500万步（真实数据训练）
N_ENVS = 8
SAVE_FREQ = 100_000

# PPO 超参数（调整以鼓励探索）
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,  # 增加探索
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1
}


def make_env(stock_files, rank, seed=0):
    def _init():
        stock_file = random.choice(stock_files)
        env = StockTradingEnvV8(
            data_file=stock_file,
            initial_balance=100000,
            llm_provider="deepseek",
            enable_llm_cache=True,
            llm_weight=0.3
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("="*70)
    print("V8 真实数据训练 - 使用 DeepSeek API 市场情报")
    print("="*70)
    print(f"\n配置:")
    print(f"  股票数量: {len(STOCK_FILES)}")
    print(f"  总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  并行环境: {N_ENVS}")
    print(f"  探索系数: {PPO_PARAMS['ent_coef']}")
    print(f"\n主要改进:")
    print(f"  1. 增强买入奖励: +0.3")
    print(f"  2. 增强持仓奖励: +0.2")
    print(f"  3. 增强空仓惩罚: -0.5")
    print(f"  4. 放大净值变化奖励: /100")
    print()
    
    os.makedirs("models_v8_fixed", exist_ok=True)
    os.makedirs("logs_v8_fixed", exist_ok=True)
    
    print("[1/3] 创建训练环境...")
    start_time = time.time()
    
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(STOCK_FILES, i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(STOCK_FILES, 0)])
    
    print(f"  完成，耗时: {time.time() - start_time:.1f}秒\n")
    
    print("[2/3] 初始化 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./logs_v8_fixed/",
        **PPO_PARAMS
    )
    
    print("[3/3] 开始训练...")
    print("="*70 + "\n")
    
    train_start = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[中断] 用户中断训练")
    
    train_time = time.time() - train_start
    
    model_path = "ppo_stock_v8_realdata.zip"
    model.save(model_path)
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"训练时长: {train_time/60:.1f} 分钟")
    print(f"最终模型: {model_path}")
    print(f"检查点: ./models_v8_fixed/")
    print(f"日志: ./logs_v8_fixed/")
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir=./logs_v8_fixed/")
    print()
    
    env.close()
    
    # 快速测试
    print("="*70)
    print("快速测试模型...")
    print("="*70 + "\n")
    
    test_file = "stockdata_v7/test/sh.600036.招商银行.csv"
    if os.path.exists(test_file):
        test_env = StockTradingEnvV8(
            data_file=test_file,
            llm_provider="deepseek"
        )
        
        obs, _ = test_env.reset()
        done = False
        actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            # 转换为整数
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            actions.append(action)
        
        # 统计
        from collections import Counter
        action_dist = Counter(actions)
        
        print(f"测试结果 (招商银行):")
        print(f"  最终净值: {test_env.net_worth:,.0f} 元")
        print(f"  收益率: {(test_env.net_worth - 100000) / 100000 * 100:+.2f}%")
        print(f"  交易次数: {test_env.trade_count}")
        print(f"\n动作分布:")
        action_names = ["持有", "买入25%", "买入50%", "买入100%", 
                       "卖出25%", "卖出50%", "卖出100%"]
        for action_id in range(7):
            count = action_dist.get(action_id, 0)
            pct = count / len(actions) * 100
            print(f"    {action_id} ({action_names[action_id]}): {count} 次 ({pct:.1f}%)")
        
        # 检查是否修复
        buy_count = sum(action_dist.get(i, 0) for i in [1, 2, 3])
        if buy_count > 0 and test_env.trade_count > 0:
            print("\n[成功] 模型已经开始交易！")
        else:
            print("\n[警告] 模型仍然不交易，可能需要更长训练时间")
        print()


if __name__ == "__main__":
    main()

