"""
V8 训练脚本 - LLM 增强版
支持多股票训练，集成市场情报分析
"""

import os
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stock_env_v8 import StockTradingEnvV8
import pandas as pd
import random


# === 配置参数 ===
STOCK_FILES = [
    # 金融板块
    "stockdata_v7/train/sh.600036.招商银行.csv",
    "stockdata_v7/train/sh.601838.成都银行.csv",
    "stockdata_v7/train/sh.601318.中国平安.csv",
    "stockdata_v7/train/sh.601939.建设银行.csv",
    "stockdata_v7/train/sh.601398.工商银行.csv",
    # "stockdata_v7/train/513750.港股通非银ETF.csv",  # 如果有
    
    # 消费板块
    "stockdata_v7/train/sz.000858.五粮液.csv",
    # "stockdata_v7/train/159928.消费ETF.csv",  # 如果有
]

# 检查并过滤存在的文件
STOCK_FILES = [f for f in STOCK_FILES if os.path.exists(f)]

LLM_PROVIDER = "deepseek"  # "deepseek" 或 "grok"
LLM_API_KEY = None  # None 则从环境变量读取，或使用模拟数据
INITIAL_BALANCE = 100000

TOTAL_TIMESTEPS = 3_000_000  # 300万步
N_ENVS = 8  # 并行环境数
SAVE_FREQ = 100_000  # 每10万步保存一次

# PPO 超参数
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # 探索系数
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1
}


def make_env(stock_files, rank, seed=0):
    """
    创建环境的工厂函数
    
    Args:
        stock_files: 股票文件列表
        rank: 进程编号
        seed: 随机种子
    """
    def _init():
        stock_file = random.choice(stock_files)
        env = StockTradingEnvV8(
            data_file=stock_file,
            initial_balance=INITIAL_BALANCE,
            llm_provider=LLM_PROVIDER,
            llm_api_key=LLM_API_KEY,
            enable_llm_cache=True,
            llm_weight=0.3  # LLM 信号权重
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    """主训练函数"""
    print("=" * 70)
    print("V8 训练 - LLM 增强的股票交易模型")
    print("=" * 70)
    print(f"\n训练配置:")
    print(f"  股票数量: {len(STOCK_FILES)}")
    print(f"  初始资金: {INITIAL_BALANCE:,} 元")
    print(f"  LLM 提供商: {LLM_PROVIDER}")
    print(f"  总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  并行环境: {N_ENVS}")
    print(f"  观察空间: 29 维 (21技术指标 + 8 LLM情报)")
    print(f"  动作空间: 7 个离散动作")
    print(f"\n股票列表:")
    for i, f in enumerate(STOCK_FILES, 1):
        name = f.split('/')[-1].replace('.csv', '')
        print(f"  {i}. {name}")
    print()
    
    # === 创建保存目录 ===
    os.makedirs("models_v8", exist_ok=True)
    os.makedirs("logs_v8", exist_ok=True)
    
    # === 创建训练环境 ===
    print("[1/4] 创建训练环境...")
    start_time = time.time()
    
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(STOCK_FILES, i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(STOCK_FILES, 0)])
    
    print(f"  完成，耗时: {time.time() - start_time:.1f}秒\n")
    
    # === 创建评估环境 ===
    print("[2/4] 创建评估环境...")
    eval_env = DummyVecEnv([make_env(STOCK_FILES, 999)])
    
    # === 初始化模型 ===
    print("[3/4] 初始化 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./logs_v8/",
        **PPO_PARAMS
    )
    
    print(f"  模型参数:")
    for key, value in PPO_PARAMS.items():
        print(f"    {key}: {value}")
    print()
    
    # === 设置回调 ===
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // N_ENVS,
        save_path="./models_v8/",
        name_prefix="ppo_v8_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_v8/",
        log_path="./logs_v8/",
        eval_freq=50_000 // N_ENVS,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # === 开始训练 ===
    print("[4/4] 开始训练...")
    print("=" * 70)
    print()
    
    train_start = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[中断] 用户中断训练")
    
    train_time = time.time() - train_start
    
    # === 保存最终模型 ===
    model_path = "ppo_stock_v8.zip"
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"训练时长: {train_time/3600:.2f} 小时")
    print(f"最终模型: {model_path}")
    print(f"检查点: ./models_v8/")
    print(f"日志: ./logs_v8/")
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir=./logs_v8/")
    print()
    
    env.close()
    eval_env.close()


def evaluate():
    """评估训练好的模型"""
    print("\n" + "=" * 70)
    print("V8 模型评估")
    print("=" * 70 + "\n")
    
    # 查找测试文件
    test_files = []
    for train_file in STOCK_FILES:
        test_file = train_file.replace("/train/", "/test/")
        if os.path.exists(test_file):
            test_files.append(test_file)
    
    if not test_files:
        print("[警告] 未找到测试文件")
        return
    
    # 加载模型
    model_path = "ppo_stock_v8.zip"
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return
    
    model = PPO.load(model_path)
    print(f"[加载] 模型: {model_path}\n")
    
    # 评估每只股票
    results = []
    
    for test_file in test_files:
        stock_name = test_file.split('/')[-1].replace('.csv', '')
        
        env = StockTradingEnvV8(
            data_file=test_file,
            initial_balance=INITIAL_BALANCE,
            llm_provider=LLM_PROVIDER,
            enable_llm_cache=True
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
        # 统计结果
        final_net_worth = env.net_worth
        total_return = (final_net_worth - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        max_drawdown = env.max_drawdown * 100
        win_rate = (env.win_trades / env.total_trades * 100) if env.total_trades > 0 else 0
        
        # 计算夏普比率（简化版）
        daily_returns = []
        temp_env = StockTradingEnvV8(
            data_file=test_file,
            initial_balance=INITIAL_BALANCE,
            llm_provider=LLM_PROVIDER,
            enable_llm_cache=True
        )
        obs, _ = temp_env.reset()
        prev_nw = INITIAL_BALANCE
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = temp_env.step(action)
            daily_return = (temp_env.net_worth - prev_nw) / prev_nw
            daily_returns.append(daily_return)
            prev_nw = temp_env.net_worth
        
        import numpy as np
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
        
        results.append({
            "股票": stock_name,
            "最终净值": final_net_worth,
            "收益率%": total_return,
            "最大回撤%": max_drawdown,
            "夏普比率": sharpe,
            "交易次数": env.trade_count,
            "胜率%": win_rate,
            "风险事件": env.risk_events
        })
        
        print(f"[完成] {stock_name}")
        print(f"  最终净值: {final_net_worth:,.0f} 元")
        print(f"  总收益率: {total_return:+.2f}%")
        print(f"  最大回撤: {max_drawdown:.2f}%")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  交易次数: {env.trade_count}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  风险事件: {env.risk_events}\n")
    
    # === 生成汇总报告 ===
    df = pd.DataFrame(results)
    
    print("=" * 70)
    print("汇总统计")
    print("=" * 70)
    print(f"平均收益率: {df['收益率%'].mean():+.2f}%")
    print(f"平均最大回撤: {df['最大回撤%'].mean():.2f}%")
    print(f"平均夏普比率: {df['夏普比率'].mean():.2f}")
    print(f"平均胜率: {df['胜率%'].mean():.2f}%")
    print(f"总交易次数: {df['交易次数'].sum()}")
    print(f"总风险事件: {df['风险事件'].sum()}")
    print()
    
    # 保存结果
    result_file = f"results_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    print(f"[保存] 详细结果: {result_file}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # 仅评估
        evaluate()
    else:
        # 训练 + 评估
        train()
        evaluate()

