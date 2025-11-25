# train_v4_final_fixed.py - 使用修复版环境的训练脚本
# -*- coding: utf-8 -*-
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stock_env_v4_final_fixed import StockTradingEnv
import random
import os
import numpy as np

# 训练数据集
stock_files = [
    'stockdata/train/sh.600000.浦发银行.csv',
    'stockdata/train/sh.600036.招商银行.csv',
    'stockdata/train/sz.002083.孚日股份.csv',
    'stockdata/train/sz.001389.广合科技.csv',
    'stockdata/train/sh.600418.江淮汽车.csv',
    'stockdata/train/159966.SZ.创蓝筹.csv',
    'stockdata/train/159876.SZ.有色基金.csv',
    'stockdata/train/159928.SZ.消费ETF.csv'
]

test_files = [f.replace("train", "test") for f in stock_files]

stock_files = [f for f in stock_files if os.path.exists(f)]
test_files = [f for f in test_files if os.path.exists(f)]

print(f"找到 {len(stock_files)} 只训练股票")
print(f"找到 {len(test_files)} 只测试股票")

if len(stock_files) == 0:
    raise ValueError("没有找到训练数据！")

def make_env():
    return StockTradingEnv(random.choice(stock_files))

def make_eval_env():
    return StockTradingEnv(stock_files[0])

print("\n" + "="*70)
print("开始训练【V4 Final 修复版】")
print("主要修复：")
print("  [修复] 改为离散动作空间（7个动作）")
print("  [修复] 增加持仓奖励，避免过度空仓")
print("  [修复] 增加交易奖励，鼓励探索")
print("  [修复] 减弱回撤惩罚，避免过度保守")
print("  [修复] 增加净值变化直接奖励")
print("="*70 + "\n")

# 创建训练环境
train_env = DummyVecEnv([make_env for _ in range(16)])
eval_env = DummyVecEnv([make_eval_env])

# 创建检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=100000 // 16,
    save_path='./models_v4_final_fixed/',
    name_prefix='ppo_stock_fixed'
)

# 创建评估回调
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models_v4_final_fixed/best/',
    log_path='./logs_v4_final_fixed/eval/',
    eval_freq=50000 // 16,
    deterministic=True,
    render=False
)

# 创建PPO模型
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
    ent_coef=0.02,  # 提高熵系数，鼓励探索
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs_v4_final_fixed/"
)

print("开始训练 1,500,000 步（先快速测试）...")
model.learn(
    total_timesteps=1_500_000,  # 先训练少一点快速验证
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

model.save("ppo_stock_v4_final_fixed.zip")
print("\n[成功] 训练完成！模型已保存：ppo_stock_v4_final_fixed.zip")

# 回测评估
print("\n" + "="*70)
print("开始在测试集（2025年数据）上回测...")
print("="*70 + "\n")

all_stats = []

for test_file in test_files:
    if not os.path.exists(test_file):
        print(f"[警告] 文件不存在: {test_file}")
        continue
    
    try:
        env = StockTradingEnv(test_file)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        stats['file'] = test_file
        all_stats.append(stats)
        
        name = test_file.split('/')[-1].replace('.csv', '')
        print(f"[结果] {name}")
        print(f"   最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"   总收益率: {stats['total_return']:+.2f}%")
        print(f"   最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"   夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"   交易次数: {stats['num_trades']}")
        print(f"   胜率: {stats['win_rate']:.2f}%")
        print()
        
    except Exception as e:
        print(f"[错误] {test_file} 测试失败: {e}\n")

# 汇总统计
if len(all_stats) > 0:
    print("="*70)
    print("[汇总] 统计")
    print("="*70)
    
    avg_return = np.mean([s['total_return'] for s in all_stats])
    avg_drawdown = np.mean([s['max_drawdown'] for s in all_stats])
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_stats])
    avg_win_rate = np.mean([s['win_rate'] for s in all_stats])
    total_trades = sum([s['num_trades'] for s in all_stats])
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    print(f"总交易次数: {total_trades}")
    print(f"测试股票数: {len(all_stats)}")
    print("="*70)
    
    best = max(all_stats, key=lambda x: x['total_return'])
    worst = min(all_stats, key=lambda x: x['total_return'])
    
    print(f"\n[最佳] {best['file'].split('/')[-1]}")
    print(f"   收益率: {best['total_return']:+.2f}%")
    
    print(f"\n[最差] {worst['file'].split('/')[-1]}")
    print(f"   收益率: {worst['total_return']:+.2f}%")

print("\n[完成] 所有测试完成！")
print(f"[保存] 模型保存位置: ppo_stock_v4_final_fixed.zip")
print(f"[日志] 训练日志: ./logs_v4_final_fixed/")
print(f"[模型] 模型检查点: ./models_v4_final_fixed/")



