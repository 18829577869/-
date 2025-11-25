# train_v4_final.py - 终极优化训练脚本
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stock_env_v4_final import StockTradingEnv
import random
import os
import numpy as np

# 训练数据集（多只股票）
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

# 测试数据集
test_files = [f.replace("train", "test") for f in stock_files]

# 过滤存在的文件
stock_files = [f for f in stock_files if os.path.exists(f)]
test_files = [f for f in test_files if os.path.exists(f)]

print(f"找到 {len(stock_files)} 只训练股票")
print(f"找到 {len(test_files)} 只测试股票")

if len(stock_files) == 0:
    raise ValueError("没有找到训练数据！请先运行 get_stock_data_v3.py 或 get_stock_data_v4.py")

def make_env():
    """创建环境（随机选择股票）"""
    return StockTradingEnv(random.choice(stock_files))

def make_eval_env():
    """创建评估环境（固定第一只股票）"""
    return StockTradingEnv(stock_files[0])

# ========== 开始训练 ==========
print("\n" + "="*70)
print("开始训练【V4 终极优化版】")
print("优化点：")
print("  ✓ 增加历史窗口（过去5天数据）")
print("  ✓ 增加技术指标（MA5/MA20/RSI/MACD/成交量比）")
print("  ✓ 增加持仓信息到观测（持仓比例、现金比例、收益率、回撤）")
print("  ✓ 平衡的奖励函数（收益+风险+成本）")
print("  ✓ 完善的评估指标（夏普比率、胜率等）")
print("  ✓ 保留真实交易成本（手续费、印花税、滑点）")
print("="*70 + "\n")

# 创建训练环境（16个并行环境）
train_env = DummyVecEnv([make_env for _ in range(16)])

# 创建评估环境
eval_env = DummyVecEnv([make_eval_env])

# 创建检查点回调（每10万步保存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=100000 // 16,  # 因为有16个并行环境
    save_path='./models_v4_final/',
    name_prefix='ppo_stock_v4_final'
)

# 创建评估回调（每5万步评估一次）
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models_v4_final/best/',
    log_path='./logs_v4_final/eval/',
    eval_freq=50000 // 16,
    deterministic=True,
    render=False
)

# 创建PPO模型（优化超参数）
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,              # 折扣因子
    gae_lambda=0.95,         # GAE参数
    clip_range=0.2,          # PPO裁剪范围
    ent_coef=0.01,           # 熵系数（鼓励探索）
    vf_coef=0.5,             # 价值函数系数
    max_grad_norm=0.5,       # 梯度裁剪
    tensorboard_log="./logs_v4_final/"
)

# 训练（300万步，比之前更多）
print("开始训练 3,000,000 步...")
model.learn(
    total_timesteps=3_000_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# 保存最终模型
model.save("ppo_stock_v4_final.zip")
print("\n✅ 训练完成！模型已保存：ppo_stock_v4_final.zip")

# ========== 回测评估 ==========
print("\n" + "="*70)
print("开始在测试集（2025年数据）上回测...")
print("="*70 + "\n")

all_stats = []

for test_file in test_files:
    if not os.path.exists(test_file):
        print(f"⚠️ 文件不存在: {test_file}")
        continue
    
    try:
        env = StockTradingEnv(test_file)
        obs, _ = env.reset()
        done = False
        
        # 静默运行（不打印每一步）
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        # 获取统计数据
        stats = env.get_stats()
        stats['file'] = test_file
        all_stats.append(stats)
        
        # 打印该股票的结果
        name = test_file.split('/')[-1].replace('.csv', '')
        print(f"📊 {name}")
        print(f"   最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"   总收益率: {stats['total_return']:+.2f}%")
        print(f"   最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"   夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"   交易次数: {stats['num_trades']}")
        print(f"   胜率: {stats['win_rate']:.2f}%")
        print(f"   交易天数: {stats['total_days']}")
        print()
        
    except Exception as e:
        print(f"❌ {test_file} 测试失败: {e}\n")

# ========== 汇总统计 ==========
if len(all_stats) > 0:
    print("="*70)
    print("📈 汇总统计")
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
    
    # 找出表现最好和最差的
    best = max(all_stats, key=lambda x: x['total_return'])
    worst = min(all_stats, key=lambda x: x['total_return'])
    
    print(f"\n🏆 最佳表现: {best['file'].split('/')[-1]}")
    print(f"   收益率: {best['total_return']:+.2f}%")
    
    print(f"\n📉 最差表现: {worst['file'].split('/')[-1]}")
    print(f"   收益率: {worst['total_return']:+.2f}%")

print("\n✅ 所有测试完成！")
print(f"💾 模型保存位置: ppo_stock_v4_final.zip")
print(f"📁 训练日志: ./logs_v4_final/")
print(f"📁 模型检查点: ./models_v4_final/")



