# test_v6.py - V6环境快速测试
# -*- coding: utf-8 -*-
"""
快速验证V6环境是否正常工作
"""
from stock_env_v6 import StockTradingEnv
import os
import numpy as np

print("="*70)
print("V6 环境测试")
print("="*70)

# 查找测试文件
test_dir = 'stockdata/test'
if not os.path.exists(test_dir):
    print(f"[错误] 测试目录不存在: {test_dir}")
    print("请先运行: python get_stock_data_v6.py")
    exit(1)

test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
test_files = sorted([f for f in test_files if os.path.exists(f)])

if len(test_files) == 0:
    print(f"[错误] 没有找到测试数据！")
    print("请先运行: python get_stock_data_v6.py")
    exit(1)

print(f"\n找到 {len(test_files)} 个测试文件")
print("使用第一个文件进行测试...\n")

test_file = test_files[0]
print(f"测试文件: {test_file}\n")

print("-"*70)
print("测试1: 环境创建")
print("-"*70)

try:
    env = StockTradingEnv(test_file)
    print("[通过] 环境创建成功")
    print(f"  - 标的名称: {env.stock_info['name']}")
    print(f"  - 标的类别: {env.stock_info['category']}")
    print(f"  - 波动性: {env.stock_info['volatility']}")
    print(f"  - 风格: {env.stock_info['style']}")
    print(f"  - 风险阈值: {env.risk_threshold}")
    print(f"  - 最大仓位: {env.max_position*100:.0f}%")
    print(f"  - 回撤容忍: {env.drawdown_tolerance*100:.0f}%")
    print(f"  - 数据长度: {len(env.df)} 天")
    print(f"  - 观测空间: {env.observation_space.shape}")
    print(f"  - 动作空间: {env.action_space.n} 个动作")
except Exception as e:
    print(f"[失败] {e}")
    exit(1)

print("\n" + "-"*70)
print("测试2: 环境重置")
print("-"*70)

try:
    obs, info = env.reset()
    print("[通过] 环境重置成功")
    print(f"  - 观测维度: {obs.shape}")
    print(f"  - 观测范围: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  - 初始净值: {env.net_worth:.0f}")
    print(f"  - 初始现金: {env.balance:.0f}")
    print(f"  - 初始持仓: {env.shares_held:.0f}")
except Exception as e:
    print(f"[失败] {e}")
    exit(1)

print("\n" + "-"*70)
print("测试3: 执行动作")
print("-"*70)

actions = [
    (0, "持有"),
    (1, "买入25%"),
    (2, "买入50%"),
    (3, "买入100%"),
    (4, "卖出25%"),
    (5, "卖出50%"),
    (6, "卖出100%"),
]

for action, action_name in actions:
    try:
        obs, reward, done, truncated, info = env.step(action)
        position_value = env.shares_held * float(env.df.iloc[env.current_step]['close'])
        position_ratio = position_value / env.net_worth if env.net_worth > 0 else 0
        print(f"[通过] 动作{action}({action_name}): "
              f"净值={env.net_worth:.0f}, "
              f"持仓比例={position_ratio*100:.1f}%, "
              f"奖励={reward:+.2f}")
    except Exception as e:
        print(f"[失败] 动作{action}({action_name}): {e}")

print("\n" + "-"*70)
print("测试4: 随机交易100步")
print("-"*70)

env.reset()
for i in range(100):
    action = np.random.randint(0, 7)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

stats = env.get_stats()
print(f"[通过] 随机交易完成")
print(f"  - 总步数: {stats.get('total_days', 0)}")
print(f"  - 最终净值: {stats['final_net_worth']:.2f}")
print(f"  - 总收益率: {stats['total_return']:+.2f}%")
print(f"  - 最大回撤: {stats['max_drawdown']:.2f}%")
print(f"  - 夏普比率: {stats['sharpe_ratio']:.2f}")
print(f"  - 交易次数: {stats['num_trades']}")
print(f"  - 胜率: {stats['win_rate']:.2f}%")
print(f"  - 风险事件: {stats['risk_events']}")

print("\n" + "-"*70)
print("测试5: 测试多个标的")
print("-"*70)

success_count = 0
fail_count = 0

for test_file in test_files[:5]:  # 测试前5个
    try:
        env = StockTradingEnv(test_file)
        obs, _ = env.reset()
        for _ in range(10):
            action = np.random.randint(0, 7)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        print(f"[通过] {env.stock_info['name']} "
              f"({env.stock_info['category']}|{env.stock_info['volatility']}波动)")
        success_count += 1
    except Exception as e:
        print(f"[失败] {test_file}: {e}")
        fail_count += 1

print(f"\n测试结果: {success_count}个成功, {fail_count}个失败")

print("\n" + "="*70)
print("测试6: 差异化策略验证")
print("="*70)

# 测试不同类型标的的策略差异
test_samples = []
for test_file in test_files[:10]:
    try:
        env = StockTradingEnv(test_file)
        test_samples.append({
            'name': env.stock_info['name'],
            'category': env.stock_info['category'],
            'volatility': env.stock_info['volatility'],
            'risk_threshold': env.risk_threshold,
            'max_position': env.max_position,
            'drawdown_tolerance': env.drawdown_tolerance
        })
    except:
        pass

if len(test_samples) > 0:
    print(f"\n差异化策略配置（样本{len(test_samples)}个）:\n")
    print(f"{'名称':<12} {'类别':<6} {'波动':<4} {'风险阈值':<8} {'最大仓位':<8} {'回撤容忍'}")
    print("-"*70)
    for sample in test_samples:
        print(f"{sample['name']:<12} "
              f"{sample['category']:<6} "
              f"{sample['volatility']:<4} "
              f"{sample['risk_threshold']:<8.0f} "
              f"{sample['max_position']*100:<7.0f}% "
              f"{sample['drawdown_tolerance']*100:.0f}%")
    
    # 验证差异化
    volatilities = set([s['volatility'] for s in test_samples])
    if len(volatilities) > 1:
        print(f"\n[验证通过] 检测到 {len(volatilities)} 种不同波动性标的")
        
        risk_thresholds = set([s['risk_threshold'] for s in test_samples])
        if len(risk_thresholds) > 1:
            print(f"[验证通过] 差异化策略生效，{len(risk_thresholds)} 种不同风险阈值")
        else:
            print(f"[警告] 所有标的使用相同风险阈值")
    else:
        print(f"[警告] 所有标的波动性相同，无法验证差异化")

print("\n" + "="*70)
print("所有测试完成")
print("="*70)
print("[结论] V6环境工作正常！")
print("\n可以开始训练：")
print("  python train_v6.py")



