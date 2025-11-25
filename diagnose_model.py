# diagnose_model.py - 诊断模型为什么不交易
# -*- coding: utf-8 -*-
"""
诊断工具：分析模型的动作分布和奖励情况
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stock_env_v4_final import StockTradingEnv
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def diagnose_model(model_path, data_file):
    """诊断模型行为"""
    
    print("="*70)
    print("[诊断] 分析模型行为")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(data_file):
        print(f"[错误] 数据文件不存在: {data_file}")
        return
    
    # 加载模型
    print(f"\n[加载] 模型: {model_path}")
    model = PPO.load(model_path)
    
    print(f"[加载] 数据: {data_file}")
    env = StockTradingEnv(data_file)
    
    # 运行一个完整回合，记录所有动作和奖励
    obs, _ = env.reset()
    done = False
    
    actions = []
    rewards = []
    net_worths = []
    action_means = []
    action_stds = []
    
    print("\n[运行] 开始回测（前50步详细输出）...")
    step = 0
    
    while not done:
        # 获取动作
        action, _states = model.predict(obs, deterministic=False)
        
        # 获取动作分布参数（连续动作空间使用Normal分布）
        try:
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            distribution = model.policy.get_distribution(obs_tensor)
            mean = distribution.distribution.mean.detach().cpu().numpy()
            std = distribution.distribution.stddev.detach().cpu().numpy()
            action_means.append(mean)
            action_stds.append(std)
        except:
            pass  # 如果获取失败就跳过
        
        obs, reward, done, truncated, _ = env.step(action)
        
        actions.append(action)
        rewards.append(reward)
        net_worths.append(env.net_worth)
        
        # 前50步详细输出
        if step < 50:
            action_type = int(np.round(action[0]))
            action_amount = action[1]
            action_name = ["卖出", "持有", "买入"][action_type]
            
            print(f"  步骤 {step+1:3d}: 动作={action_name}({action_amount:.2f}) | "
                  f"奖励={reward:+.3f} | 净值={env.net_worth:.0f}")
        
        step += 1
    
    # 分析结果
    print("\n" + "="*70)
    print("[分析] 诊断结果")
    print("="*70)
    
    # 1. 动作分布
    actions_array = np.array(actions)
    action_types = [int(np.round(a[0])) for a in actions]
    
    buy_count = action_types.count(2)
    sell_count = action_types.count(0)
    hold_count = action_types.count(1)
    total = len(action_types)
    
    print(f"\n[动作分布]")
    print(f"  买入: {buy_count} 次 ({buy_count/total*100:.1f}%)")
    print(f"  卖出: {sell_count} 次 ({sell_count/total*100:.1f}%)")
    print(f"  持有: {hold_count} 次 ({hold_count/total*100:.1f}%)")
    
    if hold_count > total * 0.95:
        print(f"  [问题] 持有动作占比过高！模型学到了'不交易'策略")
    
    # 2. 奖励分析
    rewards_array = np.array(rewards)
    print(f"\n[奖励分析]")
    print(f"  总奖励: {rewards_array.sum():.2f}")
    print(f"  平均奖励: {rewards_array.mean():.4f}")
    print(f"  奖励标准差: {rewards_array.std():.4f}")
    print(f"  最大奖励: {rewards_array.max():.4f}")
    print(f"  最小奖励: {rewards_array.min():.4f}")
    print(f"  正奖励比例: {(rewards_array > 0).sum() / len(rewards_array) * 100:.1f}%")
    
    if abs(rewards_array.mean()) < 0.01:
        print(f"  [问题] 平均奖励接近0！奖励信号太弱")
    
    # 3. 净值分析
    stats = env.get_stats()
    print(f"\n[最终结果]")
    print(f"  最终净值: {stats['final_net_worth']:,.2f} 元")
    print(f"  总收益率: {stats['total_return']:+.2f}%")
    print(f"  最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"  交易次数: {stats['num_trades']}")
    
    if stats['num_trades'] == 0:
        print(f"  [问题] 完全没有交易！")
    
    # 4. 动作分布分析（连续动作空间）
    if len(action_means) > 0:
        action_means_array = np.array(action_means)
        action_stds_array = np.array(action_stds)
        avg_mean = action_means_array.mean(axis=0).flatten()
        avg_std = action_stds_array.mean(axis=0).flatten()
        
        print(f"\n[策略分析] 动作分布参数（连续动作空间）")
        print(f"  action_type 平均值: {float(avg_mean[0]):.4f} (0=卖, 1=持有, 2=买)")
        print(f"  action_amount 平均值: {float(avg_mean[1]):.4f} (0-1比例)")
        print(f"  action_type 标准差: {float(avg_std[0]):.4f}")
        print(f"  action_amount 标准差: {float(avg_std[1]):.4f}")
        print(f"  说明: action_type接近1说明倾向于'持有'，接近0倾向'卖出'")
    else:
        print(f"\n[策略分析] 无法获取动作分布参数")
    
    # 5. 可视化
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 净值曲线
        ax1 = axes[0]
        ax1.plot(net_worths, linewidth=2)
        ax1.axhline(y=10000, color='gray', linestyle='--', label='初始资金')
        ax1.set_title('净值曲线')
        ax1.set_ylabel('净值（元）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 奖励曲线
        ax2 = axes[1]
        ax2.plot(rewards, linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('奖励曲线')
        ax2.set_ylabel('奖励')
        ax2.grid(True, alpha=0.3)
        
        # 动作分布
        ax3 = axes[2]
        action_counts = [buy_count, sell_count, hold_count]
        colors = ['green', 'red', 'gray']
        ax3.bar(['买入', '卖出', '持有'], action_counts, color=colors)
        ax3.set_title('动作分布')
        ax3.set_ylabel('次数')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_diagnosis.png', dpi=150, bbox_inches='tight')
        print(f"\n[保存] 诊断图表已保存: model_diagnosis.png")
    except Exception as e:
        print(f"\n[警告] 图表生成失败: {e}")
    
    # 6. 结论和建议
    print("\n" + "="*70)
    print("[结论]")
    print("="*70)
    
    if hold_count > total * 0.95 and stats['num_trades'] < 5:
        print("\n[诊断] 模型学到了'不交易'策略")
        print("\n[原因分析]")
        print("  1. 初始状态（全现金）的奖励可能为0或很小")
        print("  2. 一旦交易可能导致小亏损，模型学会了避免交易")
        print("  3. 回撤惩罚可能过重，导致模型过度保守")
        print("  4. 奖励信号太弱，无法有效引导学习")
        
        print("\n[建议修复]")
        print("  1. 使用修复版环境: stock_env_v4_final_fixed.py")
        print("  2. 运行: python train_v4_final_fixed.py")
        print("  3. 修复版增加了持仓奖励和交易奖励")
        print("  4. 修复版使用离散动作空间，更容易学习")
    else:
        print("\n[正常] 模型有交易行为，策略基本正常")
    
    print("="*70)

if __name__ == '__main__':
    # 诊断最新训练的模型
    model_path = 'ppo_stock_v4_final.zip'
    data_file = 'stockdata/test/sh.600036.招商银行.csv'
    
    diagnose_model(model_path, data_file)

