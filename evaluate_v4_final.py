# evaluate_v4_final.py - 详细的回测评估脚本
"""
使用方法：
python evaluate_v4_final.py --model ppo_stock_v4_final.zip --data stockdata/test/sh.600036.招商银行.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from stable_baselines3 import PPO
from stock_env_v4_final import StockTradingEnv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model_path, data_file, render=False):
    """评估模型并返回详细统计"""
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)
    
    # 创建环境
    print(f"加载数据: {data_file}")
    env = StockTradingEnv(data_file)
    
    # 运行回测
    obs, _ = env.reset()
    done = False
    
    actions_taken = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        obs, reward, done, truncated, _ = env.step(action)
        if render:
            env.render()
    
    # 获取统计数据
    stats = env.get_stats()
    
    # 获取净值曲线
    net_worth_curve = env.net_worth_history
    
    # 获取交易记录
    trade_history = env.trade_history
    
    # 计算基准收益（买入持有策略）
    df = env.df
    initial_price = float(df.iloc[env.history_window]['close'])
    final_price = float(df.iloc[-1]['close'])
    buy_hold_return = (final_price / initial_price - 1) * 100
    
    return {
        'stats': stats,
        'net_worth_curve': net_worth_curve,
        'trade_history': trade_history,
        'actions': actions_taken,
        'buy_hold_return': buy_hold_return,
        'dates': df['date'].values[env.history_window:]
    }

def plot_results(results, save_path='evaluation_result.png'):
    """绘制评估结果"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    stats = results['stats']
    net_worth = results['net_worth_curve']
    trades = results['trade_history']
    dates = results['dates'][:len(net_worth)]
    
    # 1. 净值曲线
    ax1 = axes[0]
    ax1.plot(dates, net_worth, label='策略净值', linewidth=2, color='blue')
    ax1.axhline(y=10000, color='gray', linestyle='--', label='初始资金')
    ax1.set_title(f'净值曲线 | 最终收益: {stats["total_return"]:.2f}%', fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值（元）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 标注买卖点
    for trade in trades:
        if trade['step'] < len(dates):
            date = dates[trade['step']]
            nw = net_worth[trade['step']]
            color = 'red' if trade['action'] == 'BUY' else 'green'
            marker = '^' if trade['action'] == 'BUY' else 'v'
            ax1.scatter(date, nw, color=color, marker=marker, s=50, alpha=0.6)
    
    # 2. 回撤曲线
    ax2 = axes[1]
    peak = 10000
    drawdowns = []
    for nw in net_worth:
        if nw > peak:
            peak = nw
        dd = (peak - nw) / peak * 100
        drawdowns.append(dd)
    
    ax2.fill_between(dates, 0, drawdowns, color='red', alpha=0.3)
    ax2.plot(dates, drawdowns, color='darkred', linewidth=1.5)
    ax2.set_title(f'回撤曲线 | 最大回撤: {stats["max_drawdown"]:.2f}%', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回撤（%）')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. 收益率分布
    ax3 = axes[2]
    returns = np.diff(net_worth) / net_worth[:-1] * 100
    ax3.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_title('日收益率分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('收益率（%）')
    ax3.set_ylabel('频数')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存: {save_path}")
    
def print_detailed_stats(results):
    """打印详细统计"""
    stats = results['stats']
    buy_hold = results['buy_hold_return']
    
    print("\n" + "="*70)
    print("📊 详细评估报告")
    print("="*70)
    
    print("\n【收益指标】")
    print(f"  最终净值: {stats['final_net_worth']:,.2f} 元")
    print(f"  总收益率: {stats['total_return']:+.2f}%")
    print(f"  买入持有收益: {buy_hold:+.2f}%")
    print(f"  超额收益: {stats['total_return'] - buy_hold:+.2f}%")
    
    print("\n【风险指标】")
    print(f"  最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"  夏普比率: {stats['sharpe_ratio']:.2f}")
    if stats['sharpe_ratio'] > 2:
        rating = "优秀 ⭐⭐⭐⭐⭐"
    elif stats['sharpe_ratio'] > 1:
        rating = "良好 ⭐⭐⭐⭐"
    elif stats['sharpe_ratio'] > 0:
        rating = "一般 ⭐⭐⭐"
    else:
        rating = "较差 ⭐⭐"
    print(f"  风险调整后表现: {rating}")
    
    print("\n【交易统计】")
    print(f"  总交易次数: {stats['num_trades']}")
    print(f"  交易天数: {stats['total_days']}")
    print(f"  平均每天交易: {stats['num_trades']/stats['total_days']:.2f} 次")
    print(f"  胜率: {stats['win_rate']:.2f}%")
    
    # 分析动作分布
    actions = results['actions']
    action_types = [int(np.round(a[0])) for a in actions]
    buy_count = action_types.count(2)
    sell_count = action_types.count(0)
    hold_count = action_types.count(1)
    total = len(action_types)
    
    print("\n【动作分布】")
    print(f"  买入: {buy_count} 次 ({buy_count/total*100:.1f}%)")
    print(f"  卖出: {sell_count} 次 ({sell_count/total*100:.1f}%)")
    print(f"  持有: {hold_count} 次 ({hold_count/total*100:.1f}%)")
    
    print("\n【综合评价】")
    if stats['total_return'] > buy_hold and stats['max_drawdown'] < 20:
        print("  ✅ 策略表现优于买入持有，且风险控制良好")
    elif stats['total_return'] > buy_hold:
        print("  ⚠️ 策略收益优于买入持有，但回撤较大，需优化风险控制")
    elif stats['max_drawdown'] < 20:
        print("  ⚠️ 风险控制良好，但收益低于买入持有，需提升收益能力")
    else:
        print("  ❌ 策略表现不佳，收益和风险控制都需要改进")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='评估强化学习股票交易模型')
    parser.add_argument('--model', type=str, default='ppo_stock_v4_final.zip',
                        help='模型文件路径')
    parser.add_argument('--data', type=str, 
                        default='stockdata/test/sh.600036.招商银行.csv',
                        help='测试数据文件')
    parser.add_argument('--render', action='store_true',
                        help='是否打印每一步')
    parser.add_argument('--output', type=str, default='evaluation_result.png',
                        help='输出图表文件名')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("请先运行 train_v4_final.py 训练模型")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print("请先运行 get_stock_data_v3.py 或 get_stock_data_v4.py 下载数据")
        return
    
    # 评估
    results = evaluate_model(args.model, args.data, render=args.render)
    
    # 打印统计
    print_detailed_stats(results)
    
    # 绘图
    plot_results(results, save_path=args.output)
    
    print("\n✅ 评估完成！")

if __name__ == '__main__':
    main()



