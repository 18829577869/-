"""
V9模型评估脚本
"""

import sys
import os
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stock_env_v9 import StockTradingEnvV9
from collections import Counter

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 测试股票列表
TEST_STOCKS = [
    "stockdata_v7/test/sh.600036.招商银行.csv",
    "stockdata_v7/test/sh.601838.成都银行.csv",
    "stockdata_v7/test/sh.601318.中国平安.csv",
    "stockdata_v7/test/sh.601939.建设银行.csv",
    "stockdata_v7/test/sh.601398.工商银行.csv",
    "stockdata_v7/test/sz.000858.五粮液.csv",
]

TEST_STOCKS = [f for f in TEST_STOCKS if os.path.exists(f)]

MODEL_PATH = "ppo_stock_v9.zip"


def evaluate_model(model, stock_file):
    """评估单只股票"""
    env = StockTradingEnvV9(
        data_file=stock_file,
        initial_balance=100000,
        llm_provider="deepseek",
        enable_llm_cache=True,
        llm_weight=0.05
    )
    
    obs, _ = env.reset()
    done = False
    
    actions_taken = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if isinstance(action, int):
            actions_taken.append(action)
        else:
            actions_taken.append(int(action.item()))
    
    # 获取最终统计
    final_net_worth = env.net_worth
    initial_balance = env.initial_balance
    total_return = ((final_net_worth - initial_balance) / initial_balance) * 100
    
    # 交易次数
    trade_count = len([a for a in actions_taken if a != 0])
    
    # 胜率
    action_counter = Counter(actions_taken)
    total_actions = len(actions_taken)
    buy_actions = sum(action_counter.get(i, 0) for i in [1, 2, 3])
    win_rate = (buy_actions / total_actions * 100) if total_actions else 0
    
    # 指标
    max_drawdown = env.max_drawdown
    sharpe_ratio = env.sharpe_ratio
    risk_events = env.risk_event_count
    
    return {
        'stock_file': stock_file,
        'stock_name': os.path.basename(stock_file).replace('.csv', ''),
        'final_net_worth': final_net_worth,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'risk_events': risk_events,
        'actions': actions_taken
    }


def main():
    print("\n" + "="*70)
    print("V9混合策略模型 - 完整评估")
    print("="*70 + "\n")
    print(f"测试股票数量: {len(TEST_STOCKS)}\n")
    
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        return
    
    print(f"[加载] 模型: {MODEL_PATH}\n")
    model = PPO.load(MODEL_PATH)
    
    # 评估所有股票
    results = []
    for stock_file in TEST_STOCKS:
        result = evaluate_model(model, stock_file)
        results.append(result)
        
        print(f"[完成] {result['stock_name']}")
        print(f"  最终净值: {result['final_net_worth']:,.0f} 元")
        print(f"  收益率: {result['total_return']:+.2f}%")
        print(f"  最大回撤: {result['max_drawdown']:.2f}%")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  交易次数: {result['trade_count']}")
        print(f"  胜率: {result['win_rate']:.2f}%")
        print(f"  风险事件: {result['risk_events']}\n")
    
    # 汇总统计
    print("="*70)
    print("汇总统计")
    print("="*70)
    
    avg_return = sum(r['total_return'] for r in results) / len(results)
    avg_drawdown = sum(r['max_drawdown'] for r in results) / len(results)
    avg_sharpe = sum(r['sharpe_ratio'] for r in results) / len(results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
    total_trades = sum(r['trade_count'] for r in results)
    total_risk_events = sum(r['risk_events'] for r in results)
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    print(f"总交易次数: {total_trades}")
    print(f"总风险事件: {total_risk_events}\n")
    
    # 最佳/最差表现
    best = max(results, key=lambda x: x['total_return'])
    worst = min(results, key=lambda x: x['total_return'])
    
    print(f"最佳表现:")
    print(f"  {best['stock_name']}: {best['total_return']:+.2f}%, 夏普 {best['sharpe_ratio']:.2f}\n")
    print(f"最差表现:")
    print(f"  {worst['stock_name']}: {worst['total_return']:+.2f}%, 夏普 {worst['sharpe_ratio']:.2f}\n")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results_v9_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df = df[['stock_name', 'final_net_worth', 'total_return', 'max_drawdown', 
             'sharpe_ratio', 'trade_count', 'win_rate', 'risk_events']]
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"[保存] 详细结果: {csv_file}\n")
    
    # 版本对比
    print("="*70)
    print("版本对比")
    print("="*70)
    print(f"V7 (技术指标): +10.57%")
    print(f"V8 Fixed (模拟LLM): +8.49%")
    print(f"V8 Real (真实LLM历史): +5.98%")
    print(f"V9 (混合策略): {avg_return:+.2f}%\n")
    
    if avg_return > 10.57:
        print("🏆 V9超越所有版本！")
    elif avg_return > 8.49:
        print("✅ V9超越V8，接近V7水平")
    else:
        print("⚠️ V9需要进一步优化")
    
    print()


if __name__ == "__main__":
    main()

