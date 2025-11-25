"""
完整评估 V8 修复版模型
"""
from stable_baselines3 import PPO
from stock_env_v8 import StockTradingEnvV8
import numpy as np
import pandas as pd
import os

print("="*70)
print("V8 修复版模型 - 完整评估")
print("="*70 + "\n")

# 测试文件
test_files = [
    "stockdata_v7/test/sh.600036.招商银行.csv",
    "stockdata_v7/test/sh.601838.成都银行.csv",
    "stockdata_v7/test/sh.601318.中国平安.csv",
    "stockdata_v7/test/sh.601939.建设银行.csv",
    "stockdata_v7/test/sh.601398.工商银行.csv",
    "stockdata_v7/test/sz.000858.五粮液.csv",
]

test_files = [f for f in test_files if os.path.exists(f)]

print(f"测试股票数量: {len(test_files)}\n")

# 加载模型
model = PPO.load("ppo_stock_v8_fixed.zip")
print("[加载] 模型: ppo_stock_v8_fixed.zip\n")

results = []

for test_file in test_files:
    stock_name = test_file.split('/')[-1].replace('.csv', '')
    
    # 创建环境
    env = StockTradingEnvV8(
        data_file=test_file,
        llm_provider="deepseek"
    )
    
    # 运行测试
    obs, _ = env.reset()
    done = False
    daily_returns = []
    prev_nw = 100000
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        daily_return = (env.net_worth - prev_nw) / prev_nw
        daily_returns.append(daily_return)
        prev_nw = env.net_worth
    
    # 计算指标
    final_net_worth = env.net_worth
    total_return = (final_net_worth - 100000) / 100000 * 100
    max_drawdown = env.max_drawdown * 100
    
    # 夏普比率
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
    
    # 胜率
    win_rate = (env.win_trades / env.total_trades * 100) if env.total_trades > 0 else 0
    
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
    print(f"  收益率: {total_return:+.2f}%")
    print(f"  最大回撤: {max_drawdown:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  交易次数: {env.trade_count}")
    print(f"  胜率: {win_rate:.2f}%")
    print(f"  风险事件: {env.risk_events}\n")

# 汇总统计
df = pd.DataFrame(results)

print("="*70)
print("汇总统计")
print("="*70)
print(f"平均收益率: {df['收益率%'].mean():+.2f}%")
print(f"平均最大回撤: {df['最大回撤%'].mean():.2f}%")
print(f"平均夏普比率: {df['夏普比率'].mean():.2f}")
print(f"平均胜率: {df['胜率%'].mean():.2f}%")
print(f"总交易次数: {df['交易次数'].sum()}")
print(f"总风险事件: {df['风险事件'].sum()}")
print()

# 最佳/最差
best_stock = df.loc[df['收益率%'].idxmax()]
worst_stock = df.loc[df['收益率%'].idxmin()]

print("最佳表现:")
print(f"  {best_stock['股票']}: {best_stock['收益率%']:+.2f}%, "
      f"夏普 {best_stock['夏普比率']:.2f}")

print("\n最差表现:")
print(f"  {worst_stock['股票']}: {worst_stock['收益率%']:+.2f}%, "
      f"夏普 {worst_stock['夏普比率']:.2f}")

# 保存结果
from datetime import datetime
result_file = f"results_v8_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(result_file, index=False, encoding='utf-8-sig')
print(f"\n[保存] 详细结果: {result_file}\n")

# 与 V7 对比
print("="*70)
print("与 V7 对比（参考）")
print("="*70)
print(f"V7 平均收益率: +10.57%")
print(f"V8 Fixed 平均收益率: {df['收益率%'].mean():+.2f}%")
print(f"改进: {df['收益率%'].mean() - 10.57:+.2f} 个百分点")
print()



