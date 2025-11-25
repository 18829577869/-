"""
快速测试 V8 修复版模型
"""
from stable_baselines3 import PPO
from stock_env_v8 import StockTradingEnvV8
import numpy as np
from collections import Counter

print("="*70)
print("快速测试 V8 修复版模型")
print("="*70 + "\n")

# 加载模型
model = PPO.load("ppo_stock_v8_fixed.zip")
print("[加载] 模型: ppo_stock_v8_fixed.zip\n")

# 创建测试环境
env = StockTradingEnvV8(
    data_file="stockdata_v7/test/sh.600036.招商银行.csv",
    llm_provider="deepseek"
)

# 运行测试
obs, _ = env.reset()
actions = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    # 转换为整数
    if isinstance(action, np.ndarray):
        action = int(action.item())
    else:
        action = int(action)
    actions.append(action)

# 统计结果
action_dist = Counter(actions)

print("测试结果 (招商银行):")
print(f"  最终净值: {env.net_worth:,.0f} 元")
print(f"  收益率: {(env.net_worth - 100000) / 100000 * 100:+.2f}%")
print(f"  交易次数: {env.trade_count}")
print(f"  风险事件: {env.risk_events}")

print(f"\n动作分布:")
action_names = ["持有", "买入25%", "买入50%", "买入100%", 
               "卖出25%", "卖出50%", "卖出100%"]

for i in range(7):
    count = action_dist.get(i, 0)
    pct = count / len(actions) * 100
    print(f"    {i} ({action_names[i]}): {count} 次 ({pct:.1f}%)")

# 检查是否修复
buy_count = sum(action_dist.get(i, 0) for i in [1, 2, 3])

print("\n" + "="*70)
if buy_count > 0 and env.trade_count > 0:
    print("[成功] 模型已经开始交易！")
    print("="*70)
    print("\n修复效果:")
    print(f"  买入动作次数: {buy_count}")
    print(f"  实际交易次数: {env.trade_count}")
    print(f"  收益率: {(env.net_worth - 100000) / 100000 * 100:+.2f}%")
    
    if (env.net_worth - 100000) / 100000 * 100 > 0:
        print("\n[优秀] 模型不仅交易了，还赚钱了！")
    else:
        print("\n[提示] 模型开始交易，但需要更长训练以提高收益")
else:
    print("[警告] 模型仍然不交易")
    print("="*70)
    print("\n可能的解决方案:")
    print("  1. 增加训练步数到 300 万")
    print("  2. 增加探索系数 (ent_coef=0.05)")
    print("  3. 进一步增强买入奖励")

print()



