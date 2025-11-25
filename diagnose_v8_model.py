"""
诊断 V8 模型的行为
检查为什么模型不交易
"""

from stable_baselines3 import PPO
from stock_env_v8 import StockTradingEnvV8
import numpy as np

print("="*70)
print(" " * 20 + "V8 模型诊断")
print("="*70 + "\n")

# 加载模型
model = PPO.load("ppo_stock_v8.zip")
print("[加载] 模型: ppo_stock_v8.zip\n")

# 创建环境
env = StockTradingEnvV8(
    data_file="stockdata_v7/test/sh.600036.招商银行.csv",
    llm_provider="deepseek"
)

# 重置环境
obs, _ = env.reset()

# 统计动作分布
actions = []
rewards = []

print("测试 100 步，观察模型行为...")
print("-" * 70)

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    actions.append(action)
    rewards.append(reward)
    
    if i < 10:  # 只打印前 10 步
        print(f"步骤 {i+1}: 动作={action} ({info['action']}), "
              f"奖励={reward:+.3f}, 净值={info['net_worth']:,.0f}, "
              f"市场风险={info['market_risk']:.2f}")
    
    if done or truncated:
        break

print("\n" + "-" * 70)
print("\n动作统计:")
print(f"  总步数: {len(actions)}")
print(f"  动作分布:")

for action_id in range(7):
    count = sum(1 for a in actions if a == action_id)
    pct = count / len(actions) * 100
    action_name = ["持有", "买入25%", "买入50%", "买入100%", 
                   "卖出25%", "卖出50%", "卖出100%"][action_id]
    print(f"    {action_id} ({action_name}): {count} 次 ({pct:.1f}%)")

print(f"\n  平均奖励: {np.mean(rewards):.3f}")
print(f"  奖励范围: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")

# 分析问题
print("\n" + "="*70)
print("问题分析:")
print("="*70)

hold_pct = sum(1 for a in actions if a == 0) / len(actions) * 100

if hold_pct > 95:
    print("\n[严重] 模型几乎只选择持有动作！")
    print("  可能原因:")
    print("    1. 空仓惩罚不够强")
    print("    2. 交易成本过高")
    print("    3. 奖励函数设计问题")
    print("    4. 训练不充分")
elif hold_pct > 80:
    print("\n[警告] 模型过度偏好持有动作")
    print("  需要增强交易激励")
elif hold_pct < 20:
    print("\n[警告] 模型过度交易")
    print("  可能需要增加交易成本惩罚")
else:
    print("\n[正常] 模型动作分布合理")

print()



