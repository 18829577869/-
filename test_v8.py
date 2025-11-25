"""
V8 环境测试脚本
验证 LLM 集成和环境配置
"""

import os
import sys
from stock_env_v8 import StockTradingEnvV8
from llm_market_intelligence import MarketIntelligenceAgent
import numpy as np


def test_llm_agent():
    """测试 LLM 市场情报代理"""
    print("="*70)
    print("测试 1: LLM 市场情报代理")
    print("="*70 + "\n")
    
    try:
        # 初始化代理
        agent = MarketIntelligenceAgent(
            provider="deepseek",
            enable_cache=True
        )
        
        print("✓ LLM 代理初始化成功")
        print(f"  提供商: {agent.provider}")
        print(f"  缓存目录: {agent.cache_dir}")
        print(f"  模式: {'模拟数据' if agent.mock_mode else '真实API'}\n")
        
        # 测试获取单日情报
        test_date = "2024-12-01"
        print(f"获取 {test_date} 的市场情报...")
        intelligence = agent.get_market_intelligence(test_date)
        
        print("\n市场情报详情:")
        print(f"  宏观经济评分: {intelligence['macro_economic_score']:+.3f}")
        print(f"  市场情绪评分: {intelligence['market_sentiment_score']:+.3f}")
        print(f"  风险等级: {intelligence['risk_level']:.3f}")
        print(f"  政策影响评分: {intelligence['policy_impact_score']:+.3f}")
        print(f"  突发事件影响: {intelligence['emergency_impact_score']:+.3f}")
        print(f"  资金流向评分: {intelligence['capital_flow_score']:+.3f}")
        print(f"  国际联动系数: {intelligence['international_correlation']:.3f}")
        print(f"  VIX水平: {intelligence['vix_level']:.2f}")
        print(f"  数据来源: {intelligence['source']}")
        
        # 测试特征向量
        features = agent.get_feature_vector(intelligence)
        print(f"\n特征向量 (8维): {[f'{x:.3f}' for x in features]}")
        
        print("\n✓ 测试 1 通过!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 1 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """测试 V8 交易环境"""
    print("="*70)
    print("测试 2: V8 交易环境")
    print("="*70 + "\n")
    
    # 查找测试数据
    test_files = [
        "stockdata_v7/train/sh.600036.招商银行.csv",
        "stockdata_v7/train/sh.601838.成都银行.csv",
        "stockdata_v7/train/sz.000858.五粮液.csv",
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if not test_file:
        print("✗ 未找到测试数据文件")
        print("  请先运行: python get_etf_data_akshare.py")
        return False
    
    print(f"使用数据: {test_file}\n")
    
    try:
        # 初始化环境
        print("初始化环境...")
        env = StockTradingEnvV8(
            data_file=test_file,
            initial_balance=100000,
            llm_provider="deepseek",
            enable_llm_cache=True,
            llm_weight=0.3
        )
        
        print("✓ 环境初始化成功")
        print(f"  数据长度: {len(env.df)} 天")
        print(f"  观察空间: {env.observation_space.shape}")
        print(f"  动作空间: {env.action_space.n} 个离散动作\n")
        
        # 测试 reset
        print("测试 reset()...")
        obs, info = env.reset(seed=42)
        
        print(f"✓ reset() 成功")
        print(f"  观察维度: {obs.shape}")
        print(f"  技术指标 (前21维): {obs[:21][:5]}... (显示前5)")
        print(f"  LLM情报 (后8维): {obs[21:]}\n")
        
        # 测试几步交易
        print("测试交易步骤...")
        print("-" * 70)
        
        for i in range(5):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"\n步骤 {i+1}:")
            print(f"  日期: {info['date']}")
            print(f"  动作: {info['action']}")
            print(f"  净值: {info['net_worth']:,.0f} 元")
            print(f"  奖励: {reward:+.3f}")
            print(f"  市场风险: {info['market_risk']:.2f}")
            print(f"  市场情绪: {info['market_sentiment']:+.2f}")
            print(f"  是否交易: {info['trade_executed']}")
            
            if done or truncated:
                print("\n  回合结束")
                break
        
        print("\n" + "-" * 70)
        print("\n✓ 测试 2 通过!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 2 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space():
    """测试观察空间维度和内容"""
    print("="*70)
    print("测试 3: 观察空间完整性")
    print("="*70 + "\n")
    
    test_file = "stockdata_v7/train/sh.600036.招商银行.csv"
    if not os.path.exists(test_file):
        print("✗ 测试数据不存在，跳过此测试\n")
        return False
    
    try:
        env = StockTradingEnvV8(
            data_file=test_file,
            llm_provider="deepseek"
        )
        
        obs, _ = env.reset()
        
        print(f"观察空间维度检查:")
        print(f"  期望: 29 维 (21技术 + 8LLM)")
        print(f"  实际: {len(obs)} 维")
        
        if len(obs) != 29:
            print(f"\n✗ 维度不匹配!\n")
            return False
        
        print("\n各维度值范围检查:")
        print(f"  技术指标 (0-20维):")
        print(f"    最小值: {np.min(obs[:21]):.3f}")
        print(f"    最大值: {np.max(obs[:21]):.3f}")
        print(f"    平均值: {np.mean(obs[:21]):.3f}")
        
        print(f"\n  LLM情报 (21-28维):")
        print(f"    最小值: {np.min(obs[21:]):.3f}")
        print(f"    最大值: {np.max(obs[21:]):.3f}")
        print(f"    平均值: {np.mean(obs[21:]):.3f}")
        
        # 检查是否有 NaN 或 Inf
        if np.any(np.isnan(obs)):
            print(f"\n✗ 观察包含 NaN 值!")
            return False
        
        if np.any(np.isinf(obs)):
            print(f"\n✗ 观察包含 Inf 值!")
            return False
        
        print("\n✓ 测试 3 通过!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 3 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cache_persistence():
    """测试缓存持久化"""
    print("="*70)
    print("测试 4: 缓存持久化")
    print("="*70 + "\n")
    
    try:
        agent = MarketIntelligenceAgent(
            provider="deepseek",
            enable_cache=True
        )
        
        test_date = "2024-11-20"
        
        # 首次获取（写入缓存）
        print(f"首次获取 {test_date} 情报...")
        intel1 = agent.get_market_intelligence(test_date)
        
        # 检查缓存文件是否存在
        cache_file = agent._get_cache_path(test_date)
        if not os.path.exists(cache_file):
            print(f"✗ 缓存文件未创建: {cache_file}")
            return False
        
        print(f"✓ 缓存文件已创建: {cache_file}")
        
        # 第二次获取（从缓存读取）
        print(f"\n再次获取 {test_date} 情报（应从缓存读取）...")
        intel2 = agent.get_market_intelligence(test_date)
        
        # 验证两次结果一致
        if intel1 == intel2:
            print("✓ 缓存数据一致")
        else:
            print("✗ 缓存数据不一致!")
            return False
        
        print("\n✓ 测试 4 通过!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 4 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_reward_function():
    """测试奖励函数"""
    print("="*70)
    print("测试 5: 奖励函数")
    print("="*70 + "\n")
    
    test_file = "stockdata_v7/train/sh.600036.招商银行.csv"
    if not os.path.exists(test_file):
        print("✗ 测试数据不存在，跳过此测试\n")
        return False
    
    try:
        env = StockTradingEnvV8(
            data_file=test_file,
            llm_provider="deepseek",
            llm_weight=0.3
        )
        
        obs, _ = env.reset()
        
        print("测试不同动作的奖励:")
        
        # 测试持有
        obs, reward_hold, _, _, info = env.step(0)
        print(f"\n  持有 (action=0):")
        print(f"    奖励: {reward_hold:+.3f}")
        
        # 重置
        env.reset()
        
        # 测试买入
        obs, reward_buy, _, _, info = env.step(3)  # 买入 100%
        print(f"\n  买入100% (action=3):")
        print(f"    奖励: {reward_buy:+.3f}")
        print(f"    是否交易: {info['trade_executed']}")
        
        # 测试卖出
        obs, reward_sell, _, _, info = env.step(6)  # 卖出 100%
        print(f"\n  卖出100% (action=6):")
        print(f"    奖励: {reward_sell:+.3f}")
        print(f"    是否交易: {info['trade_executed']}")
        
        print("\n✓ 奖励函数正常工作")
        print("\n✓ 测试 5 通过!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 5 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" " * 20 + "V8 环境测试套件")
    print("="*70 + "\n")
    
    results = []
    
    # 测试 1: LLM 代理
    results.append(("LLM 市场情报代理", test_llm_agent()))
    
    # 测试 2: 交易环境
    results.append(("V8 交易环境", test_environment()))
    
    # 测试 3: 观察空间
    results.append(("观察空间完整性", test_observation_space()))
    
    # 测试 4: 缓存持久化
    results.append(("缓存持久化", test_cache_persistence()))
    
    # 测试 5: 奖励函数
    results.append(("奖励函数", test_reward_function()))
    
    # 汇总结果
    print("="*70)
    print(" " * 25 + "测试结果汇总")
    print("="*70 + "\n")
    
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name:<30} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "="*70)
    print(f"  总计: {passed}/{total} 测试通过")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确，可以开始训练。\n")
        print("下一步:")
        print("  1. 批量生成市场情报缓存:")
        print("     python generate_intelligence.py")
        print("\n  2. 开始训练:")
        print("     python train_v8.py\n")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())



