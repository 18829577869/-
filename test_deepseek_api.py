"""
测试 DeepSeek API Key 是否有效
"""

from llm_market_intelligence import MarketIntelligenceAgent
from datetime import datetime

print("\n" + "="*70)
print(" " * 20 + "DeepSeek API 测试")
print("="*70 + "\n")

# 初始化代理（使用真实 API）
try:
    agent = MarketIntelligenceAgent(
        provider="deepseek",
        api_key="sk-167914945f7945d498e09a7f186c101d",  # 您的 API Key
        enable_cache=True
    )
    
    print("[成功] API 代理初始化成功")
    print(f"  提供商: {agent.provider}")
    print(f"  模式: {'真实API' if not agent.mock_mode else '模拟模式'}")
    print()
    
    # 测试 API 调用
    print("正在测试 API 调用...")
    print("（首次调用可能需要几秒钟）\n")
    
    test_date = datetime.now().strftime("%Y-%m-%d")
    
    intelligence = agent.get_market_intelligence(
        test_date,
        market_context="这是一次测试调用，请分析今日中国A股市场环境。",
        force_refresh=True  # 强制调用 API，不使用缓存
    )
    
    print("="*70)
    print("[成功] API 调用成功！")
    print("="*70 + "\n")
    
    print(f"日期: {intelligence['date']}")
    print(f"数据来源: {intelligence['source']}")
    print(f"时间戳: {intelligence['timestamp']}")
    print()
    
    print("市场情报分析结果:")
    print("-" * 70)
    print(f"  宏观经济评分: {intelligence['macro_economic_score']:+.3f}")
    print(f"  市场情绪评分: {intelligence['market_sentiment_score']:+.3f}")
    print(f"  风险等级: {intelligence['risk_level']:.3f}")
    print(f"  政策影响评分: {intelligence['policy_impact_score']:+.3f}")
    print(f"  突发事件影响: {intelligence['emergency_impact_score']:+.3f}")
    print(f"  资金流向评分: {intelligence['capital_flow_score']:+.3f}")
    print(f"  国际联动系数: {intelligence['international_correlation']:.3f}")
    print(f"  VIX水平: {intelligence['vix_level']:.2f}")
    
    if 'reasoning' in intelligence:
        print(f"\n  分析理由: {intelligence['reasoning']}")
    
    print("-" * 70)
    
    # 测试特征向量
    features = agent.get_feature_vector(intelligence)
    print(f"\n特征向量 (8维): {[f'{x:.3f}' for x in features]}")
    
    print("\n" + "="*70)
    print("[完成] 测试完成！您的 DeepSeek API Key 工作正常。")
    print("="*70 + "\n")
    
    print("下一步:")
    print("  1. 批量生成历史市场情报:")
    print("     python generate_intelligence_with_api.py")
    print()
    print("  2. 使用真实 API 训练模型:")
    print("     python train_v8.py")
    print()
    
except Exception as e:
    print("\n" + "="*70)
    print("[错误] 测试失败")
    print("="*70 + "\n")
    print(f"错误信息: {e}")
    print()
    
    import traceback
    traceback.print_exc()
    
    print("\n可能的原因:")
    print("  1. API Key 无效或已过期")
    print("  2. 账户余额不足")
    print("  3. 网络连接问题")
    print()
    print("解决方案:")
    print("  - 检查 API Key 是否正确")
    print("  - 登录 https://platform.deepseek.com 查看账户状态")
    print("  - 确认网络可以访问 DeepSeek API")
    print()

