"""
批量生成市场情报缓存
用于 V8 训练前的数据准备
"""

import sys
from datetime import datetime
from llm_market_intelligence import MarketIntelligenceAgent


def main():
    print("\n" + "="*70)
    print(" " * 15 + "批量生成市场情报缓存")
    print("="*70 + "\n")
    
    # 配置
    start_date = "2020-01-01"  # 开始日期
    end_date = datetime.now().strftime("%Y-%m-%d")  # 截止今天
    provider = "deepseek"  # LLM 提供商
    use_mock = True  # 使用模拟数据（推荐，免费）
    
    print("配置信息:")
    print(f"  起始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  LLM 提供商: {provider}")
    print(f"  数据模式: {'模拟数据（免费）' if use_mock else '真实 API（收费）'}")
    print()
    
    # 确认
    if not use_mock:
        print("⚠️  警告: 您选择了真实 API 模式")
        print("   预计调用次数: ~2000 次")
        print("   预计成本: ~2 元（DeepSeek）")
        print()
        confirm = input("   是否继续？(y/N): ")
        if confirm.lower() != 'y':
            print("\n已取消\n")
            return 0
    
    # 初始化代理
    agent = MarketIntelligenceAgent(
        provider=provider,
        enable_cache=True
    )
    
    # 批量生成
    try:
        agent.batch_generate_intelligence(
            start_date=start_date,
            end_date=end_date,
            use_mock=use_mock
        )
        
        print("\n" + "="*70)
        print("✓ 完成！")
        print("="*70)
        print(f"\n缓存目录: {agent.cache_dir}")
        print("\n下一步:")
        print("  python train_v8.py\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[中断] 用户中止生成")
        print("部分数据已缓存，下次运行会跳过已缓存的日期\n")
        return 1
    
    except Exception as e:
        print(f"\n✗ 错误: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



