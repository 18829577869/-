"""
使用真实 DeepSeek API 批量生成市场情报
注意：这会消耗 API 调用次数，预计成本约 2 元
"""

import sys
from datetime import datetime
from llm_market_intelligence import MarketIntelligenceAgent


def main(auto_confirm=False):
    print("\n" + "="*70)
    print(" " * 10 + "使用真实 DeepSeek API 批量生成市场情报")
    print("="*70 + "\n")
    
    # 配置
    start_date = "2020-01-01"  # 开始日期
    end_date = datetime.now().strftime("%Y-%m-%d")  # 截止今天
    api_key = "sk-167914945f7945d498e09a7f186c101d"  # 您的 API Key
    
    print("配置信息:")
    print(f"  起始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  LLM 提供商: DeepSeek")
    print(f"  数据模式: 真实 API（收费）")
    print(f"  API Key: {api_key[:20]}...")
    print()
    
    # 成本估算
    from datetime import datetime as dt
    from pandas import date_range
    date_count = len(date_range(start=start_date, end=end_date, freq='D'))
    estimated_cost = date_count * 0.001
    
    print("成本估算:")
    print(f"  预计调用次数: {date_count} 次")
    print(f"  预计成本: 约 {estimated_cost:.2f} 元")
    print()
    
    # 确认
    print("[警告] 这将使用真实 API 调用，会产生费用")
    if auto_confirm:
        print("是否继续？(y/N): y (自动确认)")
        confirm = 'y'
    else:
        confirm = input("是否继续？(y/N): ")
    
    if confirm.lower() != 'y':
        print("\n已取消\n")
        return 0
    
    # 初始化代理（使用真实 API）
    agent = MarketIntelligenceAgent(
        provider="deepseek",
        api_key=api_key,
        enable_cache=True
    )
    
    # 批量生成（使用真实 API）
    try:
        print("\n开始生成...")
        print("（由于 API 调用限制，可能需要较长时间）\n")
        
        agent.batch_generate_intelligence(
            start_date=start_date,
            end_date=end_date,
            use_mock=False  # 使用真实 API
        )
        
        print("\n" + "="*70)
        print("[完成] 市场情报生成完成！")
        print("="*70)
        print(f"\n缓存目录: {agent.cache_dir}")
        print(f"实际成本: 约 {estimated_cost:.2f} 元")
        print("\n下一步:")
        print("  python train_v8.py\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[中断] 用户中止生成")
        print("部分数据已缓存，下次运行会跳过已缓存的日期\n")
        return 1
    
    except Exception as e:
        print(f"\n[错误] {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 支持 --yes 参数自动确认
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    sys.exit(main(auto_confirm=auto_confirm))



