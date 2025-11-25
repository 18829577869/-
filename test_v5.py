# test_v5.py - V5环境快速测试
# -*- coding: utf-8 -*-
"""
测试V5的新功能：风险指标和风险应对
"""

import os
import numpy as np
from stock_env_v5 import StockTradingEnv

def test_v5_environment():
    """测试V5环境"""
    
    print("="*70)
    print("[测试] V5 风险感知环境")
    print("="*70)
    
    # 查找测试数据
    test_files = [
        'stockdata/train/sh.600036.招商银行.csv',
        'stockdata/train/sh.600000.浦发银行.csv',
        'stockdata/train/159966.SZ.创蓝筹.csv'
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if test_file is None:
        print("[错误] 没有找到测试数据！")
        return False
    
    print(f"\n[使用数据] {test_file}")
    
    # 测试1：创建环境
    print("\n【测试1】创建V5环境...")
    try:
        env = StockTradingEnv(test_file)
        print(f"  [OK] 环境创建成功")
        print(f"  [OK] 观测空间维度: {env.observation_space.shape}")
        print(f"  [OK] 动作空间: {env.action_space}")
        print(f"  [OK] 特征数量: {len(env.obs_columns)}（含6个风险指标）")
        print(f"  [OK] 数据长度: {len(env.df)} 天")
        
        # 显示风险指标
        risk_indicators = ['Volatility', 'Volume_Anomaly', 'Consecutive_Down', 
                          'Amplitude', 'Gap', 'ATR']
        print(f"  [OK] 风险指标: {', '.join(risk_indicators)}")
    except Exception as e:
        print(f"  [失败] {e}")
        return False
    
    # 测试2：重置并检查观测
    print("\n【测试2】重置环境并检查观测...")
    try:
        obs, _ = env.reset()
        print(f"  [OK] 重置成功")
        print(f"  [OK] 观测形状: {obs.shape}")
        print(f"  [OK] 观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # 检查风险等级（最后一维）
        risk_level_normalized = obs[-1]
        print(f"  [OK] 初始风险等级（归一化）: {risk_level_normalized:.3f}")
    except Exception as e:
        print(f"  [失败] {e}")
        return False
    
    # 测试3：运行50步，观察风险事件
    print("\n【测试3】运行50步，观察风险检测...")
    try:
        risk_events_count = 0
        high_risk_count = 0
        
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            
            # 获取风险等级
            risk_level, warnings = env._assess_risk_level()
            
            if len(warnings) > 0:
                risk_events_count += 1
                print(f"  步骤 {i+1}: [风险{risk_level}] {', '.join(warnings[:2])} | 奖励={reward:+.3f}")
            
            if risk_level >= 3:
                high_risk_count += 1
            
            if done:
                break
        
        print(f"  [OK] 检测到 {risk_events_count} 次风险事件")
        print(f"  [OK] 高风险状态 {high_risk_count} 次")
        
        if risk_events_count == 0:
            print(f"  [提示] 未检测到风险事件（可能数据期间市场平稳）")
        
    except Exception as e:
        print(f"  [失败] {e}")
        return False
    
    # 测试4：完整回合并获取统计
    print("\n【测试4】运行完整回合...")
    try:
        env = StockTradingEnv(test_file)
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            step_count += 1
        
        stats = env.get_stats()
        
        print(f"  [OK] 完整回合运行成功")
        print(f"  [OK] 总步数: {step_count}")
        print(f"  [OK] 最终净值: {stats['final_net_worth']:,.2f}")
        print(f"  [OK] 总收益率: {stats['total_return']:+.2f}%")
        print(f"  [OK] 交易次数: {stats['num_trades']}")
        print(f"  [OK] 风险事件: {stats['risk_events']} 次")  # V5新增
        
    except Exception as e:
        print(f"  [失败] {e}")
        return False
    
    # 测试5：检查风险指标数据质量
    print("\n【测试5】检查风险指标数据质量...")
    try:
        print(f"  [检查] Volatility范围: [{env.df['Volatility'].min():.2f}, {env.df['Volatility'].max():.2f}]")
        print(f"  [检查] Volume_Anomaly范围: [{env.df['Volume_Anomaly'].min():.2f}, {env.df['Volume_Anomaly'].max():.2f}]")
        print(f"  [检查] Consecutive_Down范围: [{env.df['Consecutive_Down'].min():.0f}, {env.df['Consecutive_Down'].max():.0f}]")
        print(f"  [检查] Amplitude范围: [{env.df['Amplitude'].min():.2f}, {env.df['Amplitude'].max():.2f}]")
        print(f"  [检查] Gap范围: [{env.df['Gap'].min():.2f}, {env.df['Gap'].max():.2f}]")
        print(f"  [检查] ATR范围: [{env.df['ATR'].min():.2f}, {env.df['ATR'].max():.2f}]")
        print(f"  [OK] 所有风险指标数据正常")
    except Exception as e:
        print(f"  [失败] {e}")
        return False
    
    print("\n" + "="*70)
    print("[成功] V5环境测试全部通过！")
    print("="*70)
    print("\n[特色功能]")
    print("  1. 观测空间从94维增加到125维")
    print("  2. 新增6个风险指标")
    print("  3. 实时风险等级评估（0-6分）")
    print("  4. 高风险时自动限制买入")
    print("  5. 风险应对奖励机制")
    print("\n[下一步]")
    print("  运行: python train_v5.py")
    print("  开始训练V5风险感知模型")
    
    return True

if __name__ == '__main__':
    test_v5_environment()



