# test_v4_final.py - 快速测试V4 Final环境
# -*- coding: utf-8 -*-
"""
快速测试脚本，验证新环境是否正常工作
"""

import os
import sys
import numpy as np

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from stock_env_v4_final import StockTradingEnv

def test_environment():
    """测试环境基本功能"""
    
    print("="*70)
    print("[测试] V4 Final 环境")
    print("="*70)
    
    # 查找可用的测试数据
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
        print("[错误] 没有找到测试数据文件！")
        print("请先运行 get_stock_data_v3.py 下载数据")
        return False
    
    print(f"\n[成功] 使用数据文件: {test_file}")
    
    # 测试1: 创建环境
    print("\n【测试1】创建环境...")
    try:
        env = StockTradingEnv(test_file)
        print(f"  [OK] 环境创建成功")
        print(f"  [OK] 观测空间维度: {env.observation_space.shape}")
        print(f"  [OK] 动作空间: {env.action_space}")
        print(f"  [OK] 数据长度: {len(env.df)} 天")
    except Exception as e:
        print(f"  [失败] 环境创建失败: {e}")
        return False
    
    # 测试2: 重置环境
    print("\n【测试2】重置环境...")
    try:
        obs, info = env.reset()
        print(f"  [OK] 重置成功")
        print(f"  [OK] 观测形状: {obs.shape}")
        print(f"  [OK] 观测范围: [{obs.min():.2f}, {obs.max():.2f}]")
    except Exception as e:
        print(f"  [失败] 重置失败: {e}")
        return False
    
    # 测试3: 执行随机动作
    print("\n【测试3】执行随机动作（10步）...")
    try:
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            action_type = ["卖出", "持有", "买入"][int(np.round(action[0]))]
            print(f"  步骤 {i+1}: 动作={action_type}({action[1]:.2f}) | "
                  f"奖励={reward:+.3f} | 净值={env.net_worth:.0f}")
            
            if done:
                print(f"  [信息] 第 {i+1} 步环境结束")
                break
        
        print(f"  [OK] 动作执行正常")
    except Exception as e:
        print(f"  [失败] 动作执行失败: {e}")
        return False
    
    # 测试4: 获取统计信息
    print("\n【测试4】获取统计信息...")
    try:
        stats = env.get_stats()
        print(f"  [OK] 统计信息获取成功")
        print(f"  [OK] 最终净值: {stats['final_net_worth']:.2f}")
        print(f"  [OK] 总收益率: {stats['total_return']:+.2f}%")
        print(f"  [OK] 最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"  [OK] 夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"  [OK] 交易次数: {stats['num_trades']}")
        print(f"  [OK] 胜率: {stats['win_rate']:.2f}%")
    except Exception as e:
        print(f"  [失败] 统计信息获取失败: {e}")
        return False
    
    # 测试5: 完整回合
    print("\n【测试5】运行完整回合（随机策略）...")
    try:
        env = StockTradingEnv(test_file)
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
        
        stats = env.get_stats()
        print(f"  [OK] 完整回合运行成功")
        print(f"  [OK] 总步数: {step_count}")
        print(f"  [OK] 最终净值: {stats['final_net_worth']:.2f}")
        print(f"  [OK] 总收益率: {stats['total_return']:+.2f}%")
        
    except Exception as e:
        print(f"  [失败] 完整回合失败: {e}")
        return False
    
    print("\n" + "="*70)
    print("[成功] 所有测试通过！环境工作正常！")
    print("="*70)
    print("\n下一步：")
    print("  1. 运行 python train_v4_final.py 开始训练")
    print("  2. 训练完成后运行 python evaluate_v4_final.py 查看详细评估")
    print("  3. 运行 python compare_models.py 对比不同版本")
    
    return True

def test_observation_components():
    """详细测试观测空间的各个组成部分"""
    print("\n" + "="*70)
    print("[检查] 详细检查观测空间")
    print("="*70)
    
    test_file = None
    for f in ['stockdata/train/sh.600036.招商银行.csv',
              'stockdata/train/sh.600000.浦发银行.csv']:
        if os.path.exists(f):
            test_file = f
            break
    
    if test_file is None:
        print("没有找到测试数据")
        return
    
    env = StockTradingEnv(test_file)
    obs, _ = env.reset()
    
    # 解析观测空间
    history_window = env.history_window
    num_features = len(env.obs_columns)
    history_size = history_window * num_features
    
    print(f"\n观测空间总维度: {len(obs)}")
    print(f"  - 历史窗口: {history_window} 天 x {num_features} 特征 = {history_size} 维")
    print(f"  - 持仓信息: 4 维")
    print(f"  - 总计: {history_size + 4} 维")
    
    print(f"\n特征列表 ({num_features}个):")
    for i, col in enumerate(env.obs_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 提取历史数据和持仓信息
    historical_obs = obs[:history_size]
    position_info = obs[history_size:]
    
    print(f"\n当前观测数值范围:")
    print(f"  - 历史数据: [{historical_obs.min():.3f}, {historical_obs.max():.3f}]")
    print(f"  - 持仓信息: {position_info}")
    print(f"    [持仓比例={position_info[0]:.3f}, 现金比例={position_info[1]:.3f}, "
          f"收益率={position_info[2]:.3f}, 回撤={position_info[3]:.3f}]")
    
    print("\n[成功] 观测空间检查完成！")

if __name__ == '__main__':
    # 基础功能测试
    success = test_environment()
    
    # 详细观测空间检查
    if success:
        test_observation_components()
