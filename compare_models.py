# compare_models.py - 模型版本对比脚本
"""
对比不同版本模型的性能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from stable_baselines3 import PPO
from stock_env_v4_final import StockTradingEnv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def test_model(model_path, data_file, env_class):
    """测试单个模型"""
    try:
        model = PPO.load(model_path)
        env = env_class(data_file)
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        return {
            'success': True,
            'final_net_worth': stats['final_net_worth'],
            'total_return': stats['total_return'],
            'max_drawdown': stats['max_drawdown'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'num_trades': stats['num_trades'],
            'win_rate': stats['win_rate'],
            'net_worth_history': env.net_worth_history
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compare_models():
    """对比所有可用的模型"""
    
    # 定义要对比的模型
    models = [
        {
            'name': 'V3 Final',
            'path': 'ppo_v3_no_explosion.zip',
            'env': 'stock_env_v3_final'
        },
        {
            'name': 'V4',
            'path': 'ppo_stock_model_v4.zip',
            'env': 'stock_env_v4'
        },
        {
            'name': 'V4 Final (优化版)',
            'path': 'ppo_stock_v4_final.zip',
            'env': 'stock_env_v4_final'
        }
    ]
    
    # 测试数据文件
    test_files = [
        'stockdata/test/sh.600036.招商银行.csv',
        'stockdata/test/sh.600000.浦发银行.csv',
        'stockdata/test/159966.SZ.创蓝筹.csv'
    ]
    
    # 过滤存在的文件
    test_files = [f for f in test_files if os.path.exists(f)]
    
    if len(test_files) == 0:
        print("❌ 没有找到测试数据！")
        return
    
    print("="*70)
    print("📊 模型性能对比")
    print("="*70)
    
    results = {model['name']: [] for model in models}
    
    for test_file in test_files:
        stock_name = test_file.split('/')[-1].replace('.csv', '')
        print(f"\n测试股票: {stock_name}")
        print("-"*70)
        
        for model_info in models:
            model_name = model_info['name']
            model_path = model_info['path']
            env_name = model_info['env']
            
            if not os.path.exists(model_path):
                print(f"  ⚠️ {model_name}: 模型文件不存在")
                results[model_name].append(None)
                continue
            
            # 动态导入对应的环境
            try:
                if env_name == 'stock_env_v3_final':
                    from stock_env_v3_final import StockTradingEnv as EnvClass
                elif env_name == 'stock_env_v4':
                    from stock_env_v4 import StockTradingEnv as EnvClass
                else:  # stock_env_v4_final
                    from stock_env_v4_final import StockTradingEnv as EnvClass
                
                result = test_model(model_path, test_file, EnvClass)
                
                if result['success']:
                    results[model_name].append(result)
                    print(f"  ✓ {model_name}: 收益 {result['total_return']:+.2f}% | "
                          f"回撤 {result['max_drawdown']:.2f}% | "
                          f"夏普 {result['sharpe_ratio']:.2f}")
                else:
                    results[model_name].append(None)
                    print(f"  ✗ {model_name}: {result['error']}")
                    
            except Exception as e:
                results[model_name].append(None)
                print(f"  ✗ {model_name}: {e}")
    
    # 汇总统计
    print("\n" + "="*70)
    print("📈 平均性能对比")
    print("="*70)
    
    summary = []
    for model_name, model_results in results.items():
        valid_results = [r for r in model_results if r is not None]
        
        if len(valid_results) > 0:
            avg_return = np.mean([r['total_return'] for r in valid_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in valid_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
            avg_win_rate = np.mean([r['win_rate'] for r in valid_results])
            
            summary.append({
                'model': model_name,
                'avg_return': avg_return,
                'avg_drawdown': avg_drawdown,
                'avg_sharpe': avg_sharpe,
                'avg_win_rate': avg_win_rate,
                'test_count': len(valid_results)
            })
            
            print(f"\n{model_name}:")
            print(f"  平均收益率: {avg_return:+.2f}%")
            print(f"  平均最大回撤: {avg_drawdown:.2f}%")
            print(f"  平均夏普比率: {avg_sharpe:.2f}")
            print(f"  平均胜率: {avg_win_rate:.2f}%")
            print(f"  测试数量: {len(valid_results)}/{len(model_results)}")
    
    # 绘制对比图表
    if len(summary) > 0:
        plot_comparison(summary)
    
    print("\n✅ 对比完成！")

def plot_comparison(summary):
    """绘制对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = [s['model'] for s in summary]
    returns = [s['avg_return'] for s in summary]
    drawdowns = [s['avg_drawdown'] for s in summary]
    sharpes = [s['avg_sharpe'] for s in summary]
    win_rates = [s['avg_win_rate'] for s in summary]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. 平均收益率
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, returns, color=colors[:len(models)])
    ax1.set_title('平均收益率对比', fontsize=12, fontweight='bold')
    ax1.set_ylabel('收益率 (%)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # 2. 平均最大回撤
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, drawdowns, color=colors[:len(models)])
    ax2.set_title('平均最大回撤对比', fontsize=12, fontweight='bold')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, drawdowns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # 3. 夏普比率
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, sharpes, color=colors[:len(models)])
    ax3.set_title('夏普比率对比', fontsize=12, fontweight='bold')
    ax3.set_ylabel('夏普比率')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.axhline(y=1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, sharpes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top')
    
    # 4. 胜率
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, win_rates, color=colors[:len(models)])
    ax4.set_title('平均胜率对比', fontsize=12, fontweight='bold')
    ax4.set_ylabel('胜率 (%)')
    ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, win_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 对比图表已保存: model_comparison.png")

if __name__ == '__main__':
    compare_models()



