"""
持仓状态更新工具

用于手动更新持仓状态，适用于在系统外进行交易后同步持仓信息。

使用方法:
    python update_portfolio.py --stock sh.600036 --shares 500 --balance 80000 --price 43.25

或交互式更新:
    python update_portfolio.py
"""

import os
import sys
import json
import argparse
import datetime

# 持仓状态文件
PORTFOLIO_STATE_FILE = "portfolio_state.json"

def load_current_state():
    """加载当前持仓状态"""
    if os.path.exists(PORTFOLIO_STATE_FILE):
        try:
            with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  读取持仓状态失败: {e}")
            return None
    return None

def save_portfolio_state(stock_code, shares_held, current_balance, last_price=None, initial_balance=None):
    """保存持仓状态"""
    try:
        # 如果未提供价格，尝试从当前状态获取
        current_state = load_current_state()
        if last_price is None and current_state:
            last_price = current_state.get('last_price', 0.0)
        
        if initial_balance is None and current_state:
            initial_balance = current_state.get('initial_balance', current_balance + (shares_held * last_price if last_price > 0 else 0))
        elif initial_balance is None:
            # 如果没有历史记录，使用当前总资产作为初始资金
            total_assets = current_balance + (shares_held * last_price if last_price > 0 else 0)
            initial_balance = total_assets
        
        state = {
            'stock_code': stock_code,
            'shares_held': float(shares_held),
            'current_balance': float(current_balance),
            'last_price': float(last_price) if last_price else 0.0,
            'initial_balance': float(initial_balance),
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_assets': float(current_balance + shares_held * last_price) if last_price > 0 else float(current_balance)
        }
        
        with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"❌ 保存持仓状态失败: {e}")
        return False

def show_current_state():
    """显示当前持仓状态"""
    state = load_current_state()
    if state:
        print("=" * 70)
        print("📋 当前持仓状态")
        print("=" * 70)
        print(f"股票代码: {state.get('stock_code', '未知')}")
        print(f"持仓数量: {state.get('shares_held', 0):.2f} 股")
        print(f"可用资金: {state.get('current_balance', 0):.2f} 元")
        if state.get('last_price', 0) > 0:
            position_value = state.get('shares_held', 0) * state.get('last_price', 0)
            total_assets = state.get('current_balance', 0) + position_value
            print(f"持仓市值: {position_value:.2f} 元")
            print(f"总资产: {total_assets:.2f} 元")
            print(f"上次价格: {state.get('last_price', 0):.2f} 元")
        print(f"初始资金: {state.get('initial_balance', 0):.2f} 元")
        print(f"上次更新: {state.get('last_update', '未知')}")
        print("=" * 70)
    else:
        print("ℹ️  当前没有持仓状态记录")

def interactive_update():
    """交互式更新持仓状态"""
    print("=" * 70)
    print("📝 持仓状态更新工具（交互式）")
    print("=" * 70)
    print()
    
    # 显示当前状态
    show_current_state()
    print()
    
    # 获取当前状态作为默认值
    current_state = load_current_state()
    
    # 输入股票代码
    default_stock = current_state.get('stock_code', 'sh.600036') if current_state else 'sh.600036'
    stock_code = input(f"股票代码 [{default_stock}]: ").strip()
    if not stock_code:
        stock_code = default_stock
    
    # 输入持仓数量
    default_shares = current_state.get('shares_held', 0.0) if current_state else 0.0
    shares_input = input(f"持仓数量（股） [{default_shares:.2f}]: ").strip()
    try:
        shares_held = float(shares_input) if shares_input else default_shares
    except ValueError:
        print("⚠️  输入无效，使用默认值")
        shares_held = default_shares
    
    # 输入可用资金
    default_balance = current_state.get('current_balance', 100000.0) if current_state else 100000.0
    balance_input = input(f"可用资金（元） [{default_balance:.2f}]: ").strip()
    try:
        current_balance = float(balance_input) if balance_input else default_balance
    except ValueError:
        print("⚠️  输入无效，使用默认值")
        current_balance = default_balance
    
    # 输入当前价格（可选）
    default_price = current_state.get('last_price', 0.0) if current_state else 0.0
    price_input = input(f"当前价格（元，可选） [{default_price:.2f}]: ").strip()
    try:
        last_price = float(price_input) if price_input else default_price
    except ValueError:
        last_price = default_price
    
    # 确认更新
    print()
    print("=" * 70)
    print("📋 更新信息确认")
    print("=" * 70)
    print(f"股票代码: {stock_code}")
    print(f"持仓数量: {shares_held:.2f} 股")
    print(f"可用资金: {current_balance:.2f} 元")
    if last_price > 0:
        position_value = shares_held * last_price
        total_assets = current_balance + position_value
        print(f"当前价格: {last_price:.2f} 元")
        print(f"持仓市值: {position_value:.2f} 元")
        print(f"总资产: {total_assets:.2f} 元")
    print("=" * 70)
    
    confirm = input("\n确认更新？(y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("❌ 已取消更新")
        return False
    
    # 保存状态
    if save_portfolio_state(stock_code, shares_held, current_balance, last_price):
        print("✅ 持仓状态已更新！")
        print(f"   文件位置: {PORTFOLIO_STATE_FILE}")
        return True
    else:
        print("❌ 更新失败")
        return False

def main():
    parser = argparse.ArgumentParser(description='更新持仓状态')
    parser.add_argument('--stock', type=str, help='股票代码（如：sh.600036）')
    parser.add_argument('--shares', type=float, help='持仓数量（股）')
    parser.add_argument('--balance', type=float, help='可用资金（元）')
    parser.add_argument('--price', type=float, help='当前价格（元，可选）')
    parser.add_argument('--show', action='store_true', help='显示当前持仓状态')
    
    args = parser.parse_args()
    
    # 显示当前状态
    if args.show:
        show_current_state()
        return
    
    # 命令行参数更新
    if args.stock and args.shares is not None and args.balance is not None:
        if save_portfolio_state(args.stock, args.shares, args.balance, args.price):
            print("✅ 持仓状态已更新！")
            print(f"   文件位置: {PORTFOLIO_STATE_FILE}")
            show_current_state()
        else:
            print("❌ 更新失败")
    else:
        # 交互式更新
        interactive_update()

if __name__ == "__main__":
    main()

