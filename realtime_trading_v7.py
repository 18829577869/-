"""
V7实盘交易脚本
用于生成指定日期的交易信号
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv
import warnings
warnings.filterwarnings('ignore')

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# V7使用的股票列表（你的6只股票）
PORTFOLIO_STOCKS = [
    {"code": "sh.600036", "name": "招商银行"},
    {"code": "sh.601838", "name": "成都银行"},
    {"code": "sh.601318", "name": "中国平安"},
    {"code": "sh.601939", "name": "建设银行"},
    {"code": "sh.601398", "name": "工商银行"},
    {"code": "sz.000858", "name": "五粮液"},
]

MODEL_PATH = "ppo_stock_v7.zip"  # V7模型路径
INITIAL_CAPITAL = 100000  # 初始资金10万


class RealtimeTrader:
    """实盘交易决策器"""
    
    def __init__(self, model_path, initial_capital=100000):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.model = None
        
        # 加载模型
        if os.path.exists(model_path):
            print(f"[加载] V7模型: {model_path}")
            self.model = PPO.load(model_path)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def get_latest_data_file(self, stock_code, stock_name):
        """获取最新数据文件路径"""
        # 尝试多个可能的路径（优先使用实时数据）
        possible_paths = [
            f"stockdata_v7_realtime/{stock_code}.{stock_name}.csv",  # 实时数据（优先）
            f"stockdata_v7/train/{stock_code}.{stock_name}.csv",
            f"stockdata_v7/test/{stock_code}.{stock_name}.csv",
            f"stockdata/{stock_code}.{stock_name}.csv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def get_action_description(self, action):
        """将动作转换为可读描述"""
        actions = {
            0: "持有",
            1: "买入25%可用资金",
            2: "买入50%可用资金",
            3: "买入100%可用资金",
            4: "卖出25%持仓",
            5: "卖出50%持仓",
            6: "卖出100%持仓（清仓）"
        }
        return actions.get(action, "未知动作")
    
    def get_risk_level_description(self, risk_score):
        """风险等级描述"""
        if risk_score >= 4:
            return "🔴 高风险"
        elif risk_score >= 2:
            return "🟡 中等风险"
        else:
            return "🟢 低风险"
    
    def predict_trade(self, stock_code, stock_name, target_date=None):
        """
        预测单只股票的交易信号
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            target_date: 目标日期（格式：YYYY-MM-DD），None表示最新日期
        """
        # 获取数据文件
        data_file = self.get_latest_data_file(stock_code, stock_name)
        if not data_file:
            return {
                'status': 'error',
                'message': f"找不到 {stock_name}({stock_code}) 的数据文件"
            }
        
        # 读取数据
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 检查目标日期
        if target_date:
            target_dt = pd.to_datetime(target_date)
            # 找到目标日期或之前最近的日期
            valid_dates = df[df['date'] <= target_dt]
            if len(valid_dates) == 0:
                return {
                    'status': 'error',
                    'message': f"数据不包含 {target_date} 或之前的日期"
                }
            latest_date = valid_dates['date'].max()
            
            # 提示：如果使用的不是目标日期
            if latest_date < target_dt:
                actual_date_str = latest_date.strftime('%Y-%m-%d')
                print(f"   ⚠️ 数据最新日期为 {actual_date_str}（请求日期：{target_date}）")
        else:
            latest_date = df['date'].max()
        
        # 创建环境（使用所有历史数据）
        env = StockTradingEnv(
            data_file=data_file,
            initial_balance=self.initial_capital
        )
        
        # 运行到最新日期
        obs, _ = env.reset()
        done = False
        target_found = False
        
        while not done:
            # 检查当前步骤是否有效
            if env.current_step >= len(env.df):
                break
                
            current_date = env.df.loc[env.current_step, 'date']
            
            # 如果到达或超过目标日期，记录状态并预测
            if current_date >= latest_date and not target_found:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 获取当前市场状态
                current_price = env.df.loc[env.current_step, 'close']
                current_change = env.df.loc[env.current_step, 'pctChg']
                
                # 计算持仓信息
                shares_held = env.shares_held
                position_value = shares_held * current_price
                cash_balance = env.balance
                total_value = position_value + cash_balance
                position_pct = (position_value / total_value * 100) if total_value > 0 else 0
                
                # 风险评估
                risk_score = env._calculate_risk_level(env.current_step)
                
                return {
                    'status': 'success',
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'action': int(action),
                    'action_desc': self.get_action_description(int(action)),
                    'current_price': float(current_price),
                    'price_change': float(current_change),
                    'position_pct': float(position_pct),
                    'shares_held': int(shares_held),
                    'cash_balance': float(cash_balance),
                    'total_value': float(total_value),
                    'risk_score': risk_score,
                    'risk_level': self.get_risk_level_description(risk_score)
                }
            
            # 继续执行到下一步
            if env.current_step < len(env.df) - 1:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                # 已到数据末尾
                break
        
        return {
            'status': 'error',
            'message': f"未能到达目标日期 {latest_date}，数据可能不足"
        }
    
    def generate_portfolio_report(self, target_date=None):
        """
        生成投资组合交易报告
        
        Args:
            target_date: 目标日期（格式：YYYY-MM-DD）
        """
        print("\n" + "="*80)
        print(f"V7 实盘交易信号报告")
        print("="*80)
        
        if target_date:
            print(f"\n目标日期: {target_date}")
        else:
            print(f"\n目标日期: 最新数据")
        
        print(f"初始资金: {self.initial_capital:,.0f} 元")
        print(f"股票数量: {len(PORTFOLIO_STOCKS)} 只")
        print(f"单只股票配置: {self.initial_capital/len(PORTFOLIO_STOCKS):,.0f} 元")
        
        print("\n" + "-"*80)
        print("个股交易信号")
        print("-"*80)
        
        results = []
        for stock in PORTFOLIO_STOCKS:
            result = self.predict_trade(
                stock['code'], 
                stock['name'], 
                target_date
            )
            results.append(result)
            
            if result['status'] == 'success':
                print(f"\n📊 {result['stock_name']} ({result['stock_code']})")
                print(f"   日期: {result['date']}")
                print(f"   当前价格: ¥{result['current_price']:.2f}")
                print(f"   涨跌幅: {result['price_change']:+.2f}%")
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   💼 持仓情况:")
                print(f"      持仓比例: {result['position_pct']:.1f}%")
                print(f"      持股数量: {result['shares_held']:,} 股")
                print(f"      持股市值: ¥{result['shares_held'] * result['current_price']:,.0f}")
                print(f"      现金余额: ¥{result['cash_balance']:,.0f}")
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   {result['risk_level']} (风险评分: {result['risk_score']}/6)")
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   ⚡ 交易建议: {result['action_desc']}")
                
                # 给出具体操作建议
                if result['action'] in [1, 2, 3]:
                    buy_ratios = {1: 0.25, 2: 0.5, 3: 1.0}
                    buy_amount = result['cash_balance'] * buy_ratios[result['action']]
                    shares = int(buy_amount / result['current_price'] / 100) * 100
                    cost = shares * result['current_price']
                    if shares > 0:
                        print(f"      → 买入约 {shares:,} 股")
                        print(f"      → 预计花费 ¥{cost:,.0f}")
                
                elif result['action'] in [4, 5, 6]:
                    sell_ratios = {4: 0.25, 5: 0.5, 6: 1.0}
                    shares = int(result['shares_held'] * sell_ratios[result['action']] / 100) * 100
                    revenue = shares * result['current_price']
                    if shares > 0:
                        print(f"      → 卖出约 {shares:,} 股")
                        print(f"      → 预计收入 ¥{revenue:,.0f}")
            else:
                print(f"\n❌ {stock['name']} ({stock['code']}): {result.get('message', '未知错误')}")
        
        # 汇总统计
        print("\n" + "="*80)
        print("投资组合汇总")
        print("="*80)
        
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            total_value = sum(r['total_value'] for r in successful)
            total_profit = total_value - self.initial_capital
            total_return = (total_profit / self.initial_capital) * 100
            
            buy_signals = sum(1 for r in successful if r['action'] in [1, 2, 3])
            sell_signals = sum(1 for r in successful if r['action'] in [4, 5, 6])
            hold_signals = sum(1 for r in successful if r['action'] == 0)
            
            print(f"\n当前总资产: ¥{total_value:,.0f}")
            print(f"累计收益: ¥{total_profit:,.0f} ({total_return:+.2f}%)")
            print(f"\n交易信号分布:")
            print(f"  买入信号: {buy_signals} 只")
            print(f"  卖出信号: {sell_signals} 只")
            print(f"  持有信号: {hold_signals} 只")
        
        print("\n" + "="*80)
        print("⚠️  风险提示")
        print("="*80)
        print("1. 以上信号基于历史数据训练的AI模型，仅供参考")
        print("2. 实际交易需考虑：交易费用、滑点、市场流动性")
        print("3. 建议结合实时新闻、政策、基本面等因素综合判断")
        print("4. 股市有风险，投资需谨慎")
        print("="*80 + "\n")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V7实盘交易信号生成器')
    parser.add_argument('--date', type=str, default=None,
                      help='目标日期（格式：YYYY-MM-DD），默认为最新数据日期')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                      help=f'模型文件路径，默认：{MODEL_PATH}')
    
    args = parser.parse_args()
    
    try:
        trader = RealtimeTrader(
            model_path=args.model,
            initial_capital=INITIAL_CAPITAL
        )
        
        results = trader.generate_portfolio_report(target_date=args.date)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

