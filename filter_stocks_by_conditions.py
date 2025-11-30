# -*- coding: utf-8 -*-
"""
A股股票筛选工具 V2 - 去掉月线限制
筛选条件：
1. 日K线、周K线图均为金叉（MACD或均线金叉）
2. 非亏损（盈利）
3. 股价在10元-100元之间
4. 流通市值500亿以下
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据源
TUSHARE_AVAILABLE = False
AKSHARE_AVAILABLE = False
BAOSTOCK_AVAILABLE = False

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    pass

# ==================== 配置参数 ====================
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')  # Tushare token（从环境变量或这里设置）

# 筛选条件
MIN_PRICE = 10.0      # 最低股价（元）
MAX_PRICE = 100.0     # 最高股价（元）
MAX_CIRC_MARKET_CAP = 500.0  # 最大流通市值（亿元）

# 技术指标参数
MA_SHORT = 5   # 短期均线（日）
MA_LONG = 20   # 长期均线（日）
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ==================== 初始化数据源 ====================
print("=" * 70)
print("A股股票筛选工具 V2 - 多条件筛选（去掉月线限制）")
print("=" * 70)

DATA_SOURCE = None
pro = None

# 优先尝试 Tushare
if TUSHARE_AVAILABLE and TUSHARE_TOKEN:
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        DATA_SOURCE = "tushare"
        print("✅ 数据源: Tushare")
    except Exception as e:
        print(f"⚠️  Tushare 初始化失败: {e}")
        TUSHARE_AVAILABLE = False

# 如果 Tushare 不可用，尝试 AkShare
if DATA_SOURCE is None and AKSHARE_AVAILABLE:
    try:
        DATA_SOURCE = "akshare"
        print("✅ 数据源: AkShare")
    except Exception as e:
        print(f"⚠️  AkShare 初始化失败: {e}")
        AKSHARE_AVAILABLE = False

# 如果前两者都不可用，尝试 baostock
if DATA_SOURCE is None and BAOSTOCK_AVAILABLE:
    try:
        bs.login()
        DATA_SOURCE = "baostock"
        print("✅ 数据源: baostock")
    except Exception as e:
        print(f"⚠️  baostock 初始化失败: {e}")
        BAOSTOCK_AVAILABLE = False

# 如果所有数据源都不可用，报错
if DATA_SOURCE is None:
    raise Exception("未找到可用的数据源！请安装 tushare、akshare 或 baostock")

print(f"📌 筛选条件:")
print(f"   1. 日K/周K线均为金叉（MACD或均线）")
print(f"   2. 非亏损（盈利）")
print(f"   3. 股价: {MIN_PRICE}-{MAX_PRICE} 元")
print(f"   4. 流通市值: < {MAX_CIRC_MARKET_CAP} 亿元")
print("=" * 70)
print()

# ==================== 辅助函数 ====================

def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    df = df.copy()
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    return df

def calculate_ma(df, short=5, long=20):
    """计算均线"""
    df = df.copy()
    df['MA_short'] = df['close'].rolling(window=short).mean()
    df['MA_long'] = df['close'].rolling(window=long).mean()
    return df

def check_golden_cross_macd(df):
    """检查MACD金叉（DIF上穿DEA）"""
    if len(df) < 2:
        return False
    # 当前DIF > DEA 且 上一期DIF <= DEA
    current = df.iloc[-1]
    prev = df.iloc[-2]
    return current['DIF'] > current['DEA'] and prev['DIF'] <= prev['DEA']

def check_golden_cross_ma(df):
    """检查均线金叉（短期均线上穿长期均线）"""
    if len(df) < 2:
        return False
    # 当前MA_short > MA_long 且 上一期MA_short <= MA_long
    current = df.iloc[-1]
    prev = df.iloc[-2]
    return current['MA_short'] > current['MA_long'] and prev['MA_short'] <= prev['MA_long']

def get_stock_list():
    """获取A股股票列表"""
    global DATA_SOURCE, TUSHARE_AVAILABLE, AKSHARE_AVAILABLE, BAOSTOCK_AVAILABLE, pro
    stock_list = []
    
    if DATA_SOURCE == "tushare" and TUSHARE_AVAILABLE:
        try:
            # 获取A股股票列表
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
            # 过滤：只保留主板、中小板、创业板（排除科创板、北交所等）
            df = df[~df['market'].isin(['科创板', '北交所'])]
            for _, row in df.iterrows():
                stock_list.append({
                    'ts_code': row['ts_code'],
                    'code': row['symbol'],
                    'name': row['name'],
                    'market': 'sh' if row['ts_code'].endswith('.SH') else 'sz'
                })
            print(f"✅ 从Tushare获取 {len(stock_list)} 只股票")
        except Exception as e:
            print(f"⚠️  Tushare获取股票列表失败: {e}")
    
    elif DATA_SOURCE == "akshare" and AKSHARE_AVAILABLE:
        try:
            # 获取A股股票列表
            df = ak.stock_info_a_code_name()
            for _, row in df.iterrows():
                code = row['code']
                name = row['name']
                market = 'sh' if code.startswith('6') else 'sz'
                stock_list.append({
                    'code': code,
                    'name': name,
                    'market': market,
                    'ts_code': f"{code}.{market.upper()}" if market == 'sh' else f"{code}.SZ"
                })
            print(f"✅ 从AkShare获取 {len(stock_list)} 只股票")
        except Exception as e:
            print(f"⚠️  AkShare获取股票列表失败: {e}")
    
    elif DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        try:
            # 获取沪深A股列表
            rs = bs.query_all_stock(day=datetime.now().strftime('%Y-%m-%d'))
            while rs.next():
                row = rs.get_row_data()
                code = row[0]
                name = row[1]
                if code.startswith('sh.6') or code.startswith('sz.0') or code.startswith('sz.3'):
                    stock_list.append({
                        'code': code.split('.')[1],
                        'name': name,
                        'market': 'sh' if code.startswith('sh') else 'sz',
                        'ts_code': f"{code.split('.')[1]}.{code.split('.')[0].upper()}"
                    })
            print(f"✅ 从baostock获取 {len(stock_list)} 只股票")
        except Exception as e:
            print(f"⚠️  baostock获取股票列表失败: {e}")
    
    return stock_list

def get_kline_data(code, market, period='daily', days=250):
    """获取K线数据（日K/周K/月K）"""
    global DATA_SOURCE, TUSHARE_AVAILABLE, AKSHARE_AVAILABLE, BAOSTOCK_AVAILABLE, pro
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    df = None
    
    if DATA_SOURCE == "tushare" and TUSHARE_AVAILABLE:
        try:
            ts_code = f"{code}.{market.upper()}" if market == 'sh' else f"{code}.SZ"
            if period == 'daily':
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            elif period == 'weekly':
                df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
            elif period == 'monthly':
                df = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is not None and len(df) > 0:
                df = df.rename(columns={'trade_date': 'date', 'close': 'close', 'open': 'open', 
                                       'high': 'high', 'low': 'low', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df = df.sort_values('date')
        except:
            pass
    
    elif DATA_SOURCE == "akshare" and AKSHARE_AVAILABLE:
        try:
            if period == 'daily':
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, 
                                        end_date=end_date, adjust="qfq")
            elif period == 'weekly':
                df = ak.stock_zh_a_hist(symbol=code, period="weekly", start_date=start_date, 
                                        end_date=end_date, adjust="qfq")
            elif period == 'monthly':
                df = ak.stock_zh_a_hist(symbol=code, period="monthly", start_date=start_date, 
                                        end_date=end_date, adjust="qfq")
            
            if df is not None and len(df) > 0:
                df = df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open',
                                       '最高': 'high', '最低': 'low', '成交量': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
        except:
            pass
    
    elif DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        try:
            bs_code = f"{market}.{code}"
            freq_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm'}
            rs = bs.query_history_k_data_plus(bs_code, 
                "date,open,high,low,close,volume",
                start_date=start_date, end_date=end_date,
                frequency=freq_map[period], adjustflag="3")
            
            if rs.error_code == '0':
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    df['date'] = pd.to_datetime(df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.sort_values('date')
        except:
            pass
    
    return df

def get_stock_basic_info(code, market):
    """获取股票基本信息（股价、流通市值、是否亏损）"""
    global DATA_SOURCE, TUSHARE_AVAILABLE, AKSHARE_AVAILABLE, BAOSTOCK_AVAILABLE, pro
    info = {
        'current_price': 0.0,
        'circ_market_cap': 0.0,  # 流通市值（亿元）
        'is_profitable': True,
        'pe': 0.0,
        'pb': 0.0
    }
    
    if DATA_SOURCE == "tushare" and TUSHARE_AVAILABLE:
        try:
            ts_code = f"{code}.{market.upper()}" if market == 'sh' else f"{code}.SZ"
            # 获取实时行情
            df = pro.daily_basic(ts_code=ts_code, trade_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                info['current_price'] = float(df['close'].iloc[0]) if 'close' in df.columns else 0.0
                info['circ_market_cap'] = float(df['circ_mv'].iloc[0]) / 10000 if 'circ_mv' in df.columns else 0.0  # 转换为亿元
                info['pe'] = float(df['pe'].iloc[0]) if 'pe' in df.columns else 0.0
                info['pb'] = float(df['pb'].iloc[0]) if 'pb' in df.columns else 0.0
            
            # 获取财务数据判断是否亏损
            try:
                fina = pro.fina_indicator(ts_code=ts_code, period='20231231')  # 最新年报
                if fina is not None and len(fina) > 0:
                    net_profit = float(fina['net_profit'].iloc[0]) if 'net_profit' in fina.columns else 0.0
                    info['is_profitable'] = net_profit > 0
            except:
                pass
        except:
            pass
    
    elif DATA_SOURCE == "akshare" and AKSHARE_AVAILABLE:
        try:
            # 获取实时行情
            df = ak.stock_zh_a_spot_em()
            stock_df = df[df['代码'] == code]
            if len(stock_df) > 0:
                info['current_price'] = float(stock_df['最新价'].iloc[0])
                info['circ_market_cap'] = float(stock_df['流通市值'].iloc[0]) / 100000000  # 转换为亿元
            
            # 获取财务数据
            try:
                fina = ak.stock_financial_em(symbol=code, indicator="财务指标")
                if fina is not None and len(fina) > 0:
                    latest = fina.iloc[0]
                    net_profit = float(latest.get('净利润', 0) or 0)
                    info['is_profitable'] = net_profit > 0
            except:
                pass
        except:
            pass
    
    elif DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        try:
            bs_code = f"{market}.{code}"
            rs = bs.query_history_k_data_plus(bs_code, "date,close,peTTM,pbMRQ",
                start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'), frequency="d", adjustflag="3")
            if rs.error_code == '0':
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    info['current_price'] = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
                    info['pe'] = float(df['peTTM'].iloc[-1]) if 'peTTM' in df.columns and len(df) > 0 else 0.0
                    info['pb'] = float(df['pbMRQ'].iloc[-1]) if 'pbMRQ' in df.columns and len(df) > 0 else 0.0
        except:
            pass
    
    return info

def check_all_golden_cross(code, market):
    """检查日K、周K是否均为金叉（已去掉月线限制）"""
    results = {
        'daily_macd': False,
        'daily_ma': False,
        'weekly_macd': False,
        'weekly_ma': False
    }
    
    # 检查日K
    daily_df = get_kline_data(code, market, period='daily', days=250)
    if daily_df is not None and len(daily_df) >= max(MA_LONG, MACD_SLOW):
        daily_df = calculate_macd(daily_df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        daily_df = calculate_ma(daily_df, MA_SHORT, MA_LONG)
        results['daily_macd'] = check_golden_cross_macd(daily_df)
        results['daily_ma'] = check_golden_cross_ma(daily_df)
    
    # 检查周K
    weekly_df = get_kline_data(code, market, period='weekly', days=500)
    if weekly_df is not None and len(weekly_df) >= max(MA_LONG, MACD_SLOW):
        weekly_df = calculate_macd(weekly_df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        weekly_df = calculate_ma(weekly_df, MA_SHORT, MA_LONG)
        results['weekly_macd'] = check_golden_cross_macd(weekly_df)
        results['weekly_ma'] = check_golden_cross_ma(weekly_df)
    
    # 日K、周K均为金叉（MACD或均线任一满足即可）
    all_golden_cross = (
        (results['daily_macd'] or results['daily_ma']) and
        (results['weekly_macd'] or results['weekly_ma'])
    )
    
    return all_golden_cross, results

# ==================== 主筛选流程 ====================

def main():
    print("📊 开始筛选A股股票...")
    print()
    
    # 获取股票列表
    stock_list = get_stock_list()
    if len(stock_list) == 0:
        print("❌ 未获取到股票列表")
        return
    
    print(f"📋 共 {len(stock_list)} 只股票待筛选")
    print()
    
    # 筛选结果
    filtered_stocks = []
    checked_count = 0
    
    for i, stock in enumerate(stock_list):
        code = stock['code']
        name = stock['name']
        market = stock['market']
        
        checked_count += 1
        if checked_count % 100 == 0:
            print(f"   进度: {checked_count}/{len(stock_list)} (已筛选出 {len(filtered_stocks)} 只符合条件的股票)")
        
        try:
            # 1. 获取基本信息（股价、流通市值、是否亏损）
            basic_info = get_stock_basic_info(code, market)
            current_price = basic_info['current_price']
            circ_market_cap = basic_info['circ_market_cap']
            is_profitable = basic_info['is_profitable']
            
            # 2. 初步筛选：价格、市值、盈利
            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                continue
            if circ_market_cap >= MAX_CIRC_MARKET_CAP or circ_market_cap <= 0:
                continue
            if not is_profitable:
                continue
            
            # 3. 检查日K、周K是否均为金叉
            all_golden_cross, cross_details = check_all_golden_cross(code, market)
            if not all_golden_cross:
                continue
            
            # 4. 符合所有条件，添加到结果
            filtered_stocks.append({
                'code': code,
                'name': name,
                'market': market,
                'current_price': current_price,
                'circ_market_cap': circ_market_cap,
                'pe': basic_info['pe'],
                'pb': basic_info['pb'],
                'golden_cross_details': cross_details
            })
            
            print(f"   ✅ {code} {name}: 价格={current_price:.2f}元, 流通市值={circ_market_cap:.2f}亿元")
            
            # 避免请求过快
            time.sleep(0.1)
            
        except Exception as e:
            # 出错时跳过
            continue
    
    # 输出结果
    print()
    print("=" * 70)
    print(f"📊 筛选完成！共找到 {len(filtered_stocks)} 只符合条件的股票")
    print("=" * 70)
    
    if len(filtered_stocks) > 0:
        # 保存到CSV
        result_df = pd.DataFrame(filtered_stocks)
        result_df = result_df.sort_values('current_price')
        
        output_file = f"filtered_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 结果已保存到: {output_file}")
        print()
        
        # 显示前20只
        print("📋 符合条件的股票列表（按股价排序，显示前20只）:")
        print("-" * 70)
        for idx, stock in enumerate(result_df.head(20).iterrows(), 1):
            row = stock[1]
            print(f"{idx:2d}. {row['code']:6s} {row['name']:10s} | "
                  f"价格: {row['current_price']:6.2f}元 | "
                  f"流通市值: {row['circ_market_cap']:7.2f}亿元 | "
                  f"PE: {row['pe']:6.2f} | PB: {row['pb']:6.2f}")
        
        if len(result_df) > 20:
            print(f"... 还有 {len(result_df) - 20} 只股票，请查看CSV文件")
    else:
        print("⚠️  未找到符合条件的股票")
    
    # 清理资源
    if DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        bs.logout()

if __name__ == "__main__":
    main()

