# get_stock_data_v7_tushare.py - V7版本（使用Tushare获取ETF数据）
# -*- coding: utf-8 -*-
"""
V7 完整版：
- 个股用baostock（免费，稳定）
- ETF用Tushare（需要token，但支持完整）
"""
import baostock as bs
import tushare as ts
import pandas as pd
import os
from datetime import datetime

# 设置Tushare token
TUSHARE_TOKEN = 'bfdcf2e84642bcba86bea58cca72c6aca61b7bceb24ad68267209a5f'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

print("="*70)
print("V7 Tushare版 - 数据获取")
print("="*70)

# V7 股票列表（用户自选）
stocks = [
    # === 金融板块（6只：5个股+1ETF）===
    {"code": "600036", "name": "招商银行", "start_date": "20020409", 
     "category": "金融", "volatility": "低", "style": "稳健", "type": "stock"},
    {"code": "601838", "name": "成都银行", "start_date": "20180104", 
     "category": "金融", "volatility": "低", "style": "稳健", "type": "stock"},
    {"code": "601318", "name": "中国平安", "start_date": "20070301", 
     "category": "金融", "volatility": "中", "style": "平衡", "type": "stock"},
    {"code": "601939", "name": "建设银行", "start_date": "20070925", 
     "category": "金融", "volatility": "低", "style": "稳健", "type": "stock"},
    {"code": "601398", "name": "工商银行", "start_date": "20061027", 
     "category": "金融", "volatility": "低", "style": "稳健", "type": "stock"},
    {"code": "513750", "name": "港股通非银ETF", "start_date": "20200929", 
     "category": "金融", "volatility": "中", "style": "平衡", "type": "etf"},
    
    # === 消费板块（2只：1个股+1ETF）===
    {"code": "000858", "name": "五粮液", "start_date": "19980427", 
     "category": "消费", "volatility": "中", "style": "平衡", "type": "stock"},
    {"code": "159928", "name": "消费ETF", "start_date": "20130206", 
     "category": "消费", "volatility": "中", "style": "平衡", "type": "etf"},
    
    # === 科技板块（2只ETF）===
    {"code": "588000", "name": "科创50", "start_date": "20200922", 
     "category": "科技", "volatility": "高", "style": "激进", "type": "etf"},
    {"code": "588760", "name": "人工智能ETF", "start_date": "20230508", 
     "category": "科技", "volatility": "高", "style": "激进", "type": "etf"},
    
    # === 医药板块（1只ETF）===
    {"code": "516080", "name": "创新药ETF", "start_date": "20210927", 
     "category": "医药", "volatility": "高", "style": "激进", "type": "etf"},
    
    # === 周期板块（1只ETF）===
    {"code": "515210", "name": "钢铁ETF", "start_date": "20191205", 
     "category": "周期", "volatility": "中", "style": "平衡", "type": "etf"},
]

print(f"\n总共 {len(stocks)} 只标的（用户自选完整版）")

# 统计
from collections import Counter
category_count = Counter([s['category'] for s in stocks])
type_count = Counter([s['type'] for s in stocks])

print(f"\n按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

print(f"\n按类型分布:")
print(f"  - 个股: {type_count['stock']}只（用baostock）")
print(f"  - ETF: {type_count['etf']}只（用Tushare）")

print("\n" + "="*70)
print("开始下载数据...")
print("="*70 + "\n")

# 初始化baostock
lg = bs.login(user_id="anonymous", password="123456")
print(f"Baostock登录: {lg.error_msg}\n")

success_count = 0
fail_count = 0

os.makedirs('stockdata_v7/train', exist_ok=True)
os.makedirs('stockdata_v7/test', exist_ok=True)

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    stock_type = stock["type"]
    
    print(f"[{category}|{stock_type.upper()}] 下载 {code} ({name})")
    
    try:
        if stock_type == "stock":
            # 使用baostock获取个股数据
            # 转换代码格式
            if code.startswith('6'):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                start_date=start_date, 
                end_date='20251122',
                frequency="d", 
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                print(f"  [失败] {rs.error_msg}")
                fail_count += 1
                continue
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            if len(data_list) == 0:
                print(f"  [失败] 无数据")
                fail_count += 1
                continue
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
        else:  # ETF用Tushare
            # Tushare ETF代码格式
            if code.startswith('5'):
                ts_code = f"{code}.SH"
            else:
                ts_code = f"{code}.SZ"
            
            # 获取ETF日线数据
            df = pro.fund_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date='20251122'
            )
            
            if df is None or len(df) == 0:
                print(f"  [失败] 无数据")
                fail_count += 1
                continue
            
            # 转换列名以匹配baostock格式
            df = df.rename(columns={
                'ts_code': 'code',
                'trade_date': 'date',
                'pre_close': 'preclose',
                'pct_chg': 'pctChg',
                'vol': 'volume',
                'amount': 'amount'
            })
            
            # 添加缺失列
            df['turn'] = 0
            df['tradestatus'] = 1
            df['peTTM'] = 0
            df['psTTM'] = 0
            df['pcfNcfTTM'] = 0
            df['pbMRQ'] = 0
            df['adjustflag'] = 3
            df['isST'] = 0
            
            # 金额单位转换（Tushare是千元，转为元）
            df['amount'] = df['amount'] * 1000
            
            # 按日期排序（Tushare是降序）
            df = df.sort_values('date')
        
        # 统一处理
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  [成功] 获取 {len(df)} 条数据")
        
        # 分割训练集和测试集
        train_data = df[df['date'] <= '2024-12-31']
        test_data = df[df['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            continue
        
        # 保存文件
        # 统一代码格式
        if code.startswith('6'):
            file_code = f"sh.{code}"
        elif code.startswith('0') or code.startswith('3'):
            file_code = f"sz.{code}"
        else:
            if code in ['588000', '588760', '513750', '512480', '516080', '515070', '588030', '516160']:
                file_code = f"sh.{code}"
            else:
                file_code = f"sz.{code}"
        
        train_file = f'stockdata_v7/train/{file_code}.{name}.csv'
        test_file = f'stockdata_v7/test/{file_code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
    
    print()

# 退出baostock
bs.logout()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if success_count >= 10:
    print(f"\n[优秀] 成功{success_count}只，数据完整！")
elif success_count >= 8:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df = metadata_df.drop('type', axis=1)  # 移除type列
    metadata_df.to_csv('stockdata_v7/metadata_v7.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7/metadata_v7.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v7.py")
    print("\n[说明] 使用了Tushare获取ETF数据，数据更完整！")
else:
    print("\n[失败] 没有成功下载任何数据")



