# get_stock_data_v7_myportfolio.py - V7版本（用户自选股）
# -*- coding: utf-8 -*-
"""
V7 基于用户实际自选股进行训练
共12只：金融6只 + 消费2只 + 科技2只 + 医药1只 + 周期1只
"""
import baostock as bs
import pandas as pd
import os

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V7 股票列表（用户自选）
stocks = [
    # === 金融板块（6只）===
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.601838", "name": "成都银行", "start_date": "2018-01-04", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.601318", "name": "中国平安", "start_date": "2007-03-01", 
     "category": "金融", "volatility": "中", "style": "平衡"},
    {"code": "sh.601939", "name": "建设银行", "start_date": "2007-09-25", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.601398", "name": "工商银行", "start_date": "2006-10-27", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sz.513750", "name": "港股通非银ETF", "start_date": "2020-09-29",
     "category": "金融", "volatility": "中", "style": "平衡", "is_etf": True},
    
    # === 消费板块（2只）===
    {"code": "sz.000858", "name": "五粮液", "start_date": "1998-04-27", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    {"code": "sz.159928", "name": "消费ETF", "start_date": "2013-02-06", 
     "category": "消费", "volatility": "中", "style": "平衡", "is_etf": True},
    
    # === 科技板块（2只）===
    {"code": "sh.588000", "name": "科创50", "start_date": "2020-09-22", 
     "category": "科技", "volatility": "高", "style": "激进", "is_etf": True},
    {"code": "sh.588760", "name": "人工智能ETF", "start_date": "2023-05-08",
     "category": "科技", "volatility": "高", "style": "激进", "is_etf": True},
    
    # === 医药板块（1只）===
    {"code": "sh.516080", "name": "创新药ETF", "start_date": "2021-09-27", 
     "category": "医药", "volatility": "高", "style": "激进", "is_etf": True},
    
    # === 周期板块（1只）===
    {"code": "sh.515210", "name": "钢铁ETF", "start_date": "2019-12-05", 
     "category": "周期", "volatility": "中", "style": "平衡", "is_etf": True},
]

print(f"\n总共 {len(stocks)} 只标的（用户自选）")

# 按分类统计
from collections import Counter
category_count = Counter([s['category'] for s in stocks])
print(f"\n按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

# 按风格统计
style_count = Counter([s['style'] for s in stocks])
print(f"\n按风格分布:")
for style, count in style_count.items():
    print(f"  - {style}: {count}只")

# ETF统计
etf_count = sum(1 for s in stocks if s.get('is_etf', False))
stock_count = len(stocks) - etf_count
print(f"\n标的类型:")
print(f"  - 个股: {stock_count}只")
print(f"  - ETF: {etf_count}只")

print("\n" + "="*70)
print("开始下载数据...")
print("="*70 + "\n")

success_count = 0
fail_count = 0

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    is_etf = stock.get("is_etf", False)
    
    type_str = "ETF" if is_etf else "个股"
    print(f"[{category}|{type_str}] 查询 {code} ({name}), 起始: {start_date}")
    
    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
            start_date=start_date, 
            end_date='2025-11-22',
            frequency="d", 
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"  [失败] 查询错误: {rs.error_msg}")
            fail_count += 1
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            continue
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"  [成功] 获取 {len(result)} 条数据")
        
        # 数据预处理
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 分割训练集和测试集
        train_data = result[result['date'] <= '2024-12-31']
        test_data = result[result['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            continue
        
        # 保存
        os.makedirs('stockdata_v7/train', exist_ok=True)
        os.makedirs('stockdata_v7/test', exist_ok=True)
        
        train_file = f'stockdata_v7/train/{code}.{name}.csv'
        test_file = f'stockdata_v7/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    # 移除is_etf列（不需要存储）
    if 'is_etf' in metadata_df.columns:
        metadata_df = metadata_df.drop('is_etf', axis=1)
    metadata_df.to_csv('stockdata_v7/metadata_v7.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7/metadata_v7.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v7.py")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()



