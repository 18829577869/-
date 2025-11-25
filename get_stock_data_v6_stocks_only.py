# get_stock_data_v6_stocks_only.py - V6版本（仅使用个股，兼容baostock）
# -*- coding: utf-8 -*-
"""
V6 备选方案：使用个股替代ETF
baostock对个股支持更好
"""
import baostock as bs
import pandas as pd
import os

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V6 股票列表（全部使用个股）
stocks = [
    # === 金融板块 ===
    {"code": "sh.600000", "name": "浦发银行", "start_date": "1999-11-10", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.601166", "name": "兴业银行", "start_date": "2007-02-05", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.601318", "name": "中国平安", "start_date": "2007-03-01", 
     "category": "金融", "volatility": "中", "style": "平衡"},
    
    # === 科技板块 ===
    {"code": "sz.000063", "name": "中兴通讯", "start_date": "1997-11-18", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sz.002415", "name": "海康威视", "start_date": "2010-05-28", 
     "category": "科技", "volatility": "中", "style": "平衡"},
    {"code": "sz.300059", "name": "东方财富", "start_date": "2010-03-19", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sz.001389", "name": "广合科技", "start_date": "2024-04-02", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.688981", "name": "中芯国际", "start_date": "2020-07-16", 
     "category": "科技", "volatility": "高", "style": "激进"},
    
    # === 消费板块 ===
    {"code": "sh.600519", "name": "贵州茅台", "start_date": "2001-08-27", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    {"code": "sz.000858", "name": "五粮液", "start_date": "1998-04-27", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    {"code": "sz.000333", "name": "美的集团", "start_date": "2013-09-18", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    
    # === 医药板块 ===
    {"code": "sz.300015", "name": "爱尔眼科", "start_date": "2009-10-30", 
     "category": "医药", "volatility": "高", "style": "激进"},
    {"code": "sh.600276", "name": "恒瑞医药", "start_date": "2000-10-18", 
     "category": "医药", "volatility": "中", "style": "平衡"},
    
    # === 新能源板块 ===
    {"code": "sz.300750", "name": "宁德时代", "start_date": "2018-06-11", 
     "category": "新能源", "volatility": "高", "style": "激进"},
    {"code": "sh.601012", "name": "隆基绿能", "start_date": "2012-04-11", 
     "category": "新能源", "volatility": "高", "style": "激进"},
    
    # === 周期板块 ===
    {"code": "sz.002083", "name": "孚日股份", "start_date": "2006-11-24", 
     "category": "周期", "volatility": "高", "style": "激进"},
    {"code": "sh.600418", "name": "江淮汽车", "start_date": "2001-08-24", 
     "category": "周期", "volatility": "中", "style": "平衡"},
    {"code": "sh.600111", "name": "北方稀土", "start_date": "1997-09-02", 
     "category": "周期", "volatility": "高", "style": "激进"},
]

print(f"\n总共 {len(stocks)} 只标的（全部为个股）")

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
    
    print(f"[{category}] 查询 {code} ({name}), 起始: {start_date}")
    
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
        os.makedirs('stockdata/train', exist_ok=True)
        os.makedirs('stockdata/test', exist_ok=True)
        
        train_file = f'stockdata/train/{code}.{name}.csv'
        test_file = f'stockdata/test/{code}.{name}.csv'
        
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
    metadata_df.to_csv('stockdata/metadata_v6.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata/metadata_v6.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v6.py")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()



