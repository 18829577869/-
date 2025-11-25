# get_stock_data_v6.py - 获取V6版本数据（包含新增ETF）
# -*- coding: utf-8 -*-
"""
V6 新增多只科技和主题ETF
分类：科技、消费、周期、创新
"""
import baostock as bs
import pandas as pd
import os

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V6 股票列表（增强版）
stocks = [
    # === 原有股票 ===
    {"code": "sh.600000", "name": "浦发银行", "start_date": "1999-11-10", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "sz.002083", "name": "孚日股份", "start_date": "2006-11-24", 
     "category": "周期", "volatility": "高", "style": "激进"},
    {"code": "sz.001389", "name": "广合科技", "start_date": "2024-04-02", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.600418", "name": "江淮汽车", "start_date": "2001-08-24", 
     "category": "周期", "volatility": "中", "style": "平衡"},
    
    # === 原有ETF ===
    {"code": "sz.159966", "name": "创蓝筹", "start_date": "2017-11-03", 
     "category": "宽基", "volatility": "中", "style": "平衡"},
    {"code": "sz.159876", "name": "有色基金", "start_date": "2016-06-03", 
     "category": "周期", "volatility": "高", "style": "激进"},
    {"code": "sz.159928", "name": "消费ETF", "start_date": "2013-02-06", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    
    # === V6 新增ETF ===
    {"code": "sh.513180", "name": "恒生科技", "start_date": "2020-12-16", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.512480", "name": "半导体ETF", "start_date": "2019-05-16", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.513090", "name": "港股非银", "start_date": "2020-09-25", 
     "category": "金融", "volatility": "中", "style": "平衡"},
    {"code": "sz.159992", "name": "创新药", "start_date": "2021-09-23", 
     "category": "医药", "volatility": "高", "style": "激进"},
    {"code": "sz.159770", "name": "机器人", "start_date": "2021-09-02", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.588000", "name": "科创50", "start_date": "2020-09-22", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.515070", "name": "AI创新", "start_date": "2023-05-08", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.588030", "name": "科创智能", "start_date": "2023-08-28", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "sh.516160", "name": "新能源", "start_date": "2021-02-19", 
     "category": "新能源", "volatility": "高", "style": "激进"},
]

print(f"\n总共 {len(stocks)} 只标的")
print(f"  - 原有: 8只")
print(f"  - 新增: {len(stocks) - 8}只")

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
            end_date='2025-11-22',  # 更新到最新
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

# 保存元数据
metadata_df = pd.DataFrame(stocks)
metadata_df.to_csv('stockdata/metadata_v6.csv', index=False, encoding='utf-8-sig')
print(f"\n元数据已保存: stockdata/metadata_v6.csv")

bs.logout()



