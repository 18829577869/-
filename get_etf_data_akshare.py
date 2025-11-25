# get_etf_data_akshare.py - 用AkShare获取ETF数据（免费，无需积分）
# -*- coding: utf-8 -*-
"""
优化方案：
- 6只个股已有数据（baostock）✅
- 6只ETF用AkShare获取（免费，无积分限制）✅
"""
import akshare as ak
import pandas as pd
import os
from datetime import datetime

print("="*70)
print("AkShare ETF数据获取（仅获取6只ETF，免费）")
print("="*70)

# 只获取baostock无法获取的6只ETF
etf_list = [
    {"code": "513750", "name": "港股通非银ETF", "market": "sh",
     "category": "金融", "volatility": "中", "style": "平衡"},
    {"code": "159928", "name": "消费ETF", "market": "sz",
     "category": "消费", "volatility": "中", "style": "平衡"},
    {"code": "588000", "name": "科创50", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "588760", "name": "人工智能ETF", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "516080", "name": "创新药ETF", "market": "sh",
     "category": "医药", "volatility": "高", "style": "激进"},
    {"code": "515210", "name": "钢铁ETF", "market": "sh",
     "category": "周期", "volatility": "中", "style": "平衡"},
]

print(f"\n需要获取的ETF: {len(etf_list)} 只")
print("（个股数据已有，无需重复获取）\n")

for etf in etf_list:
    print(f"  - {etf['code']} {etf['name']} ({etf['category']})")

print("\n" + "="*70)
print("开始下载ETF数据...")
print("="*70 + "\n")

success_count = 0
fail_count = 0

os.makedirs('stockdata_v7/train', exist_ok=True)
os.makedirs('stockdata_v7/test', exist_ok=True)

for etf in etf_list:
    code = etf["code"]
    name = etf["name"]
    market = etf["market"]
    category = etf["category"]
    
    print(f"[{category}] 下载 {code} ({name})")
    
    try:
        # AkShare获取ETF数据
        # 尝试fund_etf_hist_em接口
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date="20100101",
                end_date="20251122",
                adjust="qfq"
            )
            print(f"  [成功] 使用fund_etf_hist_em接口")
        except Exception as e1:
            # 如果失败，尝试stock接口（有些ETF可能在这里）
            print(f"  [提示] fund_etf_hist_em失败，尝试stock接口...")
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date="20100101",
                end_date="20251122",
                adjust="qfq"
            )
            print(f"  [成功] 使用stock_zh_a_hist接口")
        
        if df is None or len(df) == 0:
            print(f"  [失败] 无数据")
            fail_count += 1
            print()
            continue
        
        print(f"  [成功] 获取 {len(df)} 条数据")
        
        # 转换列名以匹配baostock格式
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pctChg',
            '涨跌额': 'change',
            '换手率': 'turn'
        })
        
        # 添加缺失列（保持与baostock格式一致）
        if 'preclose' not in df.columns:
            df['preclose'] = df['close'].shift(1)
        
        df['tradestatus'] = 1
        df['peTTM'] = 0
        df['psTTM'] = 0
        df['pcfNcfTTM'] = 0
        df['pbMRQ'] = 0
        df['adjustflag'] = 3
        df['isST'] = 0
        df['code'] = f"{market}.{code}"
        
        # 确保必要的列存在
        if 'turn' not in df.columns:
            df['turn'] = 0
        
        # 日期格式转换
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 选择需要的列（与baostock一致）
        columns = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose',
                   'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'psTTM',
                   'pcfNcfTTM', 'pbMRQ', 'tradestatus']
        
        # 只保留存在的列
        df = df[[col for col in columns if col in df.columns]]
        
        # 补充缺失列
        for col in columns:
            if col not in df.columns:
                if col in ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'turn']:
                    df[col] = 0
                elif col == 'tradestatus':
                    df[col] = 1
                elif col == 'preclose':
                    df[col] = df['close'].shift(1)
        
        # 调整列顺序
        df = df[columns]
        
        # 分割训练集和测试集
        train_data = df[df['date'] <= '2024-12-31']
        test_data = df[df['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条: {len(train_data)}")
            if len(train_data) < 50:
                print(f"  [跳过] 数据太少")
                fail_count += 1
                print()
                continue
        
        # 保存文件
        train_file = f'stockdata_v7/train/{market}.{code}.{name}.csv'
        test_file = f'stockdata_v7/test/{market}.{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
    
    print()

print("="*70)
print("ETF数据下载完成")
print("="*70)
print(f"成功: {success_count} / {len(etf_list)} 只ETF")
print(f"失败: {fail_count} 只")

# 检查个股数据是否存在
print("\n" + "="*70)
print("检查已有的个股数据...")
print("="*70)

stock_files = [
    'sh.600036.招商银行.csv',
    'sh.601838.成都银行.csv',
    'sh.601318.中国平安.csv',
    'sh.601939.建设银行.csv',
    'sh.601398.工商银行.csv',
    'sz.000858.五粮液.csv'
]

existing_count = 0
for stock_file in stock_files:
    file_path = f'stockdata_v7/train/{stock_file}'
    if os.path.exists(file_path):
        print(f"  ✅ {stock_file}")
        existing_count += 1
    else:
        print(f"  ❌ {stock_file} [缺失]")

print(f"\n个股数据: {existing_count} / 6")

# 总计
total_success = success_count + existing_count
total_target = len(etf_list) + len(stock_files)

print("\n" + "="*70)
print("汇总统计")
print("="*70)
print(f"总标的数: {total_target}")
print(f"  - 个股（已有）: {existing_count} / 6")
print(f"  - ETF（新获取）: {success_count} / {len(etf_list)}")
print(f"  - 总计: {total_success} / {total_target}")

if total_success >= 10:
    print(f"\n[优秀] 成功获取{total_success}只，数据完整！")
    print("\n可以开始训练：")
    print("  python train_v7.py")
elif total_success >= 8:
    print(f"\n[良好] 成功获取{total_success}只，可以训练")
    print("\n可以开始训练：")
    print("  python train_v7.py")
else:
    print(f"\n[警告] 仅{total_success}只，建议至少8只")
    if success_count < 3:
        print("\n[建议] AkShare获取ETF失败较多，可以尝试：")
        print("  方案1：重新运行此脚本（可能网络问题）")
        print("  方案2：使用增强版（用个股替代）")
        print("    python get_stock_data_v7_enhanced.py")

# 更新元数据（合并个股和ETF）
if success_count > 0 or existing_count > 0:
    # 完整的12只标的元数据
    all_stocks = [
        # 个股
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
        {"code": "sz.000858", "name": "五粮液", "start_date": "1998-04-27", 
         "category": "消费", "volatility": "中", "style": "平衡"},
        # ETF
        {"code": "sh.513750", "name": "港股通非银ETF", "start_date": "2020-09-29",
         "category": "金融", "volatility": "中", "style": "平衡"},
        {"code": "sz.159928", "name": "消费ETF", "start_date": "2013-02-06", 
         "category": "消费", "volatility": "中", "style": "平衡"},
        {"code": "sh.588000", "name": "科创50", "start_date": "2020-09-22", 
         "category": "科技", "volatility": "高", "style": "激进"},
        {"code": "sh.588760", "name": "人工智能ETF", "start_date": "2023-05-08",
         "category": "科技", "volatility": "高", "style": "激进"},
        {"code": "sh.516080", "name": "创新药ETF", "start_date": "2021-09-27", 
         "category": "医药", "volatility": "高", "style": "激进"},
        {"code": "sh.515210", "name": "钢铁ETF", "start_date": "2019-12-05", 
         "category": "周期", "volatility": "中", "style": "平衡"},
    ]
    
    metadata_df = pd.DataFrame(all_stocks)
    metadata_df.to_csv('stockdata_v7/metadata_v7.csv', index=False, encoding='utf-8-sig')
    print(f"\n[保存] 元数据: stockdata_v7/metadata_v7.csv")

print("\n" + "="*70)
print("完成！使用AkShare获取ETF数据（免费，无积分限制）")
print("="*70)



