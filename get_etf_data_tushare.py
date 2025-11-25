# get_etf_data_tushare.py - 只用Tushare获取ETF数据（节省积分）
# -*- coding: utf-8 -*-
"""
优化方案：
- 6只个股已有数据（用baostock获取）✅
- 6只ETF用Tushare获取（节省积分）
"""
import tushare as ts
import pandas as pd
import os

# 设置Tushare token
TUSHARE_TOKEN = 'bfdcf2e84642bcba86bea58cca72c6aca61b7bceb24ad68267209a5f'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

print("="*70)
print("Tushare ETF数据获取（仅获取6只ETF，节省积分）")
print("="*70)

# 只获取baostock无法获取的6只ETF
etf_list = [
    {"code": "513750", "name": "港股通非银ETF", "start_date": "20200929", 
     "category": "金融", "volatility": "中", "style": "平衡"},
    {"code": "159928", "name": "消费ETF", "start_date": "20130206", 
     "category": "消费", "volatility": "中", "style": "平衡"},
    {"code": "588000", "name": "科创50", "start_date": "20200922", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "588760", "name": "人工智能ETF", "start_date": "20230508", 
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "516080", "name": "创新药ETF", "start_date": "20210927", 
     "category": "医药", "volatility": "高", "style": "激进"},
    {"code": "515210", "name": "钢铁ETF", "start_date": "20191205", 
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
    start_date = etf["start_date"]
    category = etf["category"]
    
    # Tushare ETF代码格式
    if code.startswith('5'):
        ts_code = f"{code}.SH"
    else:
        ts_code = f"{code}.SZ"
    
    print(f"[{category}] 下载 {ts_code} ({name})")
    
    try:
        # 获取ETF日线数据
        df = pro.fund_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date='20251122'
        )
        
        if df is None or len(df) == 0:
            print(f"  [失败] 无数据")
            fail_count += 1
            print()
            continue
        
        print(f"  [成功] 获取 {len(df)} 条数据")
        
        # 转换列名以匹配baostock格式
        df = df.rename(columns={
            'ts_code': 'code',
            'trade_date': 'date',
            'pre_close': 'preclose',
            'pct_chg': 'pctChg',
            'vol': 'volume',
        })
        
        # 添加缺失列（保持与baostock格式一致）
        df['turn'] = 0
        df['tradestatus'] = 1
        df['peTTM'] = 0
        df['psTTM'] = 0
        df['pcfNcfTTM'] = 0
        df['pbMRQ'] = 0
        df['adjustflag'] = 3
        df['isST'] = 0
        
        # 金额单位转换（Tushare是千元，转为元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000
        else:
            df['amount'] = 0
        
        # 日期格式转换
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
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
        
        # 确定文件名前缀
        if code.startswith('5'):
            if code in ['588000', '588760', '516080', '515070', '588030', '516160']:
                file_prefix = 'sh'
            elif code.startswith('51'):
                file_prefix = 'sh'
            elif code.startswith('56'):
                file_prefix = 'sh'
            else:
                file_prefix = 'sh'
        else:
            file_prefix = 'sz'
        
        # 保存文件
        train_file = f'stockdata_v7/train/{file_prefix}.{code}.{name}.csv'
        test_file = f'stockdata_v7/test/{file_prefix}.{code}.{name}.csv'
        
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
print("完成！节省Tushare积分的同时获取了完整数据")
print("="*70)



