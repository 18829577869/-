import yfinance as yf
import pandas as pd
import os

# 股票列表：代码、名称、上市日期（start_date 为上市日期）
stocks = [
    {"code": "560560.SH", "name": "低碳科技", "start_date": "2021-08-27"},
    {"code": "510030.SH", "name": "华宝ETF", "start_date": "2010-05-28"},
    {"code": "513750.SH", "name": "港股非银", "start_date": "2023-11-10"},
    {"code": "159966.SZ", "name": "创蓝筹", "start_date": "2019-06-14"},
    {"code": "562500.SH", "name": "机器人", "start_date": "2021-03-12"},
    {"code": "588000.SH", "name": "科创50", "start_date": "2020-11-16"},
    {"code": "588733.SH", "name": "AI创", "start_date": "2025-01-15"},
    {"code": "588960.SH", "name": "科创能源", "start_date": "2025-02-06"},
    {"code": "516160.SH", "name": "新能源", "start_date": "2021-03-09"},
    {"code": "159876.SZ", "name": "有色基金", "start_date": "2021-03-12"},
    {"code": "159928.SZ", "name": "消费ETF", "start_date": "2013-08-23"}
]

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    print(f"查询 {code} ({name}), start_date: {start_date}")

    result = yf.download(code, start=start_date, end='2025-11-19')
    if result.empty:
        print(f"警告: {code} 无数据，跳过")
        continue

    # Flatten columns if multi-level
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    result = result.reset_index()
    result = result.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjustflag'})
    result['date'] = pd.to_datetime(result['Date'])
    result['code'] = code
    result['preclose'] = result['close'].shift(1)
    result['pctChg'] = (result['close'] - result['preclose']) / result['preclose'] * 100
    result['amount'] = result['volume'] * result['close']  # 估算
    result['turn'] = 0  # 默认
    result['tradestatus'] = '1'
    result['isST'] = '0'
    result['peTTM'] = 0  # 默认
    result['psTTM'] = 0
    result['pcfNcfTTM'] = 0
    result['pbMRQ'] = 0
    result = result.dropna()

    print(f"总行数: {len(result)}")
    print(result.head())
    print(result.tail())

    train_data = result[result['date'] <= '2024-12-31']
    test_data = result[result['date'] > '2024-12-31']

    os.makedirs('stockdata/train', exist_ok=True)
    os.makedirs('stockdata/test', exist_ok=True)
    train_data.to_csv(f'stockdata/train/{code}.{name}.csv', index=False)
    test_data.to_csv(f'stockdata/test/{code}.{name}.csv', index=False)

    print(f"训练集行数: {len(train_data)}, 测试集行数: {len(test_data)}")