# get_stock_data_v3.py (优化：添加更多股票选项，自动处理上市日期查询错误)
import baostock as bs
import pandas as pd
import os

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# 股票列表：代码、名称、上市日期（start_date 为上市日期）
stocks = [
    {"code": "sh.600000", "name": "浦发银行", "start_date": "1999-11-10"},
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09"},
    {"code": "sz.002083", "name": "孚日股份", "start_date": "2006-11-24"},
    {"code": "sz.001389", "name": "广合科技", "start_date": "2024-04-02"},
    {"code": "sh.600418", "name": "江淮汽车", "start_date": "2001-08-24"}
    # 可添加更多股票，例如：
    # {"code": "sz.000001", "name": "平安银行", "start_date": "1991-04-03"},
]

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    print(f"查询 {code} ({name}), start_date: {start_date}")

    rs = bs.query_history_k_data_plus(code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
        start_date=start_date, end_date='2025-11-19',
        frequency="d", adjustflag="3")
    print("查询响应:", rs.error_code, rs.error_msg)

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    if len(result) == 0:
        print(f"警告: {code} 无数据，跳过")
        continue

    print(f"总行数: {len(result)}")
    print(result.head())
    print(result.tail())

    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values('date')

    train_data = result[result['date'] <= '2024-12-31']
    test_data = result[result['date'] > '2024-12-31']

    os.makedirs('stockdata/train', exist_ok=True)
    os.makedirs('stockdata/test', exist_ok=True)
    train_data.to_csv(f'stockdata/train/{code}.{name}.csv', index=False)
    test_data.to_csv(f'stockdata/test/{code}.{name}.csv', index=False)

    print(f"训练集行数: {len(train_data)}, 测试集行数: {len(test_data)}")

bs.logout()