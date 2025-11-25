import baostock as bs
import pandas as pd
import os

lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

rs = bs.query_history_k_data_plus("sh.600000",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
    start_date='1999-11-10', end_date='2025-11-19',  # 扩展到历史
    frequency="d", adjustflag="3")
print("查询响应:", rs.error_code, rs.error_msg)

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

print(f"总行数: {len(result)}")  # 检查数据量，应有 6000+ 行（约 26 年 x 250 交易日/年）
print(result.head())  # 预览
print(result.tail())  # 检查最后几行，确保到 2025

# 转换为 datetime 并排序（如果需要）
result['date'] = pd.to_datetime(result['date'])
result = result.sort_values('date')

# 划分训练/测试集（更新为近期）
train_data = result[result['date'] <= '2024-12-31']  # 历史作为训练
test_data = result[result['date'] > '2024-12-31']   # 2025 年作为测试

os.makedirs('stockdata/train', exist_ok=True)
os.makedirs('stockdata/test', exist_ok=True)
train_data.to_csv('stockdata/train/sh.600000.浦发银行.csv', index=False)
test_data.to_csv('stockdata/test/sh.600000.浦发银行.csv', index=False)

print(f"训练集行数: {len(train_data)}, 测试集行数: {len(test_data)}")

bs.logout()