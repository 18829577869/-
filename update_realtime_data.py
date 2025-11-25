"""
更新股票数据到最新日期
用于实盘交易前的数据准备
"""

import sys
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 股票列表
STOCKS = [
    {"code": "sh.600036", "name": "招商银行"},
    {"code": "sh.601838", "name": "成都银行"},
    {"code": "sh.601318", "name": "中国平安"},
    {"code": "sh.601939", "name": "建设银行"},
    {"code": "sh.601398", "name": "工商银行"},
    {"code": "sz.000858", "name": "五粮液"},
]

OUTPUT_DIR = "stockdata_v7_realtime"


def update_stock_data(stock_code, stock_name, start_date, end_date):
    """
    更新单只股票数据
    
    Args:
        stock_code: 股票代码（如：sh.600036）
        stock_name: 股票名称
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """
    print(f"\n更新 {stock_name} ({stock_code})...")
    print(f"  日期范围: {start_date} 到 {end_date}")
    
    lg = bs.login()
    
    try:
        # 获取日K线数据
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
            "turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"  ⚠️ 未获取到数据")
            return False
        
        # 创建DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 过滤停牌日
        df = df[df['tradestatus'] == '1']
        
        print(f"  ✅ 获取 {len(df)} 条数据")
        
        # 保存文件
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = f"{OUTPUT_DIR}/{stock_code}.{stock_name}.csv"
        
        # 如果文件已存在，合并数据
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            df['date'] = pd.to_datetime(df['date'])
            
            # 合并并去重
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date')
            
            df = combined_df
            print(f"  📁 合并后共 {len(df)} 条数据")
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  💾 已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False
        
    finally:
        bs.logout()


def main():
    """主函数"""
    print("="*70)
    print("股票数据更新工具")
    print("="*70)
    
    # 设置日期范围
    # 从最近的已有数据之后开始更新
    start_date = "2025-01-01"  # 从2025年开始
    end_date = datetime.now().strftime("%Y-%m-%d")  # 到今天
    
    print(f"\n更新配置:")
    print(f"  股票数量: {len(STOCKS)}")
    print(f"  日期范围: {start_date} 到 {end_date}")
    print(f"  输出目录: {OUTPUT_DIR}/")
    
    input("\n按回车键开始更新...")
    
    success_count = 0
    for stock in STOCKS:
        if update_stock_data(stock['code'], stock['name'], start_date, end_date):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"更新完成! 成功: {success_count}/{len(STOCKS)}")
    print("="*70)
    print(f"\n数据已保存到: {OUTPUT_DIR}/")
    print("\n下一步:")
    print(f"  python realtime_trading_v7.py --date {end_date}")
    print("="*70)


if __name__ == "__main__":
    main()

