"""
迁移交易日志文件格式

将旧格式的 trade_log.csv 迁移到新格式（添加建议价格字段）

使用方法:
    python migrate_trade_log.py
"""

import os
import pandas as pd
import shutil
from datetime import datetime

TRADE_LOG_FILE = "trade_log.csv"
BACKUP_SUFFIX = "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def migrate_trade_log():
    """迁移交易日志文件格式"""
    if not os.path.exists(TRADE_LOG_FILE):
        print(f"ℹ️  文件 {TRADE_LOG_FILE} 不存在，无需迁移")
        return
    
    try:
        # 读取旧文件
        df = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig')
        
        # 备份原文件
        backup_file = TRADE_LOG_FILE + BACKUP_SUFFIX
        shutil.copy2(TRADE_LOG_FILE, backup_file)
        print(f"✅ 已备份原文件: {backup_file}")
        
        # 检查是否需要迁移
        if '建议买入价格' in df.columns and '建议卖出价格' in df.columns:
            print(f"ℹ️  文件格式已是最新，无需迁移")
            return
        
        # 添加新字段
        if '建议买入价格' not in df.columns:
            df['建议买入价格'] = ''
        if '建议卖出价格' not in df.columns:
            df['建议卖出价格'] = ''
        if '预测数量' not in df.columns:
            # 尝试从旧字段迁移
            if '数量' in df.columns:
                df['预测数量'] = df['数量']
            else:
                df['预测数量'] = 0.0
        if '预测金额' not in df.columns:
            # 尝试从旧字段迁移
            if '金额' in df.columns:
                df['预测金额'] = df['金额']
            else:
                df['预测金额'] = 0.0
        if '当前价格' not in df.columns:
            # 尝试从旧字段迁移
            if '价格' in df.columns:
                df['当前价格'] = df['价格']
            else:
                df['当前价格'] = 0.0
        
        # 根据操作类型填充建议价格
        for idx, row in df.iterrows():
            if row['操作类型'] == '买入':
                if pd.isna(row['建议买入价格']) or row['建议买入价格'] == '':
                    df.at[idx, '建议买入价格'] = row['当前价格']
            elif row['操作类型'] == '卖出':
                if pd.isna(row['建议卖出价格']) or row['建议卖出价格'] == '':
                    df.at[idx, '建议卖出价格'] = row['当前价格']
        
        # 重新排列列顺序
        new_columns = [
            '时间戳', '日期', '时间', '股票代码', '操作类型', '操作比例',
            '当前价格', '建议买入价格', '建议卖出价格', '预测数量', '预测金额',
            '持仓数量', '可用资金', '总资产', '操作状态', '备注'
        ]
        
        # 确保所有列都存在
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''
        
        # 重新排列
        df = df[new_columns]
        
        # 保存新文件
        df.to_csv(TRADE_LOG_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 文件格式已更新: {TRADE_LOG_FILE}")
        print(f"   新增字段: 建议买入价格, 建议卖出价格, 预测数量, 预测金额")
        print(f"   备份文件: {backup_file}")
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print("=" * 70)
    print("交易日志文件格式迁移工具")
    print("=" * 70)
    print()
    migrate_trade_log()
    print()
    print("=" * 70)

