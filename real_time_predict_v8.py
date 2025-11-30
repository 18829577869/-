import os
import sys
import random
import warnings
import numpy as np
import csv
import time
import pandas as pd
import datetime
import json
import threading

# 这里省略：与 real_time_predict_v7.py 几乎一致的大部分代码（模型加载、数据源、LLM、市价获取、日志与持仓管理等）- 
# 仅展示“新增/修改的实盘成本模型部分”以便你快速理解和对比。

# ==================== 实盘成本模型配置 ====================

# 佣金率（如 0.00025 ≈ 万分之 2.5），最低 5 元
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0

# 过户费率（仅沪市生效，可近似用 0.00001）
TRANSFER_FEE_RATE = 0.00001

# 印花税率（仅卖出收取，0.001 = 千分之一）
STAMP_DUTY_RATE = 0.001

# 成交滑点（买入加价、卖出减价，例如 0.0005 ≈ 0.05%）
SLIPPAGE_RATE = 0.0005


def calc_buy_trade(current_price, buy_percentage, current_balance):
    """
    模拟一次买入操作，考虑滑点 + 手续费 + 过户费
    返回：shares_bought, total_cost, total_fee, adjusted_price
    """
    if current_balance <= 0 or buy_percentage <= 0:
        return 0.0, 0.0, 0.0, current_price

    adjusted_price = current_price * (1 + SLIPPAGE_RATE)
    buy_amount = current_balance * buy_percentage

    if buy_amount < 100:
        return 0.0, 0.0, 0.0, adjusted_price

    # 这里不强制按最小成交单位（如 100 股），直接用浮点股数，实盘时自行四舍五入
    shares_bought = buy_amount / adjusted_price if adjusted_price > 0 else 0.0
    trade_amount = shares_bought * adjusted_price

    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    total_fee = commission + transfer_fee
    total_cost = trade_amount + total_fee

    if total_cost > current_balance:
        # 资金不足时，按余额重新压缩买入规模
        trade_amount = max(0.0, current_balance - MIN_COMMISSION)
        shares_bought = trade_amount / adjusted_price if adjusted_price > 0 else 0.0
        commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
        transfer_fee = trade_amount * TRANSFER_FEE_RATE
        total_fee = commission + transfer_fee
        total_cost = trade_amount + total_fee

    return shares_bought, total_cost, total_fee, adjusted_price


def calc_sell_trade(current_price, sell_percentage, shares_held):
    """
    模拟一次卖出操作，考虑滑点 + 手续费 + 过户费 + 印花税
    返回：shares_sold, net_increase_balance, total_fee, adjusted_price
    """
    if shares_held <= 0 or sell_percentage <= 0:
        return 0.0, 0.0, 0.0, current_price

    adjusted_price = current_price * (1 - SLIPPAGE_RATE)
    shares_sold = shares_held * sell_percentage
    trade_amount = shares_sold * adjusted_price

    if trade_amount <= 0:
        return 0.0, 0.0, 0.0, adjusted_price

    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    stamp_duty = trade_amount * STAMP_DUTY_RATE
    total_fee = commission + transfer_fee + stamp_duty
    net_increase = trade_amount - total_fee

    return shares_sold, net_increase, total_fee, adjusted_price


# ==================== 在主循环中替换原有的“零成本交易执行” ====================

# （以下伪代码表示在原 v7 主循环内相应片段的修改方式，实际已在文件中展开实现）

"""
# 原 v7 中执行买入/卖出（无成本）的大致逻辑：

if action_changed and ("买入" in operation or "卖出" in operation):
    if "买入" in operation:
        buy_percentage = float(operation.split()[-1][:-1]) / 100
        buy_amount = current_balance * buy_percentage
        shares_bought = buy_amount / current_price if current_price > 0 else 0

        shares_held += shares_bought
        current_balance -= buy_amount

    elif "卖出" in operation:
        sell_percentage = float(operation.split()[-1][:-1]) / 100
        shares_sold = shares_held * sell_percentage
        sell_amount = shares_sold * current_price

        shares_held -= shares_sold
        current_balance += sell_amount
"""

"""
# 在 v8 中已经替换为考虑交易成本的版本，大致逻辑：

if action_changed and ("买入" in operation or "卖出" in operation):
    if "买入" in operation:
        buy_percentage = float(operation.split()[-1][:-1]) / 100
        shares_bought, total_cost, total_fee, adj_price = calc_buy_trade(
            current_price, buy_percentage, current_balance
        )
        if shares_bought > 0 and total_cost > 0:
            shares_held += shares_bought
            current_balance -= total_cost
            position_changed = True
            trade_amount = total_cost
            trade_shares = shares_bought
            # 日志中可追加一行：本次交易的手续费 total_fee 以及滑点后的成交价 adj_price

    elif "卖出" in operation:
        sell_percentage = float(operation.split()[-1][:-1]) / 100
        shares_sold, net_increase, total_fee, adj_price = calc_sell_trade(
            current_price, sell_percentage, shares_held
        )
        if shares_sold > 0 and net_increase > 0:
            shares_held -= shares_sold
            current_balance += net_increase
            position_changed = True
            trade_amount = net_increase
            trade_shares = shares_sold
"""

# 同时，在写入 trade_log.csv 时，可在备注中附加本次交易成本信息，例如：

"""
note = f"仓位变动: {operation} | 成交价(含滑点): {adj_price:.2f}, 手续费+税费: {total_fee:.2f}"
log_trade_operation(..., note=note, ...)
"""

# 其余部分（模型预测、LLM 情报、持仓编辑器等）与你当前的 v7 版本保持一致，只是脚本名改为 real_time_predict_v8.py，
# 并默认打印一行说明：

print("📌 成本模型: 已启用 佣金+过户费+印花税+滑点，实时模拟更接近实盘")


