"""
V11 实时预测系统 - 全功能集成版
整合 V7、V9、V10 的所有功能：
1. V7功能：技术指标、多数据源、LLM解释、成本模型、PPO强化学习
2. V9功能：LSTM/GRU、注意力机制、动态参数优化、自动学习优化
3. V10功能：Transformer、多模态处理、实时可视化、全息动态模型

设计理念：多模型协同工作，智能融合决策
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import datetime
import time
import json
import threading

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==================== V7 成本模型配置 ====================

COMMISSION_RATE = 0.00025  # 佣金率
MIN_COMMISSION = 5.0  # 最低佣金
TRANSFER_FEE_RATE = 0.00001  # 过户费率
STAMP_DUTY_RATE = 0.001  # 印花税率（仅卖出）
SLIPPAGE_RATE = 0.0005  # 滑点率

def calc_buy_trade(current_price, buy_percentage, current_balance):
    """模拟买入操作，考虑滑点、手续费、过户费"""
    if current_balance <= 0 or buy_percentage <= 0:
        return 0.0, 0.0, 0.0, current_price
    
    adjusted_price = current_price * (1 + SLIPPAGE_RATE)
    buy_amount = current_balance * buy_percentage
    
    if buy_amount < 100:
        return 0.0, 0.0, 0.0, adjusted_price
    
    shares_bought = buy_amount / adjusted_price if adjusted_price > 0 else 0.0
    trade_amount = shares_bought * adjusted_price
    
    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    total_fee = commission + transfer_fee
    total_cost = trade_amount + total_fee
    
    if total_cost > current_balance:
        trade_amount = max(0.0, current_balance - MIN_COMMISSION)
        shares_bought = trade_amount / adjusted_price if adjusted_price > 0 else 0.0
        commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
        transfer_fee = trade_amount * TRANSFER_FEE_RATE
        total_fee = commission + transfer_fee
        total_cost = trade_amount + total_fee
    
    return shares_bought, total_cost, total_fee, adjusted_price

def calc_sell_trade(current_price, sell_percentage, shares_held):
    """模拟卖出操作，考虑滑点、手续费、过户费、印花税"""
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

# ==================== 导入模块 ====================

# V7模块：技术指标、多数据源、LLM解释
try:
    from technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    print("[警告] 技术指标模块不可用")

try:
    from multi_data_source_manager import MultiDataSourceManager
    MULTI_DATA_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_DATA_SOURCE_AVAILABLE = False
    print("[警告] 多数据源管理器不可用")

try:
    from llm_indicator_interpreter import LLMIndicatorInterpreter
    LLM_INTERPRETER_AVAILABLE = True
except ImportError:
    LLM_INTERPRETER_AVAILABLE = False
    print("[警告] LLM指标解释器不可用")

# V9模块：LSTM/GRU、动态参数优化
try:
    from lstm_gru_time_series import TimeSeriesProcessor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[警告] LSTM/GRU模块不可用")

try:
    from dynamic_parameter_optimizer import (
        DynamicParameterOptimizer, AutoLearningOptimizer, ParameterRange
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("[警告] 参数优化器模块不可用")

# V10模块：Transformer、多模态、可视化、全息模型
try:
    from transformer_model import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[警告] Transformer模块不可用")

try:
    from multimodal_data_processor import MultimodalDataProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("[警告] 多模态处理模块不可用")

try:
    from realtime_visualization import RealTimeVisualizer, WebVisualizationServer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[警告] 可视化模块不可用")

try:
    from holographic_dynamic_model import HolographicDynamicModel
    HOLOGRAPHIC_AVAILABLE = True
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False
    print("[警告] 全息动态模型不可用")

# 其他模块
# 抑制Gym的废弃警告（stable_baselines3内部使用gym）
import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*upgrade to Gymnasium.*')

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("[警告] PPO模型不可用")

try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[警告] LLM市场情报不可用")

# ==================== 工具函数 ====================

def convert_stock_code(code):
    """转换股票代码格式"""
    if '.' in code:
        market, num = code.split('.')
        return {
            'baostock': code,
            'tushare': f"{num}.{market.upper()}",
            'akshare': num,
            'market': 'sh' if market == 'sh' else 'sz'
        }
    else:
        if code.startswith('6'):
            return {
                'baostock': f"sh.{code}",
                'tushare': f"{code}.SH",
                'akshare': code,
                'market': 'sh'
            }
        else:
            return {
                'baostock': f"sz.{code}",
                'tushare': f"{code}.SZ",
                'akshare': code,
                'market': 'sz'
            }

def map_action_to_operation(action):
    """将动作映射到具体操作"""
    actions = {
        0: "卖出 100%",
        1: "卖出 50%",
        2: "卖出 25%",
        3: "持有",
        4: "买入 25%",
        5: "买入 50%",
        6: "买入 100%"
    }
    return actions.get(action, "未知动作")

def fetch_akshare_5min(code_info, days=7):
    """使用 AkShare 获取5分钟K线数据"""
    try:
        import akshare as ak
        symbol = code_info['akshare']
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="5",
                adjust="qfq",
                start_date=start_date,
                end_date=end_date
            )
            if df is None or len(df) == 0:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if df is not None and len(df) > 0:
                    df = df.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['time'] = df['date'] + '15000000'
                    return df[['date', 'time', 'close', 'volume']]
                return None
            
            column_mapping = {
                '时间': 'time',
                '收盘': 'close',
                '成交量': 'volume',
                '日期': 'date'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d%H%M%S')
                df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            return df[['date', 'time', 'close', 'volume']]
        except Exception as e:
            return None
    except ImportError:
        return None
    except Exception as e:
        return None

def init_trade_log():
    """初始化交易日志文件"""
    import csv
    TRADE_LOG_FILE = "trade_log.csv"
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '时间戳', '日期', '时间', '股票代码', '操作类型', '操作比例', 
                '当前价格', '建议买入价格', '建议卖出价格', '预测数量', '预测金额', 
                '持仓数量', '可用资金', '总资产', '操作状态', '备注'
            ])

def save_portfolio_state(stock_code, shares_held, current_balance, last_price, initial_balance,
                        actual_buy_price=None, actual_sell_price=None, cost_price=None):
    """保存持仓状态"""
    try:
        # 使用实际买入价作为成本价（如果有），否则使用last_price
        if cost_price is None:
            cost_price = actual_buy_price if actual_buy_price and actual_buy_price > 0 else last_price
        
        state = {
            'stock_code': stock_code,
            'shares_held': float(shares_held),
            'current_balance': float(current_balance),
            'last_price': float(last_price),
            'initial_balance': float(initial_balance),
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_assets': float(current_balance + shares_held * last_price)
        }
        
        # 添加可选字段
        if actual_buy_price and actual_buy_price > 0:
            state['actual_buy_price'] = float(actual_buy_price)
            # 如果未指定成本价，使用实际买入价作为成本价
            if cost_price is None or cost_price <= 0:
                state['cost_price'] = float(actual_buy_price)
        
        if cost_price and cost_price > 0:
            state['cost_price'] = float(cost_price)
            
        if actual_sell_price and actual_sell_price > 0:
            state['actual_sell_price'] = float(actual_sell_price)
        
        with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def load_portfolio_state():
    """加载持仓状态"""
    try:
        if not os.path.exists(PORTFOLIO_STATE_FILE):
            return None
        with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def log_trade_operation(stock_code, operation, current_price, shares_held, 
                       current_balance, total_assets, status='预测', note=''):
    """记录交易操作"""
    try:
        import csv
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        op_type = "买入" if "买入" in operation else "卖出" if "卖出" in operation else "持有"
        op_ratio = "0%" if "持有" in operation else operation.split()[-1] if "%" in operation else "0%"
        
        with open(TRADE_LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, date, time_str, stock_code, op_type, op_ratio,
                f"{current_price:.2f}", "", "", "", "",
                f"{shares_held:.2f}", f"{current_balance:.2f}", f"{total_assets:.2f}",
                status, note
            ])
        return True
    except:
        return False

# ==================== 配置参数 ====================

# 基础配置
MODEL_PATH = "ppo_stock_v7.zip"
STOCK_CODE = 'sh.600730'
LLM_PROVIDER = "deepseek"
ENABLE_LLM = True
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d')

# V7配置
TECHNICAL_INDICATOR_CONFIG = {
    'kdj_period': 9,
    'kdj_slow_period': 3,
    'kdj_fast_period': 3,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'obv_smooth_period': 20,
    'ma_periods': [5, 10, 20, 60]
}

# V9配置
ENABLE_LSTM_PREDICTION = True
ENABLE_DYNAMIC_OPTIMIZATION = True
LSTM_MODEL_TYPE = 'lstm_attention'
LSTM_SEQ_LENGTH = 60
LSTM_HIDDEN_SIZE = 64

# V10配置
ENABLE_TRANSFORMER = True
ENABLE_MULTIMODAL = True
ENABLE_VISUALIZATION = True
ENABLE_HOLOGRAPHIC = True

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_MAX_SEQ_LEN = 100

# V11改进配置：滑动窗口归一化
USE_SLIDING_WINDOW_NORMALIZE = True  # 使用滑动窗口归一化，避免全局偏低
SLIDING_WINDOW_SIZE = 500  # 滑动窗口大小（使用最近N个数据点）

# V11改进配置：动态权重调整
ENABLE_DYNAMIC_WEIGHTS = True  # 启用动态权重调整
WEIGHT_ADAPTATION_RATE = 0.1  # 权重调整速率
WEIGHT_MIN = 0.05  # 最小权重
WEIGHT_MAX = 0.6  # 最大权重

# V11改进配置：多模态真实数据源
USE_REAL_NEWS_SOURCE = True  # 使用真实新闻源（LLM市场情报）
FALLBACK_TO_SAMPLE_TEXTS = True  # 如果获取失败，回退到样本文本

# V11改进配置：量化回测
ENABLE_BACKTEST = True  # 启用回测功能
BACKTEST_METRICS = ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy']  # 回测指标

VISUALIZATION_PORT = 8082  # V11使用8082端口
VISUALIZATION_OUTPUT_DIR = "visualization_output"

HOLOGRAPHIC_MEMORY_SIZE = 1000

# V11持仓编辑器配置
ENABLE_WEB_EDITOR = True          # 是否启用网页持仓编辑
WEB_EDITOR_PORT = 5001           # 本地网页端口
WEB_EDITOR_HOST = "127.0.0.1"    # 仅本机访问

# V11智能融合配置
ENABLE_MULTI_MODEL_FUSION = True  # 启用多模型融合
MODEL_WEIGHTS = {
    'ppo': 0.4,          # PPO强化学习模型权重
    'lstm': 0.2,         # LSTM/GRU模型权重
    'transformer': 0.2,  # Transformer模型权重
    'holographic': 0.2   # 全息动态模型权重
}

# 文件路径
TRADE_LOG_FILE = "trade_log.csv"
PORTFOLIO_STATE_FILE = "portfolio_state.json"

# V7持仓编辑器配置
ENABLE_WEB_EDITOR = True          # 是否启用网页持仓编辑
WEB_EDITOR_PORT = 5001           # 本地网页端口（与可视化服务器分离）
WEB_EDITOR_HOST = "127.0.0.1"    # 仅本机访问

# ==================== 版本标识 ====================

print("\n" + "=" * 70)
print("V11 实时预测系统 - 全功能集成版")
print("=" * 70)
print("📌 整合功能:")
print("   V7: 技术指标、多数据源、LLM解释、成本模型、PPO强化学习")
print("   V9: LSTM/GRU、注意力机制、动态参数优化、自动学习优化")
print("   V10: Transformer、多模态处理、实时可视化、全息动态模型")
print("=" * 70)
print("⚠️  版本标识: 这是 V11 版本，整合所有功能！")
print("=" * 70 + "\n")

# ==================== 初始化模块 ====================

# V7模块初始化
tech_indicators = None
if TECHNICAL_INDICATORS_AVAILABLE:
    try:
        tech_indicators = TechnicalIndicators(**TECHNICAL_INDICATOR_CONFIG)
        print("✅ V7技术指标计算器初始化成功")
    except Exception as e:
        print(f"⚠️  技术指标计算器初始化失败: {e}")

multi_source_manager = None
if MULTI_DATA_SOURCE_AVAILABLE:
    try:
        multi_source_manager = MultiDataSourceManager(stock_code=STOCK_CODE)
        print("✅ V7多数据源管理器初始化成功")
    except Exception as e:
        print(f"⚠️  多数据源管理器初始化失败: {e}")

llm_interpreter = None
llm_agent = None
if LLM_AVAILABLE and ENABLE_LLM:
    try:
        os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY
        llm_agent = MarketIntelligenceAgent(
            provider=LLM_PROVIDER,
            api_key=DEEPSEEK_API_KEY,
            enable_cache=True
        )
        print("✅ LLM市场情报代理初始化成功")
        
        if LLM_INTERPRETER_AVAILABLE:
            llm_interpreter = LLMIndicatorInterpreter(
                llm_agent=llm_agent,
                enable_cache=True
            )
            print("✅ V7 LLM指标解释器初始化成功")
    except Exception as e:
        print(f"⚠️  LLM初始化失败: {e}")

# V9模块初始化
lstm_processor = None
if LSTM_AVAILABLE and ENABLE_LSTM_PREDICTION:
    try:
        lstm_processor = TimeSeriesProcessor(
            model_type=LSTM_MODEL_TYPE,
            seq_length=LSTM_SEQ_LENGTH,
            input_size=1,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            use_bidirectional=False,
            use_gpu=False
        )
        print(f"✅ V9 LSTM/GRU处理器初始化成功 (类型: {LSTM_MODEL_TYPE})")
    except Exception as e:
        print(f"⚠️  LSTM/GRU处理器初始化失败: {e}")

dynamic_optimizer = None
auto_learner = None
if OPTIMIZER_AVAILABLE and ENABLE_DYNAMIC_OPTIMIZATION:
    try:
        # 这里需要根据实际需求定义参数范围
        parameter_ranges = {
            'kdj_period': ParameterRange(5, 14, param_type='integer'),
            'rsi_period': ParameterRange(10, 20, param_type='integer'),
        }
        dynamic_optimizer = DynamicParameterOptimizer(
            parameter_ranges=parameter_ranges,
            optimization_method='bayesian',
            adaptation_rate=0.1,
            exploration_rate=0.2,
            performance_window=100
        )
        auto_learner = AutoLearningOptimizer(
            parameter_optimizer=dynamic_optimizer,
            learning_rate=0.01,
            momentum=0.9,
            decay_rate=0.99
        )
        print("✅ V9动态参数优化器初始化成功")
    except Exception as e:
        print(f"⚠️  参数优化器初始化失败: {e}")

# V10模块初始化
transformer_model = None
if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER:
    try:
        transformer_model = TransformerPredictor(
            input_size=1,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_encoder_layers=TRANSFORMER_NUM_LAYERS,
            num_decoder_layers=TRANSFORMER_NUM_LAYERS,
            max_seq_len=TRANSFORMER_MAX_SEQ_LEN
        )
        print("✅ V10 Transformer模型初始化成功")
    except Exception as e:
        print(f"⚠️  Transformer模型初始化失败: {e}")

multimodal_processor = None
if MULTIMODAL_AVAILABLE and ENABLE_MULTIMODAL:
    try:
        multimodal_processor = MultimodalDataProcessor(
            text_max_length=512,
            use_bert=False,
            fusion_method='attention'
        )
        print("✅ V10多模态处理器初始化成功")
    except Exception as e:
        print(f"⚠️  多模态处理器初始化失败: {e}")

visualizer = None
web_visualization = None
if VISUALIZATION_AVAILABLE and ENABLE_VISUALIZATION:
    try:
        visualizer = RealTimeVisualizer()
        print("✅ V10实时可视化器初始化成功")
        
        try:
            web_visualization = WebVisualizationServer(visualizer, port=VISUALIZATION_PORT)
            web_visualization.start()
            print(f"✅ V10 Web可视化服务器启动成功 (端口: {VISUALIZATION_PORT})")
        except Exception as e:
            print(f"⚠️  Web可视化服务器启动失败: {e}")
    except Exception as e:
        print(f"⚠️  可视化器初始化失败: {e}")

holographic_model = None
if HOLOGRAPHIC_AVAILABLE and ENABLE_HOLOGRAPHIC:
    try:
        holographic_model = HolographicDynamicModel(
            memory_size=HOLOGRAPHIC_MEMORY_SIZE,
            enable_text_analysis=True,
            enable_memory=True
        )
        print("✅ V10全息动态模型初始化成功")
    except Exception as e:
        print(f"⚠️  全息动态模型初始化失败: {e}")

# PPO模型初始化
ppo_model = None
if PPO_AVAILABLE:
    try:
        if not os.path.exists(MODEL_PATH):
            possible_models = ["ppo_stock_v7.zip", "models_v7/best/best_model.zip"]
            for model_file in possible_models:
                if os.path.exists(model_file):
                    MODEL_PATH = model_file
                    break
        
        ppo_model = PPO.load(MODEL_PATH)
        print(f"✅ PPO模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  PPO模型加载失败: {e}")

print("=" * 70)
print()

# 初始化交易日志
try:
    init_trade_log()
except:
    pass

# ==================== V7持仓编辑器 ====================

# 检查Flask是否可用于持仓编辑器
try:
    from flask import Flask, request, render_template_string
    FLASK_EDITOR_AVAILABLE = True
except ImportError:
    FLASK_EDITOR_AVAILABLE = False

portfolio_editor_app = None
portfolio_state_mtime = os.path.getmtime(PORTFOLIO_STATE_FILE) if os.path.exists(PORTFOLIO_STATE_FILE) else None

def get_current_market_price(stock_code):
    """获取当前市场价格"""
    try:
        code_info = convert_stock_code(stock_code)
        df = fetch_akshare_5min(code_info, days=1)
        if df is not None and len(df) > 0:
            df = df.sort_values('time')
            current_price = float(df['close'].iloc[-1])
            return current_price
    except Exception as e:
        pass
    return None

def create_portfolio_web_app():
    """创建持仓编辑器Web应用"""
    global portfolio_editor_app
    if not FLASK_EDITOR_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    # 禁用Flask的访问日志，避免干扰其他输出
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # 只显示错误，不显示访问日志
    
    TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>持仓编辑器 - V11 实时预测系统</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Helvetica Neue",Arial,"Hiragino Sans GB","Microsoft YaHei",sans-serif;
           background:#f5f5f5; margin:0; padding:0; }
    .container { max-width: 640px; margin: 40px auto; background:#fff; padding:24px 32px; border-radius:12px;
                 box-shadow:0 8px 24px rgba(0,0,0,0.08); }
    h1 { font-size:22px; margin-bottom:8px; }
    p.desc { color:#666; font-size:13px; margin-top:0; margin-bottom:16px;}
    label { display:block; margin-top:14px; font-weight:600; font-size:14px;}
    input[type="text"], input[type="number"] {
      width:100%; padding:8px 10px; margin-top:6px; box-sizing:border-box;
      border:1px solid #d0d7de; border-radius:6px; font-size:14px;
    }
    input[readonly] { background:#f3f4f6; color:#555; }
    .row { display:flex; gap:12px; }
    .row > div { flex:1; }
    button {
      margin-top:20px; width:100%; padding:10px 16px; border:none; border-radius:20px;
      background:#0078d4; color:white; font-size:15px; font-weight:600; cursor:pointer;
    }
    button:hover { background:#005fa3; }
    .status { margin-top:12px; font-size:13px; color:#0078d4;}
    .pnl-block { margin-top:20px; padding:14px 16px; border-radius:10px; background:#f8f9fa; border:1px solid #e1e4e8;}
    .pnl-block h3 { font-size:15px; margin:0 0 10px 0; color:#24292e;}
    .pnl-row { display:flex; justify-content:space-between; margin:8px 0; font-size:14px;}
    .pnl-label { color:#586069; font-weight:500;}
    .pnl-value { color:#24292e; font-weight:600;}
    .pnl-positive { color:#28a745;}
    .pnl-negative { color:#dc3545;}
    .footer { margin-top:24px; font-size:12px; color:#999; text-align:center;}
    .price-update { font-size:12px; color:#28a745; margin-top:4px;}
    .price-update.error { color:#dc3545;}
    .auto-refresh { font-size:11px; color:#666; margin-top:8px;}
  </style>
  <script>
    let autoRefreshInterval = null;
    
    function recalculateBalance() {
      // 重新计算可用资金：初始资金 - 实际买入价 × 持仓数量
      const sharesHeldInput = document.querySelector('input[name="shares_held"]');
      const actualBuyPriceInput = document.querySelector('input[name="actual_buy_price"]');
      const initialBalanceInput = document.querySelector('input[name="initial_balance"]');
      const currentBalanceInput = document.querySelector('input[name="current_balance"]');
      
      if (!sharesHeldInput || !initialBalanceInput || !currentBalanceInput) {
        return;
      }
      
      const sharesHeld = parseFloat(sharesHeldInput.value) || 0;
      const initialBalance = parseFloat(initialBalanceInput.value) || 0;
      const actualBuyPrice = actualBuyPriceInput ? (parseFloat(actualBuyPriceInput.value) || 0) : 0;
      
      let newBalance = 0;
      if (sharesHeld > 0) {
        if (actualBuyPrice > 0) {
          // 使用实际买入价计算
          const positionCost = sharesHeld * actualBuyPrice;
          newBalance = Math.max(0.0, initialBalance - positionCost);
        } else {
          // 如果没有实际买入价，保持当前值
          newBalance = parseFloat(currentBalanceInput.value) || 0;
        }
      } else {
        // 没有持仓，可用资金等于初始资金
        newBalance = initialBalance;
      }
      
      // 更新可用资金字段
      currentBalanceInput.value = newBalance.toFixed(2);
    }
    
    function recalculateStats() {
      // 重新计算持仓统计
      const sharesHeldInput = document.querySelector('input[name="shares_held"]');
      const lastPriceInput = document.querySelector('input[name="last_price"]');
      const currentBalanceInput = document.querySelector('input[name="current_balance"]');
      const initialBalanceInput = document.querySelector('input[name="initial_balance"]');
      const costPriceInput = document.querySelector('input[name="cost_price"]');
      
      if (!sharesHeldInput || !lastPriceInput || !currentBalanceInput || !initialBalanceInput) {
        return; // 如果元素不存在，退出
      }
      
      // 先重新计算可用资金
      recalculateBalance();
      
      const sharesHeld = parseFloat(sharesHeldInput.value) || 0;
      const lastPrice = parseFloat(lastPriceInput.value) || 0;
      const currentBalance = parseFloat(currentBalanceInput.value) || 0;
      const initialBalance = parseFloat(initialBalanceInput.value) || 0;
      const costPrice = costPriceInput ? (parseFloat(costPriceInput.value) || 0) : 0;
      
      // 计算持仓市值
      const positionValue = sharesHeld * lastPrice;
      const totalAssets = currentBalance + positionValue;
      const cumulativePnl = totalAssets - initialBalance;
      
      // 更新显示 - 使用更可靠的方式查找元素
      const pnlRows = document.querySelectorAll('.pnl-row');
      if (pnlRows.length >= 5) {
        // 持仓市值 (索引1)
        const positionValueEl = pnlRows[1].querySelector('.pnl-value');
        if (positionValueEl) {
          positionValueEl.textContent = positionValue.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
        }
        
        // 总资产 (索引3)
        const totalAssetsEl = pnlRows[3].querySelector('.pnl-value');
        if (totalAssetsEl) {
          totalAssetsEl.textContent = totalAssets.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
        }
        
        // 盈亏 (索引4)
        const pnlEl = pnlRows[4].querySelector('.pnl-value');
        if (pnlEl) {
          const pnlSign = cumulativePnl >= 0 ? '+' : '';
          let pnlText = pnlSign + cumulativePnl.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
          
          // 如果有成本价，计算基于成本价的盈亏
          if (costPrice > 0 && sharesHeld > 0) {
            const costBasedPnl = (lastPrice - costPrice) * sharesHeld;
            pnlText += ` (按成本价 ${costPrice.toFixed(2)} 计算: ${costBasedPnl >= 0 ? '+' : ''}${costBasedPnl.toFixed(2)} 元)`;
          }
          
          pnlEl.textContent = pnlText;
          pnlEl.className = 'pnl-value ' + (cumulativePnl > 0 ? 'pnl-positive' : cumulativePnl < 0 ? 'pnl-negative' : '');
        }
      }
    }
    
    function updateCurrentPrice() {
      fetch('/api/current_price')
        .then(response => response.json())
        .then(data => {
          if (data.success && data.price > 0) {
            const priceInput = document.querySelector('input[name="last_price"]');
            const oldPrice = parseFloat(priceInput.value) || 0;
            const newPrice = data.price;
            
            if (Math.abs(newPrice - oldPrice) > 0.001) {
              priceInput.value = newPrice.toFixed(4);
              
              // 重新计算统计数据
              recalculateStats();
              
              // 显示更新提示
              const updateMsg = document.getElementById('price-update-msg');
              if (updateMsg) {
                const diff = newPrice - oldPrice;
                const diffPct = oldPrice > 0 ? ((diff / oldPrice) * 100).toFixed(2) : 0;
                const sign = diff >= 0 ? '+' : '';
                updateMsg.textContent = `✓ 价格已更新: ${newPrice.toFixed(2)} (${sign}${diff.toFixed(2)}, ${sign}${diffPct}%)`;
                updateMsg.className = 'price-update';
                setTimeout(() => {
                  updateMsg.textContent = '';
                }, 5000);
              }
            }
          }
        })
        .catch(error => {
          const updateMsg = document.getElementById('price-update-msg');
          if (updateMsg) {
            updateMsg.textContent = '⚠ 价格更新失败';
            updateMsg.className = 'price-update error';
          }
        });
    }
    
    function startAutoRefresh() {
      if (autoRefreshInterval) clearInterval(autoRefreshInterval);
      // 每30秒自动更新一次价格
      autoRefreshInterval = setInterval(updateCurrentPrice, 30000);
      // 立即更新一次
      updateCurrentPrice();
    }
    
    function stopAutoRefresh() {
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }
    }
    
    // 页面加载完成后启动自动刷新
    window.addEventListener('DOMContentLoaded', function() {
      startAutoRefresh();
    });
    
    // 页面卸载时停止自动刷新
    window.addEventListener('beforeunload', function() {
      stopAutoRefresh();
    });
    
    // 监听价格输入框变化，自动重新计算盈亏
    document.addEventListener('DOMContentLoaded', function() {
      const priceInput = document.querySelector('input[name="last_price"]');
      if (priceInput) {
        priceInput.addEventListener('input', function() {
          // 延迟一下，让其他字段也更新
          setTimeout(function() {
            recalculateStats();
          }, 100);
        });
      }
      
      // 监听其他相关字段的变化
      ['shares_held', 'current_balance', 'initial_balance', 'cost_price', 'actual_buy_price'].forEach(fieldName => {
        const input = document.querySelector('input[name="' + fieldName + '"]');
        if (input) {
          input.addEventListener('input', function() {
            setTimeout(function() {
              recalculateStats();
            }, 100);
          });
        }
      });
    });
  </script>
</head>
<body>
  <div class="container">
    <h1>持仓编辑器（实时同步）- V11</h1>
    <p class="desc">修改后点击"保存持仓"，<strong>正在运行的 real_time_predict_v11.py 会自动读取最新持仓</strong>，无需停止脚本。</p>
    <form method="post">
      <label>股票代码</label>
      <input type="text" name="stock_code" value="{{ stock_code }}" readonly>

      <div class="row">
        <div>
          <label>持仓数量（股）</label>
          <input type="number" step="1" min="0" name="shares_held" value="{{ shares_held }}">
        </div>
        <div>
          <label>可用资金（元）</label>
          <input type="number" step="0.01" name="current_balance" value="{{ current_balance }}">
        </div>
      </div>

      <div class="row">
        <div>
          <label>最近成交价（元）</label>
          <input type="number" step="0.0001" name="last_price" value="{{ last_price }}" id="last_price_input">
          <div id="price-update-msg" class="price-update"></div>
          <div class="auto-refresh">🔄 价格每30秒自动更新</div>
        </div>
        <div>
          <label>初始资金（元）</label>
          <input type="number" step="0.01" name="initial_balance" value="{{ initial_balance }}">
        </div>
      </div>

      <div class="row">
        <div>
          <label>实际买入价（元）</label>
          <input type="number" step="0.0001" name="actual_buy_price" value="{{ actual_buy_price }}" placeholder="输入实际买入价格">
        </div>
        <div>
          <label>实际卖出价（元）</label>
          <input type="number" step="0.0001" name="actual_sell_price" value="{{ actual_sell_price }}" placeholder="输入实际卖出价格">
        </div>
      </div>

      <div class="row">
        <div>
          <label>成本价（元）</label>
          <input type="number" step="0.0001" name="cost_price" value="{{ cost_price }}" placeholder="持仓成本价">
        </div>
        <div>
          <label style="color:#666; font-size:12px;">💡 提示：成本价用于计算盈亏，如未填写则使用实际买入价</label>
        </div>
      </div>

      <button type="submit">💾 保存持仓</button>
    </form>
    <div class="status">{{ msg }}</div>

    <div class="pnl-block">
      <h3>📊 持仓统计</h3>
      <div class="pnl-row">
        <span class="pnl-label">初始资金：</span>
        <span class="pnl-value">{{ initial_balance_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">持仓市值：</span>
        <span class="pnl-value">{{ position_value_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">可用资金：</span>
        <span class="pnl-value">{{ current_balance_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">总资产：</span>
        <span class="pnl-value">{{ total_assets_display }} 元</span>
      </div>
      <div class="pnl-row" style="margin-top:12px; padding-top:12px; border-top:1px solid #e1e4e8;">
        <span class="pnl-label">盈亏：</span>
        <span class="pnl-value {{ pnl_class }}">{{ cumulative_pnl_display }}</span>
      </div>
    </div>

    <div class="footer">
      打开方式：在浏览器中访问 http://{{ host }}:{{ port }}<br>
      V11系统：可视化 http://127.0.0.1:8082 | 持仓编辑 http://127.0.0.1:5001
    </div>
  </div>
</body>
</html>
"""
    
    @app.route("/api/current_price")
    def api_current_price():
        """API接口：获取当前市场价格"""
        from flask import jsonify
        try:
            state = load_portfolio_state()
            stock_code = state.get("stock_code", STOCK_CODE) if state else STOCK_CODE
            current_price = get_current_market_price(stock_code)
            if current_price:
                # 更新portfolio_state.json中的last_price
                if state:
                    state['last_price'] = current_price
                    state['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                return jsonify({"success": True, "price": current_price, "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            else:
                # 如果获取失败，返回文件中的价格
                if state:
                    return jsonify({"success": True, "price": state.get("last_price", 0.0), "cached": True})
                return jsonify({"success": False, "error": "无法获取价格"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        msg = ""
        state = load_portfolio_state()
        
        # 尝试获取实时价格
        realtime_price = None
        try:
            stock_code = state.get("stock_code", STOCK_CODE) if state else STOCK_CODE
            realtime_price = get_current_market_price(stock_code)
            if realtime_price and state:
                # 更新state中的last_price
                state['last_price'] = realtime_price
        except:
            pass
        
        data = {
            "stock_code": STOCK_CODE,
            "shares_held": 0.0,
            "current_balance": 20000.0,
            "last_price": 0.0,
            "initial_balance": 20000.0,
            "actual_buy_price": "",
            "actual_sell_price": "",
            "cost_price": "",
        }
        if state:
            # 如果获取到实时价格，优先使用实时价格
            last_price = realtime_price if realtime_price else state.get("last_price", 0.0)
            shares_held = int(state.get("shares_held", 0.0))
            initial_balance = state.get("initial_balance", 20000.0)
            actual_buy_price = state.get("actual_buy_price")
            
            # 重新计算可用资金：初始资金 - 实际买入价 × 持仓数量
            if shares_held > 0 and actual_buy_price and actual_buy_price > 0:
                position_cost = shares_held * actual_buy_price
                current_balance = max(0.0, initial_balance - position_cost)
            elif shares_held > 0 and last_price > 0:
                # 如果没有实际买入价，使用最近成交价作为参考
                position_cost = shares_held * last_price
                current_balance = max(0.0, initial_balance - position_cost)
            elif shares_held <= 0:
                # 没有持仓，可用资金等于初始资金
                current_balance = initial_balance
            else:
                current_balance = state.get("current_balance", 20000.0)
            
            data.update({
                "stock_code": state.get("stock_code", STOCK_CODE),
                "shares_held": shares_held,
                "current_balance": current_balance,
                "last_price": last_price,
                "initial_balance": initial_balance,
                "actual_buy_price": str(actual_buy_price) if actual_buy_price else "",
                "actual_sell_price": state.get("actual_sell_price", "") or "",
                "cost_price": state.get("cost_price", "") or "",
            })

        if request.method == "POST":
            try:
                stock_code = request.form.get("stock_code", STOCK_CODE).strip()
                shares_held = int(float(request.form.get("shares_held") or 0))
                current_balance = float(request.form.get("current_balance") or 0)
                last_price = float(request.form.get("last_price") or 0)
                initial_balance = float(request.form.get("initial_balance") or 0)
                
                # 获取实际买入价、卖出价和成本价
                actual_buy_price_str = request.form.get("actual_buy_price", "").strip()
                actual_sell_price_str = request.form.get("actual_sell_price", "").strip()
                cost_price_str = request.form.get("cost_price", "").strip()
                
                actual_buy_price = float(actual_buy_price_str) if actual_buy_price_str else None
                actual_sell_price = float(actual_sell_price_str) if actual_sell_price_str else None
                cost_price = float(cost_price_str) if cost_price_str else None
                
                # 如果未填写成本价，使用实际买入价
                if cost_price is None and actual_buy_price and actual_buy_price > 0:
                    cost_price = actual_buy_price
                elif cost_price is None and last_price > 0:
                    cost_price = last_price

                # 重新计算可用资金：初始资金 - 实际买入价 × 持仓数量
                if shares_held > 0:
                    if actual_buy_price and actual_buy_price > 0:
                        # 使用实际买入价计算
                        position_cost = shares_held * actual_buy_price
                        current_balance = max(0.0, initial_balance - position_cost)
                    elif last_price > 0:
                        # 如果没有实际买入价，使用最近成交价作为参考
                        position_cost = shares_held * last_price
                        current_balance = max(0.0, initial_balance - position_cost)
                    else:
                        # 如果都没有，保持原有值
                        pass
                else:
                    # 没有持仓，可用资金等于初始资金
                    current_balance = initial_balance if initial_balance > 0 else current_balance

                save_portfolio_state(
                    stock_code, shares_held, current_balance, last_price, initial_balance,
                    actual_buy_price=actual_buy_price,
                    actual_sell_price=actual_sell_price,
                    cost_price=cost_price
                )
                msg = f"✅ 已保存持仓状态，V11系统将在下一轮自动同步。可用资金：{current_balance:.2f} 元"
                if cost_price:
                    msg += f"，成本价：{cost_price:.2f} 元"
                
                data.update({
                    "stock_code": stock_code,
                    "shares_held": shares_held,
                    "current_balance": current_balance,
                    "last_price": last_price,
                    "initial_balance": initial_balance,
                    "actual_buy_price": actual_buy_price_str if actual_buy_price_str else "",
                    "actual_sell_price": actual_sell_price_str if actual_sell_price_str else "",
                    "cost_price": f"{cost_price:.4f}" if cost_price else "",
                })
            except Exception as e:
                msg = f"❌ 保存失败: {e}"

        # 计算统计数据
        shares_held_val = float(data.get("shares_held", 0))
        last_price_val = float(data.get("last_price", 0))
        current_balance_val = float(data.get("current_balance", 0))
        initial_balance_val = float(data.get("initial_balance", 0))
        
        position_value = shares_held_val * last_price_val
        total_assets = current_balance_val + position_value
        cumulative_pnl = total_assets - initial_balance_val
        pnl_percentage = (cumulative_pnl / initial_balance_val * 100) if initial_balance_val > 0 else 0.0
        
        pnl_class = "pnl-positive" if cumulative_pnl > 0 else "pnl-negative" if cumulative_pnl < 0 else ""
        pnl_sign = "+" if cumulative_pnl > 0 else ""

        # 计算基于成本价的盈亏（如果有成本价）
        cost_price_str = data.get("cost_price", "")
        if cost_price_str:
            try:
                cost_price_val = float(cost_price_str)
                if cost_price_val > 0:
                    cost_based_pnl = (last_price_val - cost_price_val) * shares_held_val
                    pnl_info = f"（按成本价 {cost_price_val:.2f} 计算：{cost_based_pnl:+.2f} 元）"
                else:
                    pnl_info = ""
            except:
                pnl_info = ""
        else:
            pnl_info = ""
        
        return render_template_string(
            TEMPLATE.replace("{{ host }}", WEB_EDITOR_HOST).replace("{{ port }}", str(WEB_EDITOR_PORT))
                    .replace("{{ stock_code }}", str(data["stock_code"]))
                    .replace("{{ shares_held }}", str(int(data["shares_held"])))
                    .replace("{{ current_balance }}", str(data["current_balance"]))
                    .replace("{{ last_price }}", str(data["last_price"]))
                    .replace("{{ initial_balance }}", str(data["initial_balance"]))
                    .replace("{{ actual_buy_price }}", str(data.get("actual_buy_price", "")))
                    .replace("{{ actual_sell_price }}", str(data.get("actual_sell_price", "")))
                    .replace("{{ cost_price }}", str(data.get("cost_price", "")))
                    .replace("{{ msg }}", msg)
                    .replace("{{ initial_balance_display }}", f"{initial_balance_val:,.2f}")
                    .replace("{{ position_value_display }}", f"{position_value:,.2f}")
                    .replace("{{ current_balance_display }}", f"{current_balance_val:,.2f}")
                    .replace("{{ total_assets_display }}", f"{total_assets:,.2f}")
                    .replace("{{ cumulative_pnl_display }}", f"{pnl_sign}{cumulative_pnl:,.2f} 元 {pnl_info}")
                    .replace("{{ pnl_class }}", pnl_class)
        )
    
    portfolio_editor_app = app
    return app

def start_portfolio_web_editor():
    """在后台线程启动持仓编辑器"""
    if not FLASK_EDITOR_AVAILABLE or not ENABLE_WEB_EDITOR:
        return

    app = create_portfolio_web_app()
    if app is None:
        return

    def run():
        try:
            app.run(host=WEB_EDITOR_HOST, port=WEB_EDITOR_PORT, debug=False, use_reloader=False)
        except Exception as e:
            print(f"⚠️  持仓编辑器启动失败: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"✅ V7持仓编辑器已启动: http://{WEB_EDITOR_HOST}:{WEB_EDITOR_PORT}")
    print(f"   💡 可在V11运行时实时修改持仓信息，无需停止脚本")

# 启动持仓编辑器
if ENABLE_WEB_EDITOR:
    try:
        start_portfolio_web_editor()
        time.sleep(0.5)  # 等待服务器启动
    except Exception as e:
        print(f"⚠️  持仓编辑器启动失败: {e}")

def refresh_portfolio_from_file_if_changed(current_balance, shares_held, last_price, initial_balance):
    """
    如果 portfolio_state.json 在外部被修改，则实时刷新内存中的持仓变量。
    返回更新后的 (current_balance, shares_held, last_price, initial_balance)
    """
    global portfolio_state_mtime
    try:
        if not os.path.exists(PORTFOLIO_STATE_FILE):
            return current_balance, shares_held, last_price, initial_balance

        mtime = os.path.getmtime(PORTFOLIO_STATE_FILE)
        if portfolio_state_mtime is None or (mtime is not None and portfolio_state_mtime is not None and mtime > portfolio_state_mtime + 1e-6):
            state = load_portfolio_state()
            if state and state.get('stock_code') == STOCK_CODE:
                shares_held = state.get('shares_held', shares_held)
                last_price = state.get('last_price', last_price)
                initial_balance = state.get('initial_balance', initial_balance)
                cost_price = state.get('cost_price') or state.get('actual_buy_price') or last_price
                
                if cost_price is None or (isinstance(cost_price, (int, float)) and cost_price <= 0):
                    cost_price = last_price if last_price and last_price > 0 else 0
                
                if initial_balance and initial_balance > 0 and cost_price and cost_price > 0:
                    position_value = shares_held * cost_price
                    current_balance = max(0.0, initial_balance - position_value)
                elif shares_held <= 0:
                    current_balance = initial_balance if initial_balance and initial_balance > 0 else state.get('current_balance', current_balance)
                
                portfolio_state_mtime = mtime
                print(f"   🔄 检测到持仓状态更新: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元")
        else:
            portfolio_state_mtime = mtime
    except Exception as e:
        pass  # 静默处理错误
    
    return current_balance, shares_held, last_price, initial_balance

# ==================== 智能融合决策系统 ====================

# 动态权重调整：记录模型历史表现
model_performance_history = {
    'ppo': [],
    'lstm': [],
    'transformer': [],
    'holographic': []
}

def update_model_performance(model_name, prediction_error):
    """更新模型表现历史（用于动态权重调整）"""
    global model_performance_history
    if model_name in model_performance_history:
        model_performance_history[model_name].append(abs(prediction_error))
        # 只保留最近100次的表现
        if len(model_performance_history[model_name]) > 100:
            model_performance_history[model_name].pop(0)

def adjust_weights_dynamically(current_weights, current_price, predictions):
    """
    V11改进：动态调整模型权重
    
    Args:
        current_weights: 当前权重字典
        current_price: 当前价格
        predictions: 预测字典 {'lstm': ..., 'transformer': ..., ...}
    
    Returns:
        调整后的权重字典
    """
    if not ENABLE_DYNAMIC_WEIGHTS:
        return current_weights
    
    adjusted_weights = current_weights.copy()
    
    # 计算每个模型的预测误差
    errors = {}
    for model_name in ['lstm', 'transformer']:
        if model_name in predictions and predictions[model_name] is not None:
            error = abs(predictions[model_name] - current_price) / current_price if current_price > 0 else 1.0
            errors[model_name] = error
            update_model_performance(model_name, predictions[model_name] - current_price)
    
    # 根据历史表现调整权重
    for model_name in ['ppo', 'lstm', 'transformer', 'holographic']:
        if model_name in model_performance_history and len(model_performance_history[model_name]) > 10:
            perf_history = model_performance_history[model_name]
            # 确保数组不为空
            if len(perf_history) > 0:
                # 计算平均误差（误差越小，权重应该越大）
                avg_error = np.mean(perf_history) if len(perf_history) > 0 else 0.0
                # 归一化误差（转换为权重调整因子）
                max_error = max(perf_history) if perf_history else 1.0
                if max_error > 0 and not np.isnan(avg_error):
                    performance_score = 1.0 - (avg_error / max_error)  # 表现越好，分数越高
                    # 调整权重
                    adjustment = (performance_score - 0.5) * WEIGHT_ADAPTATION_RATE
                    adjusted_weights[model_name] = np.clip(
                        current_weights[model_name] + adjustment,
                        WEIGHT_MIN,
                        WEIGHT_MAX
                    )
    
    # 归一化权重，确保总和为1
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        for key in adjusted_weights:
            adjusted_weights[key] /= total_weight
    
    return adjusted_weights

def fuse_multi_model_predictions(ppo_action, lstm_prediction, transformer_prediction, 
                                 holographic_signal, model_weights=None, current_price=None):
    """
    融合多个模型的预测结果（V11改进版：支持动态权重）
    
    Args:
        ppo_action: PPO模型的动作（0-6）
        lstm_prediction: LSTM/GRU的预测价格
        transformer_prediction: Transformer的预测价格
        holographic_signal: 全息模型的信号
        model_weights: 模型权重字典
        current_price: 当前价格（用于动态权重调整）
    
    Returns:
        融合后的最终动作和置信度
    """
    if model_weights is None:
        model_weights = MODEL_WEIGHTS.copy()
    
    # V11改进：动态调整权重
    if current_price is not None and ENABLE_DYNAMIC_WEIGHTS:
        predictions = {
            'lstm': lstm_prediction,
            'transformer': transformer_prediction
        }
        model_weights = adjust_weights_dynamically(model_weights, current_price, predictions)
    
    # 将价格预测转换为动作倾向
    final_action = ppo_action  # 默认使用PPO的动作
    confidence = 0.5
    
    # 如果多个模型一致，提高置信度
    signals = []
    if ppo_action is not None:
        signals.append(('ppo', ppo_action))
    if holographic_signal:
        signal_type = holographic_signal.get('signal', 'hold')
        if signal_type == 'buy':
            signals.append(('holographic', 4))  # 买入倾向
        elif signal_type == 'sell':
            signals.append(('holographic', 0))  # 卖出倾向
    
    # 根据价格预测调整
    if lstm_prediction is not None and transformer_prediction is not None:
        avg_prediction = (lstm_prediction + transformer_prediction) / 2
        # 这里可以根据当前价格和预测价格的差异调整动作
        pass
    
    return final_action, confidence, model_weights

# ==================== 主循环 ====================

print("\n" + "=" * 70)
print("🚀 开始 V11 实时预测循环...")
print("=" * 70)
print("⚠️  重要提示: 这是 V11 全功能集成版本")
print("=" * 70 + "\n")

# 运行状态
current_balance = 20000.0
shares_held = 0.0
last_price = 0.0
initial_balance = 20000.0
last_action = None

# 模型训练状态
lstm_trained = False
transformer_trained = False
lstm_normalization_params = None
transformer_normalization_params = None

# V11回测数据存储
if ENABLE_BACKTEST:
    backtest_predictions = []  # 存储预测值
    backtest_actuals = []  # 存储实际值
    backtest_timestamps = []  # 存储时间戳

# 加载持仓状态
portfolio_state = load_portfolio_state()
if portfolio_state:
    if portfolio_state.get('stock_code') == STOCK_CODE:
        current_balance = portfolio_state.get('current_balance', 20000.0)
        shares_held = portfolio_state.get('shares_held', 0.0)
        last_price = portfolio_state.get('last_price', 0.0)
        initial_balance = portfolio_state.get('initial_balance', 20000.0)
        print(f"✅ 已加载持仓状态: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元")

# 启动可视化自动更新
if visualizer:
    try:
        visualizer.start_auto_update()
    except:
        pass

# 示例文本数据
sample_texts = [
    "该股票今日表现强势，市场看好其未来发展前景",
    "受利空消息影响，股价出现下跌",
    "公司业绩超预期，投资者信心增强"
]
text_index = 0

iteration_count = 0

while True:
    try:
        # 检查持仓状态更新（来自Web编辑器）
        if ENABLE_WEB_EDITOR:
            current_balance, shares_held, last_price, initial_balance = refresh_portfolio_from_file_if_changed(
                current_balance, shares_held, last_price, initial_balance
            )
        
        iteration_count += 1
        print(f"\n{'='*70}")
        print(f"📊 第 {iteration_count} 轮预测 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # 获取数据
        df = None
        if multi_source_manager:
            try:
                df, source = multi_source_manager.fetch_data(days=7)
                if df is not None and len(df) > 0:
                    print(f"   📊 数据来源: {source}")
            except Exception as e:
                print(f"   ⚠️  多数据源管理器获取失败: {e}")
        
        if df is None or len(df) == 0:
            try:
                code_info = convert_stock_code(STOCK_CODE)
                df = fetch_akshare_5min(code_info, days=7)
            except Exception as e:
                print(f"   ⚠️  数据获取失败: {e}")
                time.sleep(60)
                continue
        
        if df is None or len(df) == 0:
            print(f"⏸️  未找到数据")
            time.sleep(60)
            continue
        
        df = df.sort_values('time')
        closes = df['close'].astype(float).values
        
        if len(closes) < 126:
            print(f"⚠️  数据不足（需要126条，实际{len(closes)}条）")
            time.sleep(60)
            continue
        
        current_price = closes[-1]
        volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
        
        print(f"   💰 当前价格: {current_price:.2f}")
        print(f"   📈 成交量: {volume:,.0f}")
        
        # ========== V7: 技术指标计算 ==========
        indicator_summary = None
        if tech_indicators:
            try:
                df_with_indicators = tech_indicators.calculate_all(df)
                if 'KDJ' in df_with_indicators.columns:
                    kdj_values = df_with_indicators['KDJ'].iloc[-1]
                    rsi = df_with_indicators.get('RSI', pd.Series([0])).iloc[-1] if 'RSI' in df_with_indicators.columns else 0
                    obv_ratio = df_with_indicators.get('OBV_Ratio', pd.Series([1.0])).iloc[-1] if 'OBV_Ratio' in df_with_indicators.columns else 1.0
                    macd = df_with_indicators.get('MACD', pd.Series([0])).iloc[-1] if 'MACD' in df_with_indicators.columns else 0
                    
                    indicator_summary = {
                        'KDJ': kdj_values if isinstance(kdj_values, dict) else {'K': 0, 'D': 0, 'J': 0},
                        'RSI': rsi,
                        'OBV': {'OBV_Ratio': obv_ratio},
                        'MACD': {'MACD': macd}
                    }
                    print(f"   📊 V7技术指标: KDJ={indicator_summary['KDJ']}, RSI={rsi:.2f}")
            except Exception as e:
                print(f"   ⚠️  技术指标计算失败: {e}")
        
        # ========== V7: LLM指标解释 ==========
        if llm_interpreter and indicator_summary:
            try:
                interpretation = llm_interpreter.interpret_indicators(
                    indicator_summary,
                    current_price=current_price
                )
                if interpretation:
                    print(f"   🤖 V7 LLM解释: {interpretation.get('summary', '无')}")
            except Exception as e:
                print(f"   ⚠️  LLM解释失败: {e}")
        
        # ========== V7: PPO模型预测 ==========
        ppo_action = None
        ppo_operation = "持有"
        if ppo_model:
            try:
                obs = np.array(closes[-126:], dtype=np.float32)
                action, _states = ppo_model.predict(obs, deterministic=True)
                ppo_action = int(action)
                ppo_operation = map_action_to_operation(ppo_action)
                print(f"   🎯 V7 PPO动作: {ppo_operation} (动作={ppo_action})")
            except Exception as e:
                print(f"   ⚠️  PPO预测失败: {e}")
        
        # ========== V9: LSTM/GRU预测 ==========
        lstm_prediction = None
        if lstm_processor and ENABLE_LSTM_PREDICTION:
            try:
                if not lstm_trained and len(closes) >= LSTM_SEQ_LENGTH * 2:
                    print("   📚 V9训练LSTM模型...")
                    # V11改进：使用滑动窗口归一化
                    if USE_SLIDING_WINDOW_NORMALIZE and len(closes) > SLIDING_WINDOW_SIZE:
                        recent_closes = closes[-SLIDING_WINDOW_SIZE:]
                        print(f"      📊 使用滑动窗口归一化（窗口大小: {SLIDING_WINDOW_SIZE}）")
                    else:
                        recent_closes = closes
                        print(f"      📊 使用全局归一化（数据点: {len(closes)}）")
                    
                    normalized_data, norm_params = lstm_processor.normalize(recent_closes)
                    lstm_normalization_params = norm_params
                    X, y = lstm_processor.create_sequences(normalized_data)
                    if len(X) > 0:
                        lstm_processor.train(X, y, epochs=50, batch_size=32, verbose=False)
                        lstm_trained = True
                        print("   ✅ V9 LSTM模型训练完成")
                
                if lstm_trained and lstm_normalization_params:
                    # 使用训练时的归一化参数对输入序列进行归一化
                    seq = closes[-LSTM_SEQ_LENGTH:]
                    # 手动归一化（使用训练时的参数，而不是重新计算）
                    norm_method = lstm_normalization_params.get('method', 'minmax')
                    if norm_method == 'minmax':
                        min_val = lstm_normalization_params['min']
                        max_val = lstm_normalization_params['max']
                        if max_val - min_val > 0:
                            normalized_seq = (seq - min_val) / (max_val - min_val)
                        else:
                            normalized_seq = np.zeros_like(seq)
                    elif norm_method == 'zscore':
                        mean_val = lstm_normalization_params['mean']
                        std_val = lstm_normalization_params['std']
                        if std_val > 0:
                            normalized_seq = (seq - mean_val) / std_val
                        else:
                            normalized_seq = np.zeros_like(seq)
                    else:
                        normalized_seq = seq
                    
                    # 预测（返回归一化后的预测值）
                    prediction_norm = lstm_processor.predict_next(normalized_seq)
                    # 反归一化预测结果
                    lstm_prediction = float(lstm_processor.denormalize(
                        np.array([prediction_norm]),
                        lstm_normalization_params
                    )[0]) if prediction_norm is not None else None
                    if lstm_prediction:
                        print(f"   📈 V9 LSTM预测价格: {lstm_prediction:.2f}")
            except Exception as e:
                print(f"   ⚠️  LSTM预测失败: {e}")
        
        # ========== V10: Transformer预测 ==========
        transformer_prediction = None
        if transformer_model and ENABLE_TRANSFORMER and len(closes) >= TRANSFORMER_MAX_SEQ_LEN:
            try:
                if not transformer_trained and len(closes) >= TRANSFORMER_MAX_SEQ_LEN * 2:
                    print("   📚 V10训练Transformer模型...")
                    # V11改进：使用滑动窗口归一化，避免全局偏低
                    if USE_SLIDING_WINDOW_NORMALIZE and len(closes) > SLIDING_WINDOW_SIZE:
                        recent_closes = closes[-SLIDING_WINDOW_SIZE:]
                        print(f"      📊 使用滑动窗口归一化（窗口大小: {SLIDING_WINDOW_SIZE}）")
                    else:
                        recent_closes = closes
                        print(f"      📊 使用全局归一化（数据点: {len(closes)}）")
                    
                    normalized_closes, norm_params = transformer_model.normalize(recent_closes)
                    transformer_normalization_params = norm_params
                    
                    X_list, y_list = [], []
                    for i in range(TRANSFORMER_MAX_SEQ_LEN, len(normalized_closes)):
                        X_list.append(normalized_closes[i-TRANSFORMER_MAX_SEQ_LEN:i])
                        y_list.append(normalized_closes[i])
                    
                    if len(X_list) > 0:
                        X = np.array(X_list).reshape(len(X_list), TRANSFORMER_MAX_SEQ_LEN, 1)
                        y = np.array(y_list).reshape(len(y_list), 1)
                        # 改进建议：增加训练轮数(epochs)可以提高预测准确性
                        transformer_model.train(
                            X, y, epochs=50, batch_size=32,
                            learning_rate=0.001, validation_split=0.2, verbose=False
                        )
                        transformer_trained = True
                        print("   ✅ V10 Transformer模型训练完成")
                        # 输出归一化参数信息，便于诊断
                        if norm_params.get('method') == 'minmax':
                            print(f"      📊 归一化范围: [{norm_params['min']:.2f}, {norm_params['max']:.2f}], 当前价格: {current_price:.2f}")
                
                if transformer_trained and transformer_normalization_params:
                    seq = closes[-TRANSFORMER_MAX_SEQ_LEN:]
                    # 使用训练时的归一化参数进行归一化（而不是重新计算）
                    norm_method = transformer_normalization_params.get('method', 'minmax')
                    if norm_method == 'minmax':
                        min_val = transformer_normalization_params['min']
                        max_val = transformer_normalization_params['max']
                        if max_val - min_val > 0:
                            normalized_seq = (seq - min_val) / (max_val - min_val)
                        else:
                            normalized_seq = np.zeros_like(seq)
                    elif norm_method == 'zscore':
                        mean_val = transformer_normalization_params['mean']
                        std_val = transformer_normalization_params['std']
                        if std_val > 0:
                            normalized_seq = (seq - mean_val) / std_val
                        else:
                            normalized_seq = np.zeros_like(seq)
                    else:
                        normalized_seq = seq
                    
                    # 预测（返回归一化后的预测值）
                    prediction_norm = transformer_model.predict_next(normalized_seq)
                    # 反归一化预测结果
                    transformer_prediction = float(transformer_model.denormalize(
                        np.array([prediction_norm]),
                        transformer_normalization_params
                    )[0]) if prediction_norm is not None else None
                    if transformer_prediction:
                        # 添加诊断信息
                        norm_method = transformer_normalization_params.get('method', 'minmax')
                        if norm_method == 'minmax':
                            min_val = transformer_normalization_params['min']
                            max_val = transformer_normalization_params['max']
                            price_diff = transformer_prediction - current_price
                            price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                            print(f"   🔮 V10 Transformer预测价格: {transformer_prediction:.2f} (当前价格: {current_price:.2f}, 差异: {price_diff:+.2f} ({price_diff_pct:+.2f}%))")
                            print(f"      📊 归一化范围: [{min_val:.2f}, {max_val:.2f}], 当前价格在范围中的位置: {((current_price - min_val) / (max_val - min_val) * 100):.1f}%")
                            if transformer_prediction < current_price:
                                print(f"      ⚠️  预测偏低可能原因:")
                                print(f"         1. 训练数据中大部分价格低于当前价格，模型倾向于保守预测")
                                print(f"         2. 归一化范围 [{min_val:.2f}, {max_val:.2f}] 可能包含历史极值，导致当前价格归一化后偏小")
                                print(f"         3. 模型训练轮数较少(50轮)，可能未充分学习价格趋势")
                                print(f"         4. Transformer模型倾向于预测接近历史均值的值，而非极端值")
                        else:
                            print(f"   🔮 V10 Transformer预测价格: {transformer_prediction:.2f} (当前价格: {current_price:.2f})")
            except Exception as e:
                print(f"   ⚠️  Transformer预测失败: {e}")
        
        # ========== V10: 多模态处理 ==========
        multimodal_result = None
        if multimodal_processor and ENABLE_MULTIMODAL:
            try:
                # V11改进：使用真实新闻源（LLM市场情报）
                text_data = None
                if USE_REAL_NEWS_SOURCE and llm_agent:
                    try:
                        # 获取当前日期的市场情报
                        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
                        intelligence = llm_agent.get_market_intelligence(today_str)
                        if intelligence and 'summary' in intelligence:
                            text_data = intelligence['summary']
                            print(f"   📰 V11使用真实新闻源: {text_data[:50]}...")
                    except Exception as e:
                        if FALLBACK_TO_SAMPLE_TEXTS:
                            text_data = sample_texts[text_index % len(sample_texts)]
                            text_index += 1
                            print(f"   ⚠️  获取真实新闻失败，使用样本文本: {e}")
                        else:
                            raise
                else:
                    # 使用样本文本
                    text_data = sample_texts[text_index % len(sample_texts)]
                    text_index += 1
                
                if text_data:
                    multimodal_result = multimodal_processor.process(
                        time_series_data=closes[-60:],
                        text_data=text_data
                    )
                    print(f"   🌐 V10多模态处理: 情感={multimodal_result.get('sentiment', {}).get('polarity', 0):.2f}")
            except Exception as e:
                print(f"   ⚠️  多模态处理失败: {e}")
        
        # ========== V10: 全息动态模型 ==========
        holographic_signal = None
        if holographic_model and ENABLE_HOLOGRAPHIC:
            try:
                holographic_result = holographic_model.process(
                    time_series_data=closes[-60:],
                    text_data=sample_texts[text_index % len(sample_texts)],
                    technical_indicators=indicator_summary,
                    market_intelligence=None
                )
                holographic_signal = holographic_result.get('comprehensive_signal')
                if holographic_signal:
                    print(f"   🌟 V10全息信号: {holographic_signal.get('signal', 'hold')} (置信度={holographic_signal.get('confidence', 0):.2f})")
            except Exception as e:
                print(f"   ⚠️  全息模型处理失败: {e}")
        
        # ========== V11: 智能融合决策 ==========
        if ENABLE_MULTI_MODEL_FUSION:
            final_action, confidence, adjusted_weights = fuse_multi_model_predictions(
                ppo_action, lstm_prediction, transformer_prediction,
                holographic_signal, MODEL_WEIGHTS.copy(), current_price
            )
            final_operation = map_action_to_operation(final_action)
            print(f"\n   ⭐ V11融合决策: {final_operation} (置信度={confidence:.2f})")
            if ENABLE_DYNAMIC_WEIGHTS:
                print(f"   📊 动态权重: PPO={adjusted_weights['ppo']:.1%}, LSTM={adjusted_weights['lstm']:.1%}, Transformer={adjusted_weights['transformer']:.1%}, 全息={adjusted_weights['holographic']:.1%}")
            else:
                print(f"   📊 模型权重: PPO={MODEL_WEIGHTS['ppo']:.1%}, LSTM={MODEL_WEIGHTS['lstm']:.1%}, Transformer={MODEL_WEIGHTS['transformer']:.1%}, 全息={MODEL_WEIGHTS['holographic']:.1%}")
        else:
            final_action = ppo_action
            final_operation = ppo_operation
        
        # ========== 更新可视化 ==========
        if visualizer:
            try:
                indicators_dict = {}
                
                # 从技术指标摘要中提取指标
                if indicator_summary:
                    if 'KDJ' in indicator_summary:
                        kdj = indicator_summary['KDJ']
                        if isinstance(kdj, dict):
                            indicators_dict['KDJ_K'] = kdj.get('K', 0)
                            indicators_dict['KDJ_D'] = kdj.get('D', 0)
                            indicators_dict['KDJ_J'] = kdj.get('J', 0)
                    if 'RSI' in indicator_summary:
                        indicators_dict['RSI'] = indicator_summary['RSI']
                    if 'MACD' in indicator_summary:
                        macd = indicator_summary['MACD']
                        if isinstance(macd, dict):
                            indicators_dict['MACD'] = macd.get('MACD', 0)
                    if 'OBV' in indicator_summary:
                        obv = indicator_summary['OBV']
                        if isinstance(obv, dict):
                            indicators_dict['OBV_Ratio'] = obv.get('OBV_Ratio', 1.0)
                
                # 如果技术指标计算失败，从原始数据计算简单指标
                if not indicators_dict and len(closes) >= 5:
                    try:
                        # 计算简单的移动平均线
                        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
                        ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
                        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
                        
                        indicators_dict['MA5'] = ma5
                        indicators_dict['MA10'] = ma10
                        indicators_dict['MA20'] = ma20
                        
                        # 计算简单的RSI（如果数据足够）
                        if len(closes) >= 14:
                            try:
                                deltas = np.diff(closes[-14:])
                                if len(deltas) > 0:
                                    gains = np.where(deltas > 0, deltas, 0)
                                    losses = np.where(deltas < 0, -deltas, 0)
                                    # 只计算非零值的均值，避免空数组警告
                                    valid_gains = gains[gains > 0]
                                    valid_losses = losses[losses > 0]
                                    avg_gain = np.mean(valid_gains) if len(valid_gains) > 0 else 0.0
                                    avg_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.01
                                    if avg_loss > 0 and not np.isnan(avg_gain) and not np.isnan(avg_loss):
                                        rs = avg_gain / avg_loss
                                        rsi = 100 - (100 / (1 + rs))
                                        if not np.isnan(rsi) and not np.isinf(rsi):
                                            indicators_dict['RSI'] = rsi
                            except Exception:
                                pass  # 如果计算失败，跳过RSI
                    except Exception as e:
                        pass  # 如果计算失败，至少传递空字典
                
                # 确保至少有一些数据传递给可视化器
                visualizer.add_data_point(
                    price=current_price,
                    volume=volume,
                    indicators=indicators_dict if indicators_dict else None,
                    prediction=transformer_prediction
                )
                # 调试信息：显示已添加的数据点数量
                if iteration_count % 5 == 0:  # 每5轮输出一次
                    print(f"   📊 可视化数据: 价格点数={len(visualizer.price_history)}, 指标数={len(visualizer.indicators_history)}")
            except Exception as e:
                print(f"   ⚠️  可视化更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        # ========== 更新持仓状态 ==========
        total_assets = current_balance + shares_held * current_price
        save_portfolio_state(STOCK_CODE, shares_held, current_balance, current_price, initial_balance)
        log_trade_operation(
            STOCK_CODE, final_operation, current_price,
            shares_held, current_balance, total_assets,
            status='预测', note=f'V11融合决策'
        )
        
        print(f"   💼 持仓: {shares_held:.2f}股 | 资金: {current_balance:.2f}元 | 总资产: {total_assets:.2f}元")
        
        # ========== V11: 量化回测 ==========
        if ENABLE_BACKTEST:
            try:
                # 记录预测值和实际值（用于下一轮计算误差）
                if transformer_prediction is not None:
                    backtest_predictions.append(transformer_prediction)
                    backtest_timestamps.append(datetime.datetime.now())
                    
                    # 如果有历史实际值，计算回测指标
                    if len(backtest_predictions) > 1 and len(backtest_actuals) > 0:
                        # 使用上一轮的实际价格作为当前预测的对比
                        if len(backtest_actuals) >= len(backtest_predictions) - 1:
                            # 计算最近N次的指标
                            n = min(20, len(backtest_predictions) - 1)  # 最近20次
                            recent_preds = backtest_predictions[-n-1:-1]  # 排除最新的预测
                            recent_actuals = backtest_actuals[-n:]
                            
                            if len(recent_preds) == len(recent_actuals) and len(recent_preds) > 0:
                                try:
                                    # 转换为numpy数组并检查有效性
                                    preds_array = np.array(recent_preds, dtype=np.float64)
                                    actuals_array = np.array(recent_actuals, dtype=np.float64)
                                    
                                    # 过滤掉NaN和Inf值
                                    valid_mask = np.isfinite(preds_array) & np.isfinite(actuals_array) & (actuals_array != 0)
                                    if np.sum(valid_mask) > 0:
                                        valid_preds = preds_array[valid_mask]
                                        valid_actuals = actuals_array[valid_mask]
                                        
                                        # 计算MAE (Mean Absolute Error)
                                        mae = np.mean(np.abs(valid_preds - valid_actuals))
                                        
                                        # 计算RMSE (Root Mean Squared Error)
                                        rmse = np.sqrt(np.mean((valid_preds - valid_actuals)**2))
                                        
                                        # 计算MAPE (Mean Absolute Percentage Error)
                                        mape = np.mean(np.abs((valid_preds - valid_actuals) / valid_actuals)) * 100
                                        
                                        # 计算方向准确率 (Direction Accuracy)
                                        if len(valid_preds) > 1:
                                            pred_directions = np.sign(np.diff(valid_preds))
                                            actual_directions = np.sign(np.diff(valid_actuals))
                                            if len(pred_directions) > 0:
                                                direction_accuracy = np.mean(pred_directions == actual_directions) * 100
                                            else:
                                                direction_accuracy = 0.0
                                        else:
                                            direction_accuracy = 0.0
                                        
                                        # 检查结果是否有效
                                        if not (np.isnan(mae) or np.isnan(rmse) or np.isnan(mape) or np.isnan(direction_accuracy)):
                                            if iteration_count % 10 == 0:  # 每10轮输出一次
                                                print(f"\n   📈 V11回测指标 (最近{np.sum(valid_mask)}次有效数据):")
                                                print(f"      MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | 方向准确率: {direction_accuracy:.1f}%")
                                except Exception as e:
                                    # 静默处理计算错误
                                    pass
                
                # 记录当前实际价格（用于下一轮计算）
                backtest_actuals.append(current_price)
                
            except Exception as e:
                print(f"   ⚠️  回测计算失败: {e}")
        
        print(f"{'='*70}\n")
        
        # 等待下一轮
        time.sleep(300)  # 5分钟更新一次
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在保存状态...")
        break
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(60)

# 清理资源
print("\n🔄 正在清理资源...")
if web_visualization:
    try:
        web_visualization.stop()
    except:
        pass

print("✅ V11系统已停止")

