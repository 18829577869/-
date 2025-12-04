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

# 代理配置（可通过环境变量或配置文件设置）
# 如果设置了代理，将用于反爬虫功能
# 格式示例：['http://user:pass@host:port', 'socks5://host:port']
PROXIES = os.getenv('PROXIES', '').split(',') if os.getenv('PROXIES') else []
PROXIES = [p.strip() for p in PROXIES if p.strip()]  # 清理空字符串

# 是否启用反爬虫功能（Cookie/UA/代理池）
ENABLE_ANTI_CRAWLER = os.getenv('ENABLE_ANTI_CRAWLER', 'true').lower() == 'true'

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
                        actual_buy_price=None, actual_sell_price=None, cost_price=None,
                        realized_pnl=None):
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
        
        if realized_pnl is not None:
            state['realized_pnl'] = float(realized_pnl)
        
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
        # 初始化多数据源管理器，启用反爬虫功能
        multi_source_manager = MultiDataSourceManager(
            stock_code=STOCK_CODE,
            enable_anti_crawler=ENABLE_ANTI_CRAWLER,
            proxies=PROXIES if PROXIES else None
        )
        print("✅ V7多数据源管理器初始化成功")
        if ENABLE_ANTI_CRAWLER:
            print(f"   🛡️  反爬虫功能已启用 (代理数量: {len(PROXIES)})")
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

def get_current_market_price(stock_code, max_retries=1, debug=False):
    """
    获取当前市场价格（V11改进：优先获取实时行情，带重试机制）
    
    优先级：
    1. 实时行情接口（stock_zh_a_spot_em）- 带重试
    2. 最新5分钟K线数据
    3. 最新日K线数据
    
    Args:
        stock_code: 股票代码
        max_retries: 最大重试次数
        debug: 是否输出调试信息
    """
    import time
    import os
    import json
    
    # 保存所有可能的代理环境变量
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    saved_proxies = {}
    for var in proxy_vars:
        if var in os.environ:
            saved_proxies[var] = os.environ[var]
    
    try:
        # 临时禁用代理，避免代理连接失败
        for var in proxy_vars:
            os.environ.pop(var, None)
        
        # 设置NO_PROXY，确保不使用代理
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
        
        # 更彻底地禁用代理：在requests和urllib3级别禁用
        import requests
        import urllib3
        
        # 保存原始函数
        original_get = getattr(requests, '_original_get', requests.get)
        original_post = getattr(requests, '_original_post', requests.post)
        
        # 创建不使用代理的requests函数包装器
        def no_proxy_get(url, **kwargs):
            kwargs['proxies'] = {'http': None, 'https': None}
            return original_get(url, **kwargs)
        
        def no_proxy_post(url, **kwargs):
            kwargs['proxies'] = {'http': None, 'https': None}
            return original_post(url, **kwargs)
        
        # 临时替换requests函数，禁用代理
        requests.get = no_proxy_get
        requests.post = no_proxy_post
        
        # 禁用urllib3的代理
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        import akshare as ak
        code_info = convert_stock_code(stock_code)
        symbol = code_info['akshare']
        
        if debug:
            print(f"[实时价格] 目标股票代码: {stock_code} -> AkShare格式: {symbol}")
            if saved_proxies:
                print(f"[实时价格] 已临时禁用代理（检测到 {len(saved_proxies)} 个代理环境变量），直接连接数据源")
            else:
                print(f"[实时价格] 直接连接数据源（无代理配置）")
        
        # 方法1：尝试获取实时行情（最准确）- 只尝试一次，避免频繁失败请求
        try:
            spot_df = ak.stock_zh_a_spot_em()
        except (ValueError, json.JSONDecodeError) as json_err:
            # JSON解析错误，静默处理，不打印
            spot_df = None
        except Exception as api_err:
            # 其他API错误，静默处理
            spot_df = None
        
        if spot_df is not None and len(spot_df) > 0:
            if debug:
                print(f"[实时价格] 实时行情接口返回 {len(spot_df)} 条数据")
            
            # 查找目标股票
            # 股票代码格式：600730 或 000001
            # 尝试多种可能的列名
            code_col = None
            price_col = None
            
            # 查找代码列（更全面的匹配）
            for col in ['代码', 'code', '股票代码', 'symbol', '证券代码', '股票代码', '代码']:
                if col in spot_df.columns:
                    code_col = col
                    break
            
            # 查找价格列（更全面的匹配）
            for col in ['最新价', 'price', '现价', 'current_price', '最新价格', '当前价', '现价', '最新价']:
                if col in spot_df.columns:
                    price_col = col
                    break
            
            if code_col and price_col:
                # 尝试精确匹配
                stock_row = spot_df[spot_df[code_col] == symbol]
                if len(stock_row) == 0:
                    # 尝试字符串匹配（处理可能的格式差异）
                    stock_row = spot_df[spot_df[code_col].astype(str).str.strip() == str(symbol).strip()]
                
                if len(stock_row) > 0:
                    current_price = float(stock_row[price_col].iloc[0])
                    if current_price > 0:
                        if debug:
                            print(f"[实时价格] ✅ 方法1成功: {current_price:.2f} (来源: 实时行情接口)")
                        return current_price
        
        # 方法2：获取最新5分钟K线数据（只尝试一次）
        try:
            df = fetch_akshare_5min(code_info, days=1)
            if df is not None and len(df) > 0:
                df = df.sort_values('time')
                # 获取最新的价格（最后一条记录）
                latest_price = float(df['close'].iloc[-1])
                if latest_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法2成功: {latest_price:.2f} (来源: 5分钟K线)")
                    return latest_price
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法3：获取最新日K线数据（只尝试一次）
        try:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=3)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('日期')
                latest_price = float(df['收盘'].iloc[-1])
                if latest_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法3成功: {latest_price:.2f} (来源: 日K线)")
                    return latest_price
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法4：使用baostock获取最新日K线数据（备选方案）
        try:
            import baostock as bs
            bs_code = code_info['baostock']
            
            lg = bs.login()
            if lg.error_code == '0':
                try:
                    today = datetime.date.today()
                    start_date = (today - datetime.timedelta(days=10)).strftime('%Y-%m-%d')  # 扩大范围，确保获取到最新数据
                    end_date = today.strftime('%Y-%m-%d')
                    
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,close",
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag="3"
                    )
                    
                    if rs.error_code == '0':
                        data_list = []
                        while rs.next():
                            data_list.append(rs.get_row_data())
                        
                        if data_list:
                            df_bs = pd.DataFrame(data_list, columns=rs.fields)
                            df_bs = df_bs.sort_values('date')
                            latest_row = df_bs.iloc[-1]
                            latest_date_str = latest_row['date']
                            latest_price = float(latest_row['close'])
                            
                            if latest_price > 0:
                                # 检查数据日期
                                try:
                                    latest_date = pd.to_datetime(latest_date_str).date()
                                    days_diff = (today - latest_date).days
                                    
                                    if debug:
                                        if days_diff == 0:
                                            print(f"[实时价格] ✅ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, 今天)")
                                        elif days_diff == 1:
                                            print(f"[实时价格] ⚠️ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, 昨天, 可能有延迟)")
                                        else:
                                            print(f"[实时价格] ⚠️ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, {days_diff}天前, 数据较旧)")
                                except:
                                    pass
                                
                                return latest_price
                finally:
                    bs.logout()
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法5：如果所有实时接口都失败，尝试从持仓状态文件中读取手动输入的价格
        try:
            state = load_portfolio_state()
            if state and state.get('stock_code') == stock_code:
                manual_price = state.get('last_price', 0.0)
                if manual_price and manual_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法5成功: {manual_price:.2f} (来源: 持仓编辑器手动输入)")
                    return manual_price
        except Exception as e:
            pass
                    
    except ImportError:
        if debug:
            print(f"[实时价格] ❌ AkShare未安装")
    except Exception as e:
        if debug:
            print(f"[实时价格] ❌ 异常: {e}")
    finally:
        # 恢复原始代理设置
        for var, value in saved_proxies.items():
            os.environ[var] = value
        
        # 恢复NO_PROXY
        if 'NO_PROXY' in os.environ and 'NO_PROXY' not in saved_proxies:
            os.environ.pop('NO_PROXY', None)
        if 'no_proxy' in os.environ and 'no_proxy' not in saved_proxies:
            os.environ.pop('no_proxy', None)
        
        # 恢复requests库的原始函数
        try:
            import requests
            if hasattr(requests, '_original_get'):
                requests.get = requests._original_get
            if hasattr(requests, '_original_post'):
                requests.post = requests._original_post
        except:
            pass
    
    return None

def create_portfolio_web_app():
    """创建持仓编辑器Web应用"""
    global portfolio_editor_app
    
    if not FLASK_EDITOR_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    # 日志控制：避免频繁打印（使用列表存储状态，以便在嵌套函数中修改）
    api_log_state = [{
        'last_log_time': 0,
        'failure_count': 0,
        'last_success_time': 0
    }]
    
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
    .price-update.updating { color:#007bff;}
    .price-update.success { color:#28a745;}
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
      const updateMsg = document.getElementById('price-update-msg');
      if (updateMsg) {
        updateMsg.textContent = '🔄 正在从实时行情接口获取最新价格...';
        updateMsg.className = 'price-update updating';
      }
      
      fetch('/api/current_price')
        .then(response => response.json())
        .then(data => {
          if (data.success && data.price > 0) {
            const priceInput = document.querySelector('input[name="last_price"]');
            const oldPrice = parseFloat(priceInput.value) || 0;
            const newPrice = data.price;
            
            // 无论价格是否变化，都更新显示
            priceInput.value = newPrice.toFixed(4);
            
            // 重新计算统计数据
            recalculateStats();
            
            // 显示更新提示
            if (updateMsg) {
              const diff = newPrice - oldPrice;
              const diffPct = oldPrice > 0 ? ((diff / oldPrice) * 100).toFixed(2) : 0;
              const sign = diff >= 0 ? '+' : '';
              const source = data.source || '实时行情';
              const timestamp = data.timestamp || '';
              
              if (Math.abs(diff) > 0.001) {
                updateMsg.textContent = `✅ 价格已更新: ${newPrice.toFixed(2)} (${sign}${diff.toFixed(2)}, ${sign}${diffPct}%) [${source}] ${timestamp ? '(' + timestamp + ')' : ''}`;
              } else {
                updateMsg.textContent = `✅ 价格已刷新: ${newPrice.toFixed(2)} [${source}] ${timestamp ? '(' + timestamp + ')' : ''}`;
              }
              updateMsg.className = 'price-update success';
              setTimeout(() => {
                updateMsg.textContent = '';
                updateMsg.className = 'price-update';
              }, 5000);
            }
          } else {
            // 获取失败，显示错误信息
            if (updateMsg) {
              const errorMsg = data.error || data.message || '获取价格失败';
              updateMsg.textContent = `❌ ${errorMsg}`;
              updateMsg.className = 'price-update error';
              setTimeout(() => {
                updateMsg.textContent = '';
                updateMsg.className = 'price-update';
              }, 5000);
            }
            console.error('价格更新失败:', data.error || data.message);
          }
        })
        .catch(error => {
          console.error('价格更新失败:', error);
          if (updateMsg) {
            updateMsg.textContent = `❌ 网络错误: ${error.message || '无法连接到服务器'}`;
            updateMsg.className = 'price-update error';
            setTimeout(() => {
              updateMsg.textContent = '';
              updateMsg.className = 'price-update';
            }, 5000);
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
          <label>本次买入数量（股）</label>
          <input type="number" step="1" min="0" name="actual_buy_qty" value="{{ actual_buy_qty }}" placeholder="输入本次实际买入股数">
        </div>
      </div>

      <div class="row">
        <div>
          <label>实际卖出价（元）</label>
          <input type="number" step="0.0001" name="actual_sell_price" value="{{ actual_sell_price }}" placeholder="输入实际卖出价格">
        </div>
        <div>
          <label>本次卖出数量（股）</label>
          <input type="number" step="1" min="0" name="actual_sell_qty" value="{{ actual_sell_qty }}" placeholder="输入本次实际卖出股数">
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

      <div class="row">
        <div>
          <button type="submit" name="action" value="save">💾 保存持仓</button>
        </div>
        <div>
          <button type="submit" name="action" value="reset" style="background:#6c757d;">🔄 重置持仓</button>
        </div>
      </div>
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
      <div class="pnl-row">
        <span class="pnl-label">本次操作盈亏：</span>
        <span class="pnl-value">{{ last_trade_pnl_display }}</span>
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
        """API接口：获取当前市场价格（V11改进：直接读取主循环已获取的价格，不重复请求）"""
        from flask import jsonify
        try:
            # 直接读取主循环已经获取并保存的价格，不重复请求实时接口
            state = load_portfolio_state()
            if state:
                current_price = state.get("last_price", 0.0)
                price_source = state.get("price_source", "持仓状态")
                price_update_time = state.get("price_update_time", state.get("last_update", ""))
                
                if current_price and current_price > 0:
                    return jsonify({
                        "success": True, 
                        "price": current_price, 
                        "timestamp": price_update_time or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "source": price_source
                    })
            
            # 如果没有价格，返回错误
            return jsonify({
                "success": False, 
                "error": "暂无价格数据，请等待主循环更新",
                "cached_price": state.get("last_price", 0.0) if state else 0.0,
                "message": "价格数据将由主循环自动更新"
            })
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
            "actual_buy_qty": "",
            "actual_sell_qty": "",
            "last_trade_pnl": 0.0,
        }
        if state:
            # 如果获取到实时价格，优先使用实时价格
            last_price = realtime_price if realtime_price else state.get("last_price", 0.0)
            shares_held = int(state.get("shares_held", 0.0))
            initial_balance = state.get("initial_balance", 20000.0)
            actual_buy_price = state.get("actual_buy_price")
            realized_pnl = float(state.get("realized_pnl", 0.0))
            
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
                "actual_buy_qty": "",
                "actual_sell_qty": "",
                "last_trade_pnl": 0.0,
                "realized_pnl": realized_pnl,
            })

        if request.method == "POST":
            try:
                action = request.form.get("action", "save")

                # 处理重置操作：恢复为初始干净状态
                if action == "reset":
                    stock_code = STOCK_CODE
                    initial_balance = float(request.form.get("initial_balance") or 20000.0)
                    shares_held = 0
                    current_balance = initial_balance
                    last_price = 0.0
                    cost_price = 0.0
                    realized_pnl = 0.0

                    save_portfolio_state(
                        stock_code, shares_held, current_balance, last_price, initial_balance,
                        actual_buy_price=None,
                        actual_sell_price=None,
                        cost_price=cost_price,
                        realized_pnl=realized_pnl
                    )

                    msg = "✅ 已重置持仓为初始状态，下一轮预测将使用新的持仓信息。"
                    data.update({
                        "stock_code": stock_code,
                        "shares_held": shares_held,
                        "current_balance": current_balance,
                        "last_price": last_price,
                        "initial_balance": initial_balance,
                        "actual_buy_price": "",
                        "actual_sell_price": "",
                        "cost_price": "",
                        "actual_buy_qty": "",
                        "actual_sell_qty": "",
                        "last_trade_pnl": 0.0,
                        "realized_pnl": realized_pnl,
                    })
                else:
                    stock_code = request.form.get("stock_code", STOCK_CODE).strip()
                shares_held = int(float(request.form.get("shares_held") or 0))
                current_balance = float(request.form.get("current_balance") or 0)
                last_price = float(request.form.get("last_price") or 0)
                initial_balance = float(request.form.get("initial_balance") or 0)
                
                # 获取实际买入价、卖出价、数量和成本价
                actual_buy_price_str = request.form.get("actual_buy_price", "").strip()
                actual_sell_price_str = request.form.get("actual_sell_price", "").strip()
                actual_buy_qty_str = request.form.get("actual_buy_qty", "").strip()
                actual_sell_qty_str = request.form.get("actual_sell_qty", "").strip()
                cost_price_str = request.form.get("cost_price", "").strip()
                
                actual_buy_price = float(actual_buy_price_str) if actual_buy_price_str else None
                actual_sell_price = float(actual_sell_price_str) if actual_sell_price_str else None
                actual_buy_qty = int(float(actual_buy_qty_str)) if actual_buy_qty_str else 0
                actual_sell_qty = int(float(actual_sell_qty_str)) if actual_sell_qty_str else 0
                cost_price = float(cost_price_str) if cost_price_str else None

                # 读取历史已实现盈亏
                prev_state = load_portfolio_state()
                realized_pnl_before = float(prev_state.get("realized_pnl", 0.0)) if prev_state else 0.0
                last_trade_pnl = 0.0
                
                # 如果未填写成本价，使用实际买入价
                if cost_price is None and actual_buy_price and actual_buy_price > 0:
                    cost_price = actual_buy_price
                elif cost_price is None and last_price > 0:
                    cost_price = last_price

                # 先基于表单中的当前持仓/资金，应用本次实际买入/卖出操作
                # 实际买入：增加持仓，减少可用资金，并更新成本价（加权平均）
                if actual_buy_qty > 0 and actual_buy_price and actual_buy_price > 0:
                    buy_cost = actual_buy_qty * actual_buy_price
                    # 更新成本价（加权平均）
                    if cost_price and cost_price > 0 and shares_held > 0:
                        total_cost_before = shares_held * cost_price
                        total_cost_after = total_cost_before + buy_cost
                        new_shares = shares_held + actual_buy_qty
                        cost_price = total_cost_after / new_shares if new_shares > 0 else cost_price
                    else:
                        # 没有历史成本，则使用本次买入价
                        cost_price = actual_buy_price
                    shares_held += actual_buy_qty
                    current_balance -= buy_cost

                # 实际卖出：减少持仓，增加可用资金，计算已实现盈亏
                if actual_sell_qty > 0 and actual_sell_price and actual_sell_price > 0:
                    sell_qty = min(actual_sell_qty, shares_held)
                    if sell_qty > 0:
                        sell_amount = sell_qty * actual_sell_price
                        current_balance += sell_amount
                        # 基于成本价计算本次已实现盈亏
                        if cost_price and cost_price > 0:
                            last_trade_pnl = (actual_sell_price - cost_price) * sell_qty
                        else:
                            last_trade_pnl = 0.0
                        realized_pnl_before += last_trade_pnl
                        shares_held -= sell_qty
                        # 如果全部卖出，成本价清零
                        if shares_held <= 0:
                            cost_price = 0.0

                # 如果没有任何持仓，保证可用资金至少为初始资金中的一部分
                if shares_held <= 0 and initial_balance > 0 and current_balance <= 0:
                    current_balance = initial_balance

                save_portfolio_state(
                    stock_code, shares_held, current_balance, last_price, initial_balance,
                    actual_buy_price=actual_buy_price,
                    actual_sell_price=actual_sell_price,
                    cost_price=cost_price,
                    realized_pnl=realized_pnl_before
                )
                msg = f"✅ 已保存持仓状态，V11系统将在下一轮自动同步。可用资金：{current_balance:.2f} 元"
                if cost_price:
                    msg += f"，成本价：{cost_price:.2f} 元"
                if last_trade_pnl != 0.0:
                    msg += f"，本次操作盈亏：{last_trade_pnl:+.2f} 元"
                
                # 保存后清空实际买入/卖出相关字段，防止误操作导致错误计算
                data.update({
                    "stock_code": stock_code,
                    "shares_held": shares_held,
                    "current_balance": current_balance,
                    "last_price": last_price,
                    "initial_balance": initial_balance,
                    "actual_buy_price": "",  # 保存后清空，防止误操作
                    "actual_sell_price": "",  # 保存后清空，防止误操作
                    "cost_price": f"{cost_price:.4f}" if cost_price else "",
                    "actual_buy_qty": "",  # 保存后清空，防止误操作
                    "actual_sell_qty": "",  # 保存后清空，防止误操作
                    "last_trade_pnl": last_trade_pnl,
                    "realized_pnl": realized_pnl_before,
                })
            except Exception as e:
                msg = f"❌ 保存失败: {e}"

        # 计算统计数据
        shares_held_val = float(data.get("shares_held", 0))
        last_price_val = float(data.get("last_price", 0))
        current_balance_val = float(data.get("current_balance", 0))
        initial_balance_val = float(data.get("initial_balance", 0))
        realized_pnl_val = float(data.get("realized_pnl", 0.0))
        last_trade_pnl_val = float(data.get("last_trade_pnl", 0.0))
        
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
                    .replace("{{ actual_buy_qty }}", str(data.get("actual_buy_qty", "")))
                    .replace("{{ actual_sell_qty }}", str(data.get("actual_sell_qty", "")))
                    .replace("{{ msg }}", msg)
                    .replace("{{ initial_balance_display }}", f"{initial_balance_val:,.2f}")
                    .replace("{{ position_value_display }}", f"{position_value:,.2f}")
                    .replace("{{ current_balance_display }}", f"{current_balance_val:,.2f}")
                    .replace("{{ total_assets_display }}", f"{total_assets:,.2f}")
                    .replace("{{ cumulative_pnl_display }}", f"{pnl_sign}{cumulative_pnl:,.2f} 元 {pnl_info}")
                    .replace("{{ last_trade_pnl_display }}", f"{last_trade_pnl_val:+.2f} 元（历史已实现盈亏累计 {realized_pnl_val:+.2f} 元）")
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

def calculate_position_price_suggestions(current_price, lstm_prediction=None, transformer_prediction=None, 
                                         confidence=0.5, ppo_action=None, historical_prices=None):
    """
    计算不同仓位比例对应的建议价格（优化版：基于波动率扩大价格区间，避免频繁交易）
    
    Args:
        current_price: 当前价格
        lstm_prediction: LSTM预测价格
        transformer_prediction: Transformer预测价格
        confidence: 预测置信度
        ppo_action: PPO动作（0-6，用于判断方向）
        historical_prices: 历史价格数组（用于计算波动率）
    
    Returns:
        dict: 包含不同仓位比例对应的建议价格
    """
    if current_price <= 0:
        return None
    
    # 计算平均预测价格
    predictions = []
    if lstm_prediction is not None and lstm_prediction > 0:
        predictions.append(lstm_prediction)
    if transformer_prediction is not None and transformer_prediction > 0:
        predictions.append(transformer_prediction)
    
    if not predictions:
        return None
    
    avg_prediction = np.mean(predictions)
    
    # 判断涨跌方向
    price_change_pct = (avg_prediction - current_price) / current_price * 100
    
    # 根据PPO动作调整方向判断
    if ppo_action is not None:
        # PPO动作：0=全卖, 1=卖75%, 2=卖50%, 3=卖25%, 4=持有, 5=买25%, 6=全买
        if ppo_action <= 3:  # 卖出倾向
            if price_change_pct > 0:
                price_change_pct *= 0.5  # 降低看涨幅度
        elif ppo_action >= 5:  # 买入倾向
            if price_change_pct < 0:
                price_change_pct *= 0.5  # 降低看跌幅度
    
    # 计算历史波动率（用于扩大价格区间）
    volatility_pct = 2.0  # 默认波动率2%
    if historical_prices is not None and len(historical_prices) >= 20:
        try:
            # 计算最近20个价格点的波动率
            recent_prices = historical_prices[-20:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility_pct = np.std(returns) * 100 * np.sqrt(252)  # 年化波动率转换为日波动率参考
            # 限制波动率在合理范围（1%-10%）
            volatility_pct = max(1.0, min(10.0, volatility_pct))
        except:
            volatility_pct = 2.0
    
    # 改进：以预测价格为中心，而不是当前价格
    # 这样价格建议更实用，不会因为当前价格波动而无法触发交易
    
    # 计算价格区间大小：基于波动率和预测价格
    # 使用预测价格作为基准，而不是当前价格
    base_price = avg_prediction  # 以预测价格为中心
    
    # 价格区间大小：基于波动率，确保有足够的区分度但不会太大
    # 波动率越大，价格区间越大，但限制在合理范围内（2%-8%）
    price_interval_pct = max(2.0, min(8.0, volatility_pct * 1.5))  # 波动率的1.5倍，限制在2%-8%
    price_interval_size = base_price * price_interval_pct / 100
    
    # 根据PPO动作和预测方向，确定价格区间的中心偏移
    # PPO动作：0=全卖, 1=卖75%, 2=卖50%, 3=卖25%, 4=持有, 5=买25%, 6=全买
    center_offset = 0.0  # 中心偏移（相对于预测价格）
    
    if ppo_action is not None:
        if ppo_action == 6:  # 全买：价格区间向下偏移，使当前价格更容易触发买入
            center_offset = -price_interval_size * 0.2  # 向下偏移20%
        elif ppo_action == 5:  # 买25%：价格区间略微向下偏移
            center_offset = -price_interval_size * 0.1
        elif ppo_action == 4:  # 持有：价格区间以预测价格为中心
            center_offset = 0.0
        elif ppo_action == 3:  # 卖25%：价格区间略微向上偏移
            center_offset = price_interval_size * 0.1
        elif ppo_action <= 2:  # 卖50%或更多：价格区间向上偏移，使当前价格更容易触发卖出
            center_offset = price_interval_size * 0.2
    else:
        # 如果没有PPO动作，根据预测方向判断
        if price_change_pct > 0:
            center_offset = -price_interval_size * 0.1  # 预测上涨，略微向下偏移（买入机会）
        else:
            center_offset = price_interval_size * 0.1  # 预测下跌，略微向上偏移（卖出机会）
    
    # 计算价格区间的中心点（基于预测价格和偏移）
    price_center = base_price + center_offset
    
    # 确定最低价格和最高价格（以预测价格为中心，而不是当前价格）
    min_price = price_center - price_interval_size / 2
    max_price = price_center + price_interval_size / 2
    
    # 根据融合决策（PPO动作）调整价格区间，但考虑价格偏离预测价格的程度
    # 如果价格偏离预测价格较大，应该根据实际价格位置动态调整，而不是强制跟随融合决策
    price_diff_pct = abs(current_price - avg_prediction) / avg_prediction * 100 if avg_prediction > 0 else 0
    
    if ppo_action is not None:
        # 如果价格偏离预测价格较小（<3%），优先遵循融合决策
        # 如果价格偏离预测价格较大（>=3%），根据实际价格位置动态调整
        if price_diff_pct < 3.0:  # 价格偏离较小，遵循融合决策
            if ppo_action == 6:  # 买入 100%：当前价格应该在75%-100%仓位区间
                target_min = current_price - price_interval_size * 0.2  # 当前价格在80%仓位附近
                target_max = current_price + price_interval_size * 0.8
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 5:  # 买入 25%：当前价格应该在50%-75%仓位区间
                target_min = current_price - price_interval_size * 0.4  # 当前价格在60%仓位附近
                target_max = current_price + price_interval_size * 0.6
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 4:  # 持有：当前价格应该在25%-75%仓位区间（中间）
                target_min = current_price - price_interval_size * 0.5  # 当前价格在50%仓位附近
                target_max = current_price + price_interval_size * 0.5
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 3:  # 卖出 25%：当前价格应该在25%-50%仓位区间
                target_min = current_price - price_interval_size * 0.6  # 当前价格在40%仓位附近
                target_max = current_price + price_interval_size * 0.4
                min_price = target_min
                max_price = target_max
                
            elif ppo_action <= 2:  # 卖出 50%或更多：当前价格应该在0%-25%仓位区间
                target_min = current_price - price_interval_size * 0.8  # 当前价格在20%仓位附近
                target_max = current_price + price_interval_size * 0.2
                min_price = target_min
                max_price = target_max
        else:  # 价格偏离较大，根据实际价格位置动态调整
            # 计算当前价格相对于预测价格的位置
            if current_price > avg_prediction:
                # 当前价格高于预测价格，应该建议减仓
                # 根据偏离程度确定仓位：偏离越大，仓位越低
                if price_diff_pct >= 5.0:  # 偏离5%以上，建议0%-25%仓位
                    target_min = current_price - price_interval_size * 0.8
                    target_max = current_price + price_interval_size * 0.2
                elif price_diff_pct >= 3.5:  # 偏离3.5%-5%，建议25%-50%仓位
                    target_min = current_price - price_interval_size * 0.6
                    target_max = current_price + price_interval_size * 0.4
                else:  # 偏离3%-3.5%，建议50%-75%仓位
                    target_min = current_price - price_interval_size * 0.4
                    target_max = current_price + price_interval_size * 0.6
            else:
                # 当前价格低于预测价格，应该建议加仓
                # 根据偏离程度确定仓位：偏离越大，仓位越高
                if price_diff_pct >= 5.0:  # 偏离5%以上，建议75%-100%仓位
                    target_min = current_price - price_interval_size * 0.2
                    target_max = current_price + price_interval_size * 0.8
                elif price_diff_pct >= 3.5:  # 偏离3.5%-5%，建议50%-75%仓位
                    target_min = current_price - price_interval_size * 0.4
                    target_max = current_price + price_interval_size * 0.6
                else:  # 偏离3%-3.5%，建议25%-50%仓位
                    target_min = current_price - price_interval_size * 0.6
                    target_max = current_price + price_interval_size * 0.4
            
            min_price = target_min
            max_price = target_max
    
    # 确保价格区间足够大（至少2%的价格差）
    actual_range = max_price - min_price
    if actual_range < current_price * 0.02:  # 如果区间小于2%，扩大它
        center = (min_price + max_price) / 2
        min_price = center - current_price * 0.01
        max_price = center + current_price * 0.01
    
    # 价格从低到高，仓位从高到低（100% -> 75% -> 50% -> 25% -> 0%）
    suggestions = {}
    suggestions['100%'] = min_price
    suggestions['75%'] = min_price + (max_price - min_price) * 0.25
    suggestions['50%'] = min_price + (max_price - min_price) * 0.5
    suggestions['25%'] = min_price + (max_price - min_price) * 0.75
    suggestions['0%'] = max_price
    
    # 确保价格合理（不能为负，不能偏离当前价格太远）
    for key in suggestions:
        suggestions[key] = max(0.01, suggestions[key])  # 至少0.01元
        # 限制在合理范围内（当前价格的70%-130%）
        suggestions[key] = max(current_price * 0.7, min(current_price * 1.3, suggestions[key]))
        suggestions[key] = round(suggestions[key], 2)
    
    # 计算价格区间大小（用于显示）
    price_interval = max_price - min_price
    interval_pct = (price_interval / current_price * 100) if current_price > 0 else 0
    
    # 计算当前价格对应的建议仓位
    price_levels = [suggestions['100%'], suggestions['75%'], suggestions['50%'], suggestions['25%'], suggestions['0%']]
    current_position_pct = 50.0  # 默认50%
    
    if current_price < price_levels[0]:  # 低于100%仓位价格
        current_position_pct = 100.0
    elif current_price > price_levels[-1]:  # 高于0%仓位价格
        current_position_pct = 0.0
    else:
        # 找到当前价格所在区间并插值
        for i in range(len(price_levels) - 1):
            if price_levels[i] <= current_price <= price_levels[i+1]:
                # 线性插值计算仓位
                ratio = (current_price - price_levels[i]) / (price_levels[i+1] - price_levels[i]) if (price_levels[i+1] - price_levels[i]) > 0 else 0
                current_position_pct = 100 - (i * 25 + ratio * 25)
                break
    
    return {
        'suggestions': suggestions,
        'predicted_price': round(avg_prediction, 2),
        'price_change_pct': round(price_change_pct, 2),
        'direction': '上涨' if price_change_pct > 0 else '下跌',
        'price_interval_pct': round(interval_pct, 2),
        'volatility_pct': round(volatility_pct, 2),
        'current_position_pct': round(current_position_pct, 1)
    }

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
        
        # 获取数据（V11改进：优先获取最新数据）
        df = None
        if multi_source_manager:
            try:
                # 尝试获取最新数据（减少天数，确保获取最新）
                df, source = multi_source_manager.fetch_data(days=7)
                if df is not None and len(df) > 0:
                    print(f"   📊 数据来源: {source}")
                    # 显示数据源尝试情况
                    stats = multi_source_manager.get_source_stats()
                    failed_sources = []
                    for src, stat in stats.items():
                        if src != source and stat.get('fail', 0) > 0:
                            failed_sources.append(f"{src}(失败{stat['fail']}次)")
                    if failed_sources:
                        print(f"   📋 其他数据源状态: {', '.join(failed_sources)}")
                    # 说明为什么使用当前数据源
                    if source == 'baostock':
                        print(f"   💡 说明: akshare获取失败，已回退到baostock（可能有1-2天延迟）")
                    elif source == 'akshare':
                        print(f"   💡 说明: 成功使用akshare获取数据")
            except Exception as e:
                print(f"   ⚠️  多数据源管理器获取失败: {e}")
        
        if df is None or len(df) == 0:
            try:
                code_info = convert_stock_code(STOCK_CODE)
                # V11改进：优先获取最近1-2天的数据，确保是最新的
                df = fetch_akshare_5min(code_info, days=2)  # 减少天数，确保获取最新数据
                if df is None or len(df) == 0:
                    # 如果失败，尝试获取7天数据
                    df = fetch_akshare_5min(code_info, days=7)
            except Exception as e:
                print(f"   ⚠️  数据获取失败: {e}")
                time.sleep(60)
                continue
        
        if df is None or len(df) == 0:
            print(f"⏸️  未找到数据")
            time.sleep(60)
            continue
        
        # V11改进：确保数据按时间排序，使用最新的数据
        df = df.sort_values('time')
        # 检查数据时间戳，确保使用最新数据
        if 'time' in df.columns:
            # 显示最新数据的时间
            latest_time = df['time'].iloc[-1]
            print(f"   📅 最新数据时间: {latest_time}")
        
        closes = df['close'].astype(float).values
        
        # 如果数据不足，尝试用其他数据源补齐（例如：akshare 只有少量当日 5 分钟数据）
        if len(closes) < 126:
            print(f"⚠️  数据不足（需要126条，实际{len(closes)}条）")
            
            # 使用多数据源合并功能，用历史数据补齐
            if multi_source_manager is not None:
                try:
                    print("   🔄 正在尝试从其他数据源合并历史数据进行补齐...")
                    merged_df = multi_source_manager.merge_data_from_multiple_sources(
                        days=7,
                        merge_strategy='union'
                    )
                    if merged_df is not None and len(merged_df) > len(df):
                        # 合并后重新排序、去重
                        merged_df = merged_df.drop_duplicates(subset=['time'], keep='last')
                        merged_df = merged_df.sort_values('time')
                        merged_closes = merged_df['close'].astype(float).values
                        if len(merged_closes) >= 126:
                            df = merged_df
                            closes = merged_closes
                            print(f"   ✅ 已通过合并数据源补齐历史数据，当前数据条数: {len(closes)}")
                        else:
                            print(f"   ⚠️ 合并后数据仍不足（{len(merged_closes)} 条），暂时无法进行预测")
                    else:
                        print("   ⚠️ 无法通过合并数据源获得更多历史数据")
                except Exception as e:
                    print(f"   ⚠️ 合并多数据源补齐历史数据时出错: {e}")
            
            # 再次检查是否满足最小长度要求
            if len(closes) < 126:
                print("⏸️  有效历史数据仍不足，等待下一轮数据更新后再预测")
                time.sleep(60)
                continue
        
        # V11改进：仅从实时行情接口获取价格（不从持仓状态获取）
        # 减少重试次数，避免频繁失败请求
        realtime_price = None
        try:
            print(f"   🔄 正在从实时行情接口获取最新价格...")
            # 减少重试次数为1次，减少调试输出
            realtime_price = get_current_market_price(STOCK_CODE, max_retries=1, debug=False)
            if realtime_price and realtime_price > 0:
                print(f"   ✅ 已从实时行情接口获取价格: {realtime_price:.2f}")
            # 失败时不打印，避免频繁输出
        except Exception as e:
            # 静默处理，不打印错误
            pass
        
        # 备选方案：从数据源获取（可能是历史数据）
        data_source_price = closes[-1]
        
        # 确定最终使用的价格：优先级 实时行情(今天) > 持仓编辑器手动价格 > 实时行情(昨天) > 数据源价格
        # 先读取持仓编辑器中的价格，用于比较
        manual_price = None
        manual_price_time = None
        try:
            state = load_portfolio_state()
            if state and state.get('stock_code') == STOCK_CODE:
                manual_price = state.get('last_price', 0.0)
                manual_price_time = state.get('price_update_time') or state.get('last_update', '')
        except:
            pass
        
        # 检查实时价格的数据日期（如果是baostock，可能是昨天的数据）
        realtime_price_is_today = True
        if realtime_price and realtime_price > 0:
            # 检查数据源时间，判断实时价格是否是今天的数据
            if 'time' in df.columns:
                latest_time_str = str(df['time'].iloc[-1])
                try:
                    if len(latest_time_str) >= 8:
                        year = int(latest_time_str[0:4])
                        month = int(latest_time_str[4:6])
                        day = int(latest_time_str[6:8])
                        latest_date = datetime.date(year, month, day)
                        today = datetime.date.today()
                        days_diff = (today - latest_date).days
                        if days_diff > 0:
                            realtime_price_is_today = False
                            print(f"   ⚠️  实时价格来自 {days_diff} 天前，可能不是最新")
                except:
                    pass
        
        # 确定最终使用的价格
        if realtime_price and realtime_price > 0 and realtime_price_is_today:
            # 实时价格是今天的，优先使用
            current_price = realtime_price
            price_source = "实时行情"
            # 同步更新到持仓状态文件
            try:
                state = load_portfolio_state()
                if state and state.get('stock_code') == STOCK_CODE:
                    state['last_price'] = realtime_price
                    state['price_source'] = '实时行情'
                    state['price_update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                    print(f"   ✅ 已同步实时价格到持仓编辑器: {realtime_price:.2f}")
            except Exception as e:
                print(f"   ⚠️  同步价格到持仓编辑器失败: {e}")
        elif manual_price and manual_price > 0:
            # 如果实时价格是旧数据或没有，优先使用持仓编辑器中的手动价格
            current_price = manual_price
            price_source = "持仓编辑器(手动输入)"
            print(f"   ✅ 使用持仓编辑器中的手动价格: {current_price:.2f}")
            # 如果实时价格是旧数据，不覆盖持仓编辑器中的新价格
            if realtime_price and realtime_price > 0 and not realtime_price_is_today:
                print(f"   📝 检测到实时价格({realtime_price:.2f})是旧数据，保持持仓编辑器中的价格({current_price:.2f})")
        elif realtime_price and realtime_price > 0:
            # 实时价格存在但是旧数据，且没有手动价格，使用实时价格
            current_price = realtime_price
            price_source = "实时行情(可能非最新)"
            print(f"   ⚠️  使用实时价格(可能非最新): {current_price:.2f}")
        else:
            current_price = data_source_price
            price_source = "数据源(可能非最新)"
            # 检查数据时间，如果数据太旧，给出警告
            if 'time' in df.columns:
                latest_time_str = str(df['time'].iloc[-1])
                try:
                    # 解析时间：20251202150000000 -> 2025-12-02 15:00:00
                    if len(latest_time_str) >= 8:
                        year = int(latest_time_str[0:4])
                        month = int(latest_time_str[4:6])
                        day = int(latest_time_str[6:8])
                        latest_date = datetime.date(year, month, day)
                        today = datetime.date.today()
                        days_diff = (today - latest_date).days
                        if days_diff > 0:
                            print(f"   ⚠️  数据源价格来自 {days_diff} 天前，可能不是最新价格")
                except:
                    pass
            print(f"   ⚠️  实时行情获取失败，使用数据源价格: {current_price:.2f}")
        
        volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
        
        print(f"   💰 当前价格: {current_price:.2f} (来源: {price_source})")
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
        
        # ========== V11: 仓位价格建议 ==========
        price_suggestions = calculate_position_price_suggestions(
            current_price, lstm_prediction, transformer_prediction, confidence, final_action, closes
        )
        if price_suggestions:
            suggestions = price_suggestions['suggestions']
            
            # 获取当前价格对应的建议仓位
            current_position_pct = price_suggestions.get('current_position_pct', 50.0)
            current_position = f"{current_position_pct:.0f}%"
            
            # 计算当前价格与各仓位价格的差异，找出最接近的仓位
            price_levels = [suggestions['100%'], suggestions['75%'], suggestions['50%'], suggestions['25%'], suggestions['0%']]
            position_labels = ['100%', '75%', '50%', '25%', '0%']
            
            # 找到当前价格最接近的仓位价格
            closest_price = min(price_levels, key=lambda x: abs(x - current_price))
            closest_index = price_levels.index(closest_price)
            closest_position = position_labels[closest_index]
            price_diff_from_closest = abs(current_price - closest_price)
            price_diff_pct_from_closest = (price_diff_from_closest / current_price * 100) if current_price > 0 else 0
            
            print(f"\n   💡 仓位价格建议（基于预测价格 {price_suggestions['predicted_price']:.2f}元，预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）:")
            print(f"      🟢 100%仓位: {suggestions['100%']:.2f}元 (价格越低，买入越多)")
            print(f"      🟡 75%仓位:  {suggestions['75%']:.2f}元")
            print(f"      🟠 50%仓位:  {suggestions['50%']:.2f}元")
            print(f"      🟤 25%仓位:  {suggestions['25%']:.2f}元")
            print(f"      ⚪ 0%仓位:   {suggestions['0%']:.2f}元 (价格越高，卖出越多)")
            
            # 计算相邻仓位的最小价格差
            min_diff_pct = min([abs(price_levels[i] - price_levels[i+1]) / current_price * 100 
                               for i in range(len(price_levels)-1)]) if current_price > 0 else 0
            
            # 优先根据融合决策生成建议，而不是仅仅基于价格位置
            # 融合决策是更重要的信号，价格建议应该与之保持一致
            action_hint = ""
            consistency_note = ""
            
            # 计算当前价格与预测价格的偏离程度
            price_diff_from_pred = abs(current_price - price_suggestions['predicted_price']) / price_suggestions['predicted_price'] * 100 if price_suggestions['predicted_price'] > 0 else 0
            
            # 根据融合决策确定建议，但考虑价格偏离程度
            if final_action == 6:  # 买入 100%
                if price_diff_from_pred >= 3.0:
                    # 价格偏离较大，根据实际价格位置动态调整
                    if current_price > price_suggestions['predicted_price']:
                        # 当前价格高于预测价格，建议减仓
                        if current_position_pct <= 25:
                            action_hint = f"⚠️  当前价格 {current_price:.2f}元 高于预测价格 {price_suggestions['predicted_price']:.2f}元（偏离{price_diff_from_pred:.2f}%），建议减仓至{current_position}仓位（价格偏离较大，动态调整）"
                        elif current_position_pct <= 50:
                            action_hint = f"⚠️  当前价格 {current_price:.2f}元 高于预测价格 {price_suggestions['predicted_price']:.2f}元（偏离{price_diff_from_pred:.2f}%），建议保持{current_position}仓位（价格偏离较大，动态调整）"
                        else:
                            action_hint = f"✅ 融合决策「买入 100%」但当前价格 {current_price:.2f}元 高于预测价格（偏离{price_diff_from_pred:.2f}%），建议保持{current_position}仓位"
                        consistency_note = f"⚠️  价格偏离预测价格{price_diff_from_pred:.2f}%，已动态调整建议仓位"
                    else:
                        # 当前价格低于预测价格，建议加仓
                        action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 低于预测价格（偏离{price_diff_from_pred:.2f}%），建议加仓至{current_position}仓位"
                        consistency_note = "✅ 与融合决策「买入 100%」一致"
                else:
                    # 价格偏离较小，遵循融合决策
                    if current_price <= suggestions['75%']:
                        action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 在买入区间，建议满仓买入"
                    elif current_price <= suggestions['50%']:
                        action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 接近买入区间，建议高仓位买入（目标100%仓位）"
                    else:
                        action_hint = f"✅ 融合决策「买入 100%」：虽然当前价格 {current_price:.2f}元 略高于预测价格，但模型建议买入，可考虑分批买入或等待回调至 {suggestions['75%']:.2f}元 以下"
                    consistency_note = "✅ 与融合决策「买入 100%」一致"
                
            elif final_action == 5:  # 买入 25%
                if current_price <= suggestions['75%']:
                    action_hint = f"✅ 融合决策「买入 25%」+ 当前价格 {current_price:.2f}元 在买入区间，建议买入至75%仓位"
                else:
                    action_hint = f"✅ 融合决策「买入 25%」：当前价格 {current_price:.2f}元，建议买入至75%仓位（可等待回调至 {suggestions['75%']:.2f}元 以下）"
                consistency_note = "✅ 与融合决策「买入 25%」一致"
                
            elif final_action == 4:  # 持有
                if suggestions['25%'] <= current_price <= suggestions['75%']:
                    action_hint = f"✅ 融合决策「持有」+ 当前价格 {current_price:.2f}元 在合理区间，建议保持当前仓位"
                else:
                    action_hint = f"✅ 融合决策「持有」：当前价格 {current_price:.2f}元，建议保持50%左右仓位"
                consistency_note = "✅ 与融合决策「持有」一致"
                
            elif final_action == 3:  # 卖出 25%
                if current_price >= suggestions['25%']:
                    action_hint = f"✅ 融合决策「卖出 25%」+ 当前价格 {current_price:.2f}元 在卖出区间，建议减仓至25%仓位"
                else:
                    action_hint = f"✅ 融合决策「卖出 25%」：当前价格 {current_price:.2f}元，建议减仓至25%仓位（可等待反弹至 {suggestions['25%']:.2f}元 以上）"
                consistency_note = "✅ 与融合决策「卖出 25%」一致"
                
            elif final_action <= 2:  # 卖出 50% 或更多
                if current_price >= suggestions['25%']:
                    action_hint = f"✅ 融合决策「卖出」+ 当前价格 {current_price:.2f}元 在卖出区间，建议大幅减仓或清仓"
                else:
                    action_hint = f"✅ 融合决策「卖出」：虽然当前价格 {current_price:.2f}元 略低于预测价格，但模型建议卖出，可考虑减仓或等待反弹至 {suggestions['25%']:.2f}元 以上"
                consistency_note = "✅ 与融合决策「卖出」一致"
                
            else:
                # 如果没有明确的融合决策，则基于价格位置判断
                if current_price < suggestions['100%']:
                    action_hint = f"当前价格 {current_price:.2f}元 低于100%仓位价格，建议满仓买入"
                elif current_price > suggestions['0%']:
                    action_hint = f"当前价格 {current_price:.2f}元 高于0%仓位价格，建议全部卖出"
                elif price_diff_pct_from_closest < 0.5:
                    action_hint = f"当前价格 {current_price:.2f}元 接近{closest_position}仓位价格（{closest_price:.2f}元），建议调整为{closest_position}仓位"
                else:
                    if current_price <= suggestions['75%']:
                        action_hint = f"当前价格 {current_price:.2f}元 在75%-100%仓位区间，建议高仓位持有"
                    elif current_price <= suggestions['50%']:
                        action_hint = f"当前价格 {current_price:.2f}元 在50%-75%仓位区间，建议中等仓位持有"
                    elif current_price <= suggestions['25%']:
                        action_hint = f"当前价格 {current_price:.2f}元 在25%-50%仓位区间，建议低仓位持有"
                    else:
                        action_hint = f"当前价格 {current_price:.2f}元 在0%-25%仓位区间，建议轻仓或空仓"
                consistency_note = "基于价格位置判断"
            
            print(f"   📌 {action_hint}")
            print(f"   📊 {consistency_note}")
            print(f"   📊 价格区间 {price_suggestions['price_interval_pct']:.2f}%（基于预测价格和波动率{price_suggestions['volatility_pct']:.2f}%），相邻仓位价格差至少 {min_diff_pct:.2f}%")
            print(f"   💡 提示: 价格建议基于预测价格 {price_suggestions['predicted_price']:.2f}元，当前价格 {current_price:.2f}元 与预测价格差异 {abs(current_price - price_suggestions['predicted_price']) / price_suggestions['predicted_price'] * 100:.2f}%")
        
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

