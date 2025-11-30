"""
V9 实时预测系统
新增功能：
1. LSTM/GRU 时间序列处理
2. 实时动态参数调整和自动学习优化
3. 注意力机制
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import datetime
import time
import json

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# 抑制警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==================== 导入模块 ====================

# 导入V7基础模块
try:
    from real_time_predict_v7_600730 import (
        calc_buy_trade, calc_sell_trade, is_trading_day, is_trading_time,
        convert_stock_code, fetch_tushare_5min, fetch_akshare_5min,
        fetch_baostock_5min, map_action_to_operation, save_portfolio_state,
        load_portfolio_state, log_trade_operation, init_trade_log
    )
    V7_MODULES_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入V7模块，部分功能可能受限")
    V7_MODULES_AVAILABLE = False

# 导入新技术模块
try:
    from technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    print("[警告] 技术指标模块不可用")
    TECHNICAL_INDICATORS_AVAILABLE = False

try:
    from multi_data_source_manager import MultiDataSourceManager
    MULTI_DATA_SOURCE_AVAILABLE = True
except ImportError:
    print("[警告] 多数据源管理器不可用")
    MULTI_DATA_SOURCE_AVAILABLE = False

try:
    from llm_indicator_interpreter import LLMIndicatorInterpreter
    LLM_INTERPRETER_AVAILABLE = True
except ImportError:
    print("[警告] LLM指标解释器不可用")
    LLM_INTERPRETER_AVAILABLE = False

# 导入V9新模块
try:
    from lstm_gru_time_series import TimeSeriesProcessor
    LSTM_AVAILABLE = True
except ImportError:
    print("[警告] LSTM/GRU模块不可用")
    LSTM_AVAILABLE = False

try:
    from dynamic_parameter_optimizer import (
        DynamicParameterOptimizer, AutoLearningOptimizer, ParameterRange
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    print("[警告] 参数优化器模块不可用")
    OPTIMIZER_AVAILABLE = False

# 导入强化学习模型
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    print("[警告] PPO模型不可用")
    PPO_AVAILABLE = False

# 导入LLM市场情报
try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    print("[警告] LLM市场情报不可用")
    LLM_AVAILABLE = False

# ==================== 配置参数 ====================

# 基础配置
MODEL_PATH = "ppo_stock_v7.zip"
STOCK_CODE = 'sh.600730'
LLM_PROVIDER = "deepseek"
ENABLE_LLM = True
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d')

# V9新功能配置
ENABLE_LSTM_PREDICTION = True  # 启用LSTM预测
ENABLE_DYNAMIC_OPTIMIZATION = True  # 启用动态参数优化
LSTM_MODEL_TYPE = 'lstm_attention'  # 'lstm', 'gru', 'lstm_attention'
LSTM_SEQ_LENGTH = 60  # 序列长度
LSTM_HIDDEN_SIZE = 64  # 隐藏层大小

# 参数优化配置
OPTIMIZATION_PARAMETERS = {
    'kdj_period': ParameterRange(5, 14, param_type='integer'),
    'rsi_period': ParameterRange(10, 20, param_type='integer'),
    'macd_fast': ParameterRange(8, 16, param_type='integer'),
    'lstm_hidden_size': ParameterRange(32, 128, param_type='integer'),
    'lstm_num_layers': ParameterRange(1, 3, param_type='integer'),
    'learning_rate': ParameterRange(0.0001, 0.01, param_type='continuous')
}

# 技术指标配置
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

TRADE_LOG_FILE = "trade_log.csv"
PORTFOLIO_STATE_FILE = "portfolio_state.json"

# ==================== 初始化 ====================

def init_v9_system():
    """V9系统初始化（仅在主程序运行时执行）"""
    print("=" * 70)
    print("V9 实时预测系统 - LSTM/GRU + 动态参数优化 + 注意力机制")
    print("=" * 70)
    print("📌 新增功能:")
    print("   - LSTM/GRU 时间序列预测")
    print("   - 实时动态参数调整")
    print("   - 自动学习参数优化")
    print("   - 注意力机制")
    print("=" * 70)

# 只有在直接运行时才执行初始化
if __name__ == "__main__":
    init_v9_system()

# 初始化技术指标计算器
tech_indicators = None
if TECHNICAL_INDICATORS_AVAILABLE:
    try:
        tech_indicators = TechnicalIndicators(**TECHNICAL_INDICATOR_CONFIG)
        print("✅ 技术指标计算器初始化成功")
    except Exception as e:
        print(f"⚠️  技术指标计算器初始化失败: {e}")

# 初始化多数据源管理器
multi_source_manager = None
if MULTI_DATA_SOURCE_AVAILABLE:
    try:
        multi_source_manager = MultiDataSourceManager(stock_code=STOCK_CODE)
        print("✅ 多数据源管理器初始化成功")
    except Exception as e:
        print(f"⚠️  多数据源管理器初始化失败: {e}")

# 初始化LSTM/GRU时间序列处理器
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
        print(f"✅ LSTM/GRU时间序列处理器初始化成功 (类型: {LSTM_MODEL_TYPE})")
    except Exception as e:
        print(f"⚠️  LSTM/GRU处理器初始化失败: {e}")
        lstm_processor = None

# 初始化动态参数优化器
param_optimizer = None
auto_learner = None
if OPTIMIZER_AVAILABLE and ENABLE_DYNAMIC_OPTIMIZATION:
    try:
        param_optimizer = DynamicParameterOptimizer(
            parameter_ranges=OPTIMIZATION_PARAMETERS,
            optimization_method='bayesian',
            adaptation_rate=0.1,
            exploration_rate=0.2,
            performance_window=100
        )
        
        auto_learner = AutoLearningOptimizer(
            parameter_optimizer=param_optimizer,
            learning_rate=0.01,
            momentum=0.9,
            decay_rate=0.99
        )
        
        print("✅ 动态参数优化器初始化成功")
        print("✅ 自动学习优化器初始化成功")
    except Exception as e:
        print(f"⚠️  参数优化器初始化失败: {e}")

# 初始化LLM市场情报
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
    except Exception as e:
        print(f"⚠️  LLM初始化失败: {e}")

# 初始化LLM指标解释器
llm_interpreter = None
if LLM_INTERPRETER_AVAILABLE and llm_agent:
    try:
        llm_interpreter = LLMIndicatorInterpreter(
            llm_agent=llm_agent,
            enable_cache=True
        )
        print("✅ LLM指标解释器初始化成功")
    except Exception as e:
        print(f"⚠️  LLM指标解释器初始化失败: {e}")

# 加载PPO模型
model = None
if PPO_AVAILABLE:
    try:
        if not os.path.exists(MODEL_PATH):
            possible_models = [
                "ppo_stock_v7.zip",
                "models_v7/best/best_model.zip",
            ]
            for model_file in possible_models:
                if os.path.exists(model_file):
                    MODEL_PATH = model_file
                    break
        
        model = PPO.load(MODEL_PATH)
        print(f"✅ PPO模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  PPO模型加载失败: {e}")

print("=" * 70)
print()

# 初始化交易日志
if V7_MODULES_AVAILABLE:
    try:
        init_trade_log()
    except:
        pass

# ==================== 主循环 ====================

def calculate_performance_score(prediction_accuracy: float, 
                               profit: float, 
                               risk_metric: float) -> float:
    """
    计算性能评分（用于参数优化）
    
    参数:
        prediction_accuracy: 预测准确率 (0-1)
        profit: 盈利（元）
        risk_metric: 风险指标（越低越好，0-1）
    
    返回:
        性能评分
    """
    # 综合评分：准确率 * 0.4 + 盈利标准化 * 0.4 + 风险控制 * 0.2
    profit_normalized = np.tanh(profit / 1000)  # 归一化盈利（1000元为基准）
    risk_score = 1 - risk_metric  # 风险越低，得分越高
    
    score = (prediction_accuracy * 0.4 + 
             profit_normalized * 0.4 + 
             risk_score * 0.2)
    
    return float(score)

def fetch_data():
    """获取数据（使用多数据源管理器或回退方式）"""
    if multi_source_manager:
        try:
            df, source = multi_source_manager.fetch_data(days=7)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"   ⚠️  多数据源管理器获取失败: {e}")
    
    # 回退到原有方式（简化版）
    try:
        code_info = convert_stock_code(STOCK_CODE)
        df = fetch_akshare_5min(code_info, days=7)
        if df is not None and len(df) > 0:
            return df
    except:
        pass
    
    return None

# 只有在直接运行时才执行主程序
if __name__ == "__main__":
    print("🚀 开始V9实时预测循环...")
    print()

    # 运行状态
    consecutive_empty_count = 0
    last_action = None
    last_price_value = None
    current_balance = 20000.0
    shares_held = 0.0
    last_price = 0.0
    initial_balance = 20000.0

    # LSTM训练状态
    lstm_trained = False
    lstm_normalization_params = None

    while True:
    try:
        # 获取数据
        df = fetch_data()
        
        if df is None or len(df) == 0:
            consecutive_empty_count += 1
            print(f"⏸️  时间: {time.ctime()}, 未找到数据")
            time.sleep(60)
            continue
        
        consecutive_empty_count = 0
        df = df.sort_values('time')
        
        # 提取价格序列
        closes = df['close'].astype(float).values
        
        # 确保有足够的数据
        if len(closes) < 126:
            print(f"⚠️  数据不足（需要126条，实际{len(closes)}条）")
            time.sleep(60)
            continue
        
        # 构建V7模型观察向量（最后126条）
        obs = np.array(closes[-126:], dtype=np.float32)
        current_price = closes[-1]
        
        # PPO模型预测
        if model:
            action, _states = model.predict(obs, deterministic=True)
            operation = map_action_to_operation(action)
        else:
            operation = "持有"
        
        # LSTM/GRU预测（如果启用）
        lstm_prediction = None
        attention_weights = None
        if lstm_processor and len(closes) >= LSTM_SEQ_LENGTH:
            try:
                # 训练LSTM（如果还未训练）
                if not lstm_trained and len(closes) >= LSTM_SEQ_LENGTH * 2:
                    print("   📚 训练LSTM/GRU模型...")
                    
                    # 归一化数据
                    normalized_closes, norm_params = lstm_processor.normalize(closes)
                    lstm_normalization_params = norm_params
                    
                    # 创建序列
                    X, y = lstm_processor.create_sequences(normalized_closes)
                    
                    if len(X) > 0:
                        # 训练模型（快速训练，少量epoch）
                        lstm_processor.train(
                            X, y,
                            epochs=50,
                            batch_size=32,
                            learning_rate=0.001,
                            validation_split=0.2,
                            verbose=False
                        )
                        lstm_trained = True
                        print("   ✅ LSTM/GRU模型训练完成")
                
                # 进行预测
                if lstm_trained and lstm_normalization_params:
                    # 归一化最新序列
                    recent_sequence = closes[-LSTM_SEQ_LENGTH:]
                    normalized_seq = lstm_processor.normalize(recent_sequence, 
                                                             method=lstm_normalization_params['method'])[0]
                    
                    # 预测下一个价格
                    if LSTM_MODEL_TYPE == 'lstm_attention':
                        # 对于attention模型，使用predict方法获取注意力权重
                        seq_reshaped = normalized_seq.reshape(1, LSTM_SEQ_LENGTH, 1)
                        prediction_result = lstm_processor.predict(seq_reshaped, return_attention=True)
                        if isinstance(prediction_result, tuple):
                            prediction_norm = prediction_result[0][0, 0]
                            attention_weights = prediction_result[1][0]
                        else:
                            prediction_norm = prediction_result[0, 0]
                            attention_weights = None
                    else:
                        prediction_norm = lstm_processor.predict_next(normalized_seq)
                        attention_weights = None
                    
                    # 反归一化
                    lstm_prediction = lstm_processor.denormalize(
                        np.array([prediction_norm]), 
                        lstm_normalization_params
                    )[0]
                    
                    print(f"   📊 LSTM预测价格: {lstm_prediction:.2f} 元 (当前: {current_price:.2f} 元)")
                    
                    # 显示注意力权重（如果有）
                    if attention_weights is not None:
                        print(f"   🔍 注意力权重已计算")
            except Exception as e:
                print(f"   ⚠️  LSTM预测失败: {e}")
        
        # 计算技术指标
        indicator_summary = None
        if tech_indicators:
            try:
                calc_df = df.copy()
                if 'high' not in calc_df.columns:
                    calc_df['high'] = calc_df['close']
                if 'low' not in calc_df.columns:
                    calc_df['low'] = calc_df['close']
                if 'open' not in calc_df.columns:
                    calc_df['open'] = calc_df['close']
                
                calc_df = tech_indicators.calculate_all(calc_df)
                indicator_summary = tech_indicators.get_indicator_summary(calc_df)
            except Exception as e:
                print(f"   ⚠️  技术指标计算失败: {e}")
        
        # 动态参数优化（如果启用）
        if auto_learner and indicator_summary:
            try:
                # 计算当前性能（简化版）
                # 这里可以根据实际需求计算更复杂的性能指标
                prediction_accuracy = 0.5  # 占位符
                profit = (current_price - last_price) * shares_held if last_price > 0 else 0
                risk_metric = 0.3  # 占位符
                
                performance = calculate_performance_score(prediction_accuracy, profit, risk_metric)
                
                # 获取当前参数
                current_params = {
                    'kdj_period': TECHNICAL_INDICATOR_CONFIG.get('kdj_period', 9),
                    'rsi_period': TECHNICAL_INDICATOR_CONFIG.get('rsi_period', 14),
                    'macd_fast': TECHNICAL_INDICATOR_CONFIG.get('macd_fast', 12),
                    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
                    'lstm_num_layers': 2,
                    'learning_rate': 0.001
                }
                
                # 学习步骤
                next_params = auto_learner.learn_step(performance, current_params)
                
                # 更新参数（简化版，实际应用中可以更细致地更新）
                if abs(next_params.get('kdj_period', 9) - TECHNICAL_INDICATOR_CONFIG.get('kdj_period', 9)) > 1:
                    TECHNICAL_INDICATOR_CONFIG['kdj_period'] = int(next_params['kdj_period'])
                    if tech_indicators:
                        tech_indicators.kdj_period = int(next_params['kdj_period'])
                    print(f"   🔧 动态调整: KDJ周期 -> {int(next_params['kdj_period'])}")
            except Exception as e:
                print(f"   ⚠️  参数优化失败: {e}")
        
        # 显示结果
        print("=" * 70)
        print(f"✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}")
        print(f"   当前价格: {current_price:.2f} 元")
        print(f"   PPO预测: {operation}")
        if lstm_prediction:
            price_change_pct = ((lstm_prediction - current_price) / current_price) * 100
            print(f"   LSTM预测: {lstm_prediction:.2f} 元 ({price_change_pct:+.2f}%)")
        
        if indicator_summary:
            print(f"   技术指标: KDJ(K={indicator_summary.get('KDJ', {}).get('K', 0):.1f}), "
                  f"RSI={indicator_summary.get('RSI', 0):.1f}")
        
        print("=" * 70)
        print()
        
        # 更新状态
        last_action = operation
        last_price_value = current_price
        
        # 等待下一轮
        time.sleep(60)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断，正在退出...")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            print(traceback.format_exc())
            time.sleep(60)
            continue

    print("\n✅ V9程序已退出")
else:
    # 当作为模块导入时，只定义函数，不执行初始化代码
    pass

