"""
V10 实时预测系统
新增功能：
1. Transformer模型
2. 多模态数据处理
3. 实时数据可视化
4. 文本处理和全息动态模型
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

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==================== 导入模块 ====================

# 注意：不导入V9模块，避免触发V7/V9的初始化代码
# 如果需要calculate_performance_score功能，可以在V10中重新实现
V9_MODULES_AVAILABLE = False
# try:
#     from real_time_predict_v9 import calculate_performance_score
#     V9_MODULES_AVAILABLE = True
# except ImportError:
#     pass

# 导入独立模块（避免触发V7的初始化代码）
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

# 在V10中直接实现必要函数（避免导入V7文件触发初始化）
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
    if action == 0: return "卖出 100%"
    elif action == 1: return "卖出 50%"
    elif action == 2: return "卖出 25%"
    elif action == 3: return "持有"
    elif action == 4: return "买入 25%"
    elif action == 5: return "买入 50%"
    elif action == 6: return "买入 100%"
    else: return "未知动作"

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

# 简化的交易日志和持仓状态函数
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

def save_portfolio_state(stock_code, shares_held, current_balance, last_price, initial_balance):
    """保存持仓状态"""
    try:
        state = {
            'stock_code': stock_code,
            'shares_held': float(shares_held),
            'current_balance': float(current_balance),
            'last_price': float(last_price),
            'initial_balance': float(initial_balance),
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_assets': float(current_balance + shares_held * last_price)
        }
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
    """记录交易操作（简化版）"""
    try:
        import csv
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        op_type = "买入" if "买入" in operation else "卖出" if "卖出" in operation else "持有"
        op_percentage = operation.split()[-1] if "%" in operation else "0%"
        
        with open(TRADE_LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, date, time_str, stock_code, op_type, op_percentage,
                f"{current_price:.2f}", "", "", "0.00", "0.00",
                f"{shares_held:.2f}", f"{current_balance:.2f}", f"{total_assets:.2f}",
                status, note
            ])
        return True
    except:
        return False

# 导入V10新模块
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

# 导入其他模块
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ==================== 配置参数 ====================

MODEL_PATH = "ppo_stock_v7.zip"
STOCK_CODE = 'sh.600730'
LLM_PROVIDER = "deepseek"
ENABLE_LLM = True
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d')

# V10新功能配置
ENABLE_TRANSFORMER = True
ENABLE_MULTIMODAL = True
ENABLE_VISUALIZATION = True
ENABLE_HOLOGRAPHIC = True

# Transformer配置
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_MAX_SEQ_LEN = 100

# 可视化配置
VISUALIZATION_PORT = 8081  # 改为8081避免与8080冲突
VISUALIZATION_OUTPUT_DIR = "visualization_output"

# 全息模型配置
HOLOGRAPHIC_MEMORY_SIZE = 1000

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

# 版本标识 - 确保运行的是V10
print("\n" + "=" * 70)
print("V10 实时预测系统 - Transformer + 多模态 + 可视化 + 全息动态模型")
print("=" * 70)
print("📌 V10 新功能:")
print("   - Transformer 深度学习模型")
print("   - 多模态数据处理（时间序列+文本）")
print("   - 实时数据可视化（端口: 8081）")
print("   - 文本处理和全息动态模型")
print("=" * 70)
print("⚠️  版本标识: 这是 V10 版本，不是 V7！")
print("=" * 70 + "\n")

# 初始化技术指标计算器
tech_indicators = None
if TECHNICAL_INDICATORS_AVAILABLE:
    try:
        tech_indicators = TechnicalIndicators(**TECHNICAL_INDICATOR_CONFIG)
        print("✅ 技术指标计算器初始化成功")
        print(f"   KDJ参数: 周期={TECHNICAL_INDICATOR_CONFIG['kdj_period']}, "
              f"慢速={TECHNICAL_INDICATOR_CONFIG['kdj_slow_period']}, "
              f"快速={TECHNICAL_INDICATOR_CONFIG['kdj_fast_period']}")
        print(f"   RSI周期: {TECHNICAL_INDICATOR_CONFIG['rsi_period']}")
    except Exception as e:
        print(f"⚠️  技术指标计算器初始化失败: {e}")

# 初始化多数据源管理器
multi_source_manager_v10 = None
if MULTI_DATA_SOURCE_AVAILABLE:
    try:
        multi_source_manager_v10 = MultiDataSourceManager(stock_code=STOCK_CODE)
        print("✅ 多数据源管理器初始化成功")
    except Exception as e:
        print(f"⚠️  多数据源管理器初始化失败: {e}")

# 初始化Transformer模型
transformer_model = None
if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER:
    try:
        transformer_model = TransformerPredictor(
            input_size=1,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_encoder_layers=TRANSFORMER_NUM_LAYERS,
            num_decoder_layers=TRANSFORMER_NUM_LAYERS,
            dim_feedforward=256,
            dropout=0.1,
            output_size=1,
            max_seq_len=TRANSFORMER_MAX_SEQ_LEN,
            use_gpu=False
        )
        print(f"✅ Transformer模型初始化成功")
    except Exception as e:
        print(f"⚠️  Transformer初始化失败: {e}")

# 初始化多模态处理器
multimodal_processor = None
if MULTIMODAL_AVAILABLE and ENABLE_MULTIMODAL:
    try:
        multimodal_processor = MultimodalDataProcessor(
            text_max_length=512,
            use_bert=False,  # 可以根据需要启用BERT
            fusion_method='attention'
        )
        print("✅ 多模态数据处理器初始化成功")
    except Exception as e:
        print(f"⚠️  多模态处理器初始化失败: {e}")

# 初始化实时可视化器
visualizer = None
web_visualization = None
if VISUALIZATION_AVAILABLE and ENABLE_VISUALIZATION:
    try:
        visualizer = RealTimeVisualizer(
            data_window_size=100,
            update_interval=5.0,
            output_dir=VISUALIZATION_OUTPUT_DIR
        )
        print("✅ 实时可视化器初始化成功")
        
        # 启动Web可视化服务器
        if VISUALIZATION_AVAILABLE and ENABLE_VISUALIZATION:
            try:
                # 检查端口是否被占用
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', VISUALIZATION_PORT))
                sock.close()
                
                if result == 0:
                    print(f"⚠️  端口 {VISUALIZATION_PORT} 已被占用，Web服务器将不启动")
                    print(f"   💡 提示: 请关闭占用该端口的程序，或修改 VISUALIZATION_PORT 配置")
                    web_visualization = None
                else:
                    web_visualization = WebVisualizationServer(visualizer, port=VISUALIZATION_PORT)
                    web_visualization.start(host='127.0.0.1', debug=False)
                    print(f"   🌐 Web可视化服务器已启动: http://127.0.0.1:{VISUALIZATION_PORT}")
            except ImportError:
                print(f"⚠️  无法检查端口，Web服务器可能无法启动")
                web_visualization = None
            except Exception as e:
                print(f"⚠️  Web可视化服务器启动失败: {e}")
                print(f"   💡 提示: 端口 {VISUALIZATION_PORT} 可能已被占用，或需要安装 Flask")
                web_visualization = None
        else:
            web_visualization = None
    except Exception as e:
        print(f"⚠️  可视化器初始化失败: {e}")

# 初始化全息动态模型
holographic_model = None
if HOLOGRAPHIC_AVAILABLE and ENABLE_HOLOGRAPHIC:
    try:
        holographic_model = HolographicDynamicModel(
            memory_size=HOLOGRAPHIC_MEMORY_SIZE,
            enable_text_analysis=True,
            enable_memory=True
        )
        print("✅ 全息动态模型初始化成功")
    except Exception as e:
        print(f"⚠️  全息动态模型初始化失败: {e}")

# 初始化LLM
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
if VISUALIZATION_AVAILABLE:  # 使用VISUALIZATION_AVAILABLE作为LLM_INTERPRETER的代理检查
    try:
        from llm_indicator_interpreter import LLMIndicatorInterpreter
        if llm_agent:
            llm_interpreter = LLMIndicatorInterpreter(
                llm_agent=llm_agent,
                enable_cache=True
            )
            print("✅ LLM指标解释器初始化成功")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  LLM指标解释器初始化失败: {e}")

# 加载PPO模型
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

# ==================== 主循环 ====================

print("\n" + "=" * 70)
print("🚀 开始 V10 实时预测循环...")
print("=" * 70)
print("⚠️  重要提示: 这是 V10 版本，包含 Transformer、多模态、可视化等功能")
print("=" * 70 + "\n")

# 运行状态
current_balance = 20000.0
shares_held = 0.0
last_price = 0.0
initial_balance = 20000.0
last_action = None

# 模型训练状态
transformer_trained = False
transformer_normalization_params = None

# 加载持仓状态（在主循环之前）
portfolio_state = load_portfolio_state()
if portfolio_state:
    if portfolio_state.get('stock_code') == STOCK_CODE:
        current_balance = portfolio_state.get('current_balance', 20000.0)
        shares_held = portfolio_state.get('shares_held', 0.0)
        last_price = portfolio_state.get('last_price', 0.0)
        initial_balance = portfolio_state.get('initial_balance', 20000.0)
        print(f"✅ 已加载持仓状态: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元")
    else:
        print(f"⚠️  持仓状态文件中的股票代码不匹配，使用默认状态")

# 启动可视化自动更新
if visualizer:
    visualizer.start_auto_update()

# 示例文本数据（可以从新闻API、社交媒体等获取）
sample_texts = [
    "该股票今日表现强势，市场看好其未来发展前景",
    "受利空消息影响，股价出现下跌",
    "公司业绩超预期，投资者信心增强"
]
text_index = 0

while True:
    try:
        # 获取数据（使用多数据源管理器或直接获取）
        df = None
        if multi_source_manager_v10:
            try:
                df, source = multi_source_manager_v10.fetch_data(days=7)
                if df is not None and len(df) > 0:
                    print(f"   📊 数据来源: {source}")
            except Exception as e:
                print(f"   ⚠️  多数据源管理器获取失败: {e}")
        
        if df is None or len(df) == 0:
            # 回退到直接获取
            try:
                code_info = convert_stock_code(STOCK_CODE)
                df = fetch_akshare_5min(code_info, days=7)
            except Exception as e:
                print(f"   ⚠️  数据获取失败: {e}")
                time.sleep(60)
                continue
        
        if df is None or len(df) == 0:
            print(f"⏸️  时间: {time.ctime()}, 未找到数据")
            time.sleep(60)
            continue
        
        df = df.sort_values('time')
        closes = df['close'].astype(float).values
        
        if len(closes) < 126:
            print(f"⚠️  数据不足（需要126条，实际{len(closes)}条）")
            time.sleep(60)
            continue
        
        # 构建PPO观察向量
        obs = np.array(closes[-126:], dtype=np.float32)
        current_price = closes[-1]
        volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
        
        # PPO模型预测
        ppo_operation = "持有"
        if ppo_model:
            action, _states = ppo_model.predict(obs, deterministic=True)
            ppo_operation = map_action_to_operation(action)
        
        # Transformer预测
        transformer_prediction = None
        if transformer_model and len(closes) >= TRANSFORMER_MAX_SEQ_LEN:
            try:
                # 训练Transformer（如果还未训练）
                if not transformer_trained and len(closes) >= TRANSFORMER_MAX_SEQ_LEN * 2:
                    print("   📚 训练Transformer模型...")
                    
                    # 归一化
                    normalized_closes, norm_params = transformer_model.normalize(closes)
                    transformer_normalization_params = norm_params
                    
                    # 创建序列
                    X_list, y_list = [], []
                    for i in range(TRANSFORMER_MAX_SEQ_LEN, len(normalized_closes)):
                        X_list.append(normalized_closes[i-TRANSFORMER_MAX_SEQ_LEN:i])
                        y_list.append(normalized_closes[i])
                    
                    if len(X_list) > 0:
                        X = np.array(X_list).reshape(len(X_list), TRANSFORMER_MAX_SEQ_LEN, 1)
                        y = np.array(y_list).reshape(len(y_list), 1)
                        
                        transformer_model.train(
                            X, y,
                            epochs=50,
                            batch_size=32,
                            learning_rate=0.001,
                            validation_split=0.2,
                            verbose=False
                        )
                        transformer_trained = True
                        print("   ✅ Transformer模型训练完成")
                
                # 进行预测
                if transformer_trained and transformer_normalization_params:
                    recent_seq = closes[-TRANSFORMER_MAX_SEQ_LEN:]
                    normalized_seq, _ = transformer_model.normalize(
                        recent_seq, 
                        method=transformer_normalization_params['method']
                    )
                    
                    # 预测下一个值
                    if len(normalized_seq) >= transformer_model.max_seq_len:
                        prediction_norm = transformer_model.predict_next(normalized_seq)
                        transformer_prediction = transformer_model.denormalize(
                            np.array([prediction_norm]),
                            transformer_normalization_params
                        )[0]
                    else:
                        # 序列长度不足，填充
                        padding = np.full(transformer_model.max_seq_len - len(normalized_seq), normalized_seq[0])
                        full_seq = np.concatenate([padding, normalized_seq])
                        prediction_norm = transformer_model.predict_next(full_seq)
                        transformer_prediction = transformer_model.denormalize(
                            np.array([prediction_norm]),
                            transformer_normalization_params
                        )[0]
                    
                    print(f"   📊 Transformer预测: {transformer_prediction:.2f} 元")
            except Exception as e:
                print(f"   ⚠️  Transformer预测失败: {e}")
        
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
        
        # 多模态处理（时间序列+文本）
        multimodal_result = None
        current_text = sample_texts[text_index % len(sample_texts)] if multimodal_processor else None
        if multimodal_processor:
            try:
                multimodal_result = multimodal_processor.process(
                    time_series_data=closes[-60:],
                    text_data=current_text
                )
                print(f"   📝 文本情感: {multimodal_result['sentiment']['sentiment_score']:.2f}")
            except Exception as e:
                print(f"   ⚠️  多模态处理失败: {e}")
        
        # 全息动态模型处理
        holographic_result = None
        if holographic_model:
            try:
                # 获取市场情报
                market_intel = None
                if llm_agent:
                    try:
                        latest_date = df['date'].iloc[-1] if 'date' in df.columns else datetime.date.today().strftime('%Y-%m-%d')
                        market_intel = llm_agent.get_market_intelligence(latest_date, force_refresh=False)
                    except:
                        pass
                
                holographic_result = holographic_model.process(
                    time_series_data=closes[-60:],
                    text_data=current_text,
                    technical_indicators=indicator_summary,
                    market_intelligence=market_intel
                )
                
                signal = holographic_result.get('comprehensive_signal', {})
                print(f"   🌟 全息模型信号: {signal.get('signal', 'unknown')} "
                      f"(置信度: {signal.get('confidence', 0):.2f})")
            except Exception as e:
                print(f"   ⚠️  全息模型处理失败: {e}")
        
        # 更新可视化
        if visualizer:
            try:
                # 准备指标字典（用于可视化）
                indicators_dict = {}
                if indicator_summary:
                    if 'KDJ' in indicator_summary:
                        indicators_dict['KDJ_K'] = indicator_summary['KDJ'].get('K', 0)
                        indicators_dict['KDJ_D'] = indicator_summary['KDJ'].get('D', 0)
                    if 'RSI' in indicator_summary:
                        indicators_dict['RSI'] = indicator_summary['RSI']
                    if 'OBV' in indicator_summary:
                        indicators_dict['OBV_Ratio'] = indicator_summary['OBV'].get('OBV_Ratio', 1.0)
                    if 'MACD' in indicator_summary:
                        indicators_dict['MACD'] = indicator_summary['MACD'].get('MACD', 0)
                
                visualizer.add_data_point(
                    price=current_price,
                    volume=volume,
                    indicators=indicators_dict if indicators_dict else None,
                    prediction=transformer_prediction
                )
            except Exception as e:
                print(f"   ⚠️  可视化更新失败: {e}")
        
        # LLM指标解释（如果可用）
        if llm_interpreter and indicator_summary:
            try:
                indicator_interpretation = llm_interpreter.interpret_indicators(
                    indicator_summary,
                    STOCK_CODE,
                    current_price,
                    force_refresh=False
                )
                if indicator_interpretation:
                    print()
                    print(llm_interpreter.format_interpretation(indicator_interpretation))
            except Exception as e:
                pass  # 静默失败
        
        # 显示结果
        print("=" * 70)
        print(f"✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}")
        print(f"   当前价格: {current_price:.2f} 元")
        print(f"   PPO预测: {ppo_operation}")
        if transformer_prediction:
            change_pct = ((transformer_prediction - current_price) / current_price) * 100
            print(f"   Transformer预测: {transformer_prediction:.2f} 元 ({change_pct:+.2f}%)")
        
        if indicator_summary:
            kdj = indicator_summary.get('KDJ', {})
            print(f"   技术指标: KDJ(K={kdj.get('K', 0):.1f}), "
                  f"RSI={indicator_summary.get('RSI', 0):.1f}")
        
        if holographic_result:
            signal = holographic_result.get('comprehensive_signal', {})
            print(f"   全息模型: {signal.get('signal', 'unknown')} "
                  f"(置信度: {signal.get('confidence', 0):.2f})")
        
        if web_visualization and web_visualization.running:
            print(f"   📊 可视化: http://127.0.0.1:{VISUALIZATION_PORT}")
        elif visualizer:
            print(f"   📊 可视化: 图表已保存到 {VISUALIZATION_OUTPUT_DIR}/")
        
        print("=" * 70)
        print()
        
        # 更新状态
        last_action = ppo_operation
        last_price = current_price
        text_index += 1
        
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

# 清理
if visualizer:
    visualizer.stop_auto_update()

print("\n✅ V10程序已退出")

