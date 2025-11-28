import os
import sys
import random
import warnings
import numpy as np
import csv  # 用于记录交易日志
import time
import pandas as pd
import datetime  # 用于日期计算
import json  # 用于保存和加载持仓状态
import threading

# 可选的图形化持仓编辑（基于 Flask 简单网页）
try:
    from flask import Flask, request, redirect
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# 抑制 Gym 相关的废弃警告（已使用 Gymnasium）
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# 临时重定向 stderr 以捕获 gym 的警告输出
class SuppressGymWarning:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = self
        
    def __exit__(self, *args):
        sys.stderr = self.original_stderr
        
    def write(self, text):
        if 'Gym has been unmaintained' in text or 'Please upgrade to Gymnasium' in text:
            return  # 忽略这些警告
        self.original_stderr.write(text)
        
    def flush(self):
        self.original_stderr.flush()


# 在导入可能触发 gym 的包之前抑制警告
with SuppressGymWarning():
    from stable_baselines3 import PPO
    import gymnasium as gym  # 使用 Gymnasium 替换 Gym 以避免警告


# 导入 LLM 市场情报模块
try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入 llm_market_intelligence 模块，将仅使用技术指标")
    LLM_AVAILABLE = False

# 尝试导入数据源
DATA_SOURCE = None
TUSHARE_AVAILABLE = False
AKSHARE_AVAILABLE = False
BAOSTOCK_AVAILABLE = False

# 尝试导入 Tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    pass

# 尝试导入 AkShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    pass

# 尝试导入 baostock（备用）
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    pass

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# ==================== 配置参数 ====================
MODEL_PATH = "ppo_stock_v7.zip"  # 使用 V7 模型
STOCK_CODE = 'sh.600036'  # 股票代码
LLM_PROVIDER = "deepseek"  # LLM 提供商：deepseek 或 grok
ENABLE_LLM = True  # 是否启用 LLM 市场情报（作为参考）
DEEPSEEK_API_KEY = "sk-167914945f7945d498e09a7f186c101d"  # DeepSeek API 密钥

# 数据源配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')  # Tushare token（从环境变量或这里设置）
PREFER_REALTIME = True  # 是否优先使用实时数据源

# 交易日志文件
TRADE_LOG_FILE = "trade_log.csv"  # 交易记录文件
TRADE_SUMMARY_FILE = "trade_summary.txt"  # 操作汇总文件
PORTFOLIO_STATE_FILE = "portfolio_state.json"  # 持仓状态文件

# 建议价格偏移配置（相对于当前价格的基础偏移）
# 例如：基础设置为买入 -0.5%，卖出 +0.5%，再根据波动率动态放大/缩小
BASE_SUGGESTED_BUY_OFFSET = -0.005   # 基础买入偏移：-0.5%
BASE_SUGGESTED_SELL_OFFSET = 0.005   # 基础卖出偏移：+0.5%

# 图形化持仓编辑配置
ENABLE_WEB_EDITOR = True          # 是否启用网页持仓编辑
WEB_EDITOR_PORT = 5001           # 本地网页端口
WEB_EDITOR_HOST = "127.0.0.1"    # 仅本机访问


def get_dynamic_offsets(price_volatility):
    """
    根据价格波动率动态调整建议价格偏移
    price_volatility: 价格波动率（百分比，例如 0.07 表示 0.07%）
    规则（举例）:
        - 波动率 < 0.2%  : 偏移缩小一半（更容易成交）
        - 0.2% ~ 0.5%    : 使用基础偏移
        - > 0.5%         : 偏移放大一倍（给更多空间）
    """
    if price_volatility is None:
        return BASE_SUGGESTED_BUY_OFFSET, BASE_SUGGESTED_SELL_OFFSET

    vol = abs(price_volatility)  # 百分比
    if vol < 0.2:
        factor = 0.5
    elif vol < 0.5:
        factor = 1.0
    else:
        factor = 2.0

    return BASE_SUGGESTED_BUY_OFFSET * factor, BASE_SUGGESTED_SELL_OFFSET * factor


# ==================== 初始化 ====================
print("=" * 70)
print("V7 实时预测系统 - V7 模型 + LLM 情报参考 + 实时数据源 + 操作记录 + 图形化持仓管理")
print("=" * 70)
print("📌 模型: V7 (126维价格序列)")
print("📌 LLM 情报: 作为决策参考，不输入模型")
print("📌 数据源: 支持 Tushare/AkShare/baostock（自动选择）")
print("📌 操作记录: 自动记录买入/卖出操作，支持汇总查看")
print("📌 持仓管理: 支持网页实时修改持仓，无需停止脚本")
print("=" * 70)

# 初始化数据源
if TUSHARE_AVAILABLE and TUSHARE_TOKEN:
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        DATA_SOURCE = "tushare"
        print("✅ 数据源: Tushare（支持实时数据）")
    except Exception as e:
        print(f"⚠️  Tushare 初始化失败: {e}")
        TUSHARE_AVAILABLE = False

if DATA_SOURCE is None and AKSHARE_AVAILABLE:
    DATA_SOURCE = "akshare"
    print("✅ 数据源: AkShare（支持实时数据）")

if DATA_SOURCE is None and BAOSTOCK_AVAILABLE:
    bs.login()
    DATA_SOURCE = "baostock"
    print("✅ 数据源: baostock（免费，但有延迟，不支持实时数据）")

if DATA_SOURCE is None:
    raise Exception("未找到可用的数据源！请安装 tushare、akshare 或 baostock")

# 加载 V7 模型（在抑制警告的上下文中）
with SuppressGymWarning():
    if not os.path.exists(MODEL_PATH):
        # 尝试查找 V7 模型
        possible_models = [
            "ppo_stock_v7.zip",
            "models_v7/best/best_model.zip",
            "ppo_stock_v7_2500000_steps.zip",
            "ppo_stock_v7_2400000_steps.zip",
            "ppo_stock_v7_2300000_steps.zip",
        ]
        for model_file in possible_models:
            if os.path.exists(model_file):
                MODEL_PATH = model_file
                break
        else:
            raise FileNotFoundError(f"未找到 V7 模型文件，请检查: {MODEL_PATH}")
    
    model = PPO.load(MODEL_PATH)
    
    # 验证模型版本（必须是 126 维）
    obs_shape = model.observation_space.shape
    if len(obs_shape) == 1 and obs_shape[0] == 126:
        print(f"✅ V7 模型加载成功！")
        print(f"   模型路径: {MODEL_PATH}")
        print(f"   观察空间: {obs_shape} (V7 标准: 126维价格序列)")
    else:
        print(f"⚠️  警告: 模型观察空间为 {obs_shape}，不是标准的 V7 模型 (126维)")
        print(f"   将继续使用，但可能不是最优配置")
    
    print(f"   动作空间: {model.action_space}")

# 初始化 LLM 市场情报代理（仅用于参考信息）
llm_agent = None
if LLM_AVAILABLE and ENABLE_LLM:
    try:
        # 设置 API 密钥到环境变量（如果未设置）
        if LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
            os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY
        
        llm_agent = MarketIntelligenceAgent(
            provider=LLM_PROVIDER,
            api_key=DEEPSEEK_API_KEY if LLM_PROVIDER == "deepseek" else None,
            enable_cache=True
        )
        
        # 验证 API 密钥状态
        api_key_status = "✅ 已配置" if (hasattr(llm_agent, 'api_key') and llm_agent.api_key) else "❌ 未配置"
        mock_mode_status = "❌ 模拟模式" if (hasattr(llm_agent, 'mock_mode') and llm_agent.mock_mode) else "✅ 真实 API 模式"
        
        print(f"✅ LLM 市场情报代理初始化成功！")
        print(f"   提供商: {LLM_PROVIDER.upper()}")
        print(f"   API 密钥: {api_key_status}")
        print(f"   运行模式: {mock_mode_status}")
        print(f"   用途: 决策参考信息（不输入模型）")
        print(f"   缓存: 已启用")
        
        if hasattr(llm_agent, 'mock_mode') and llm_agent.mock_mode:
            print(f"   ⚠️  警告: 当前为模拟模式，API 密钥可能未正确配置")
            print(f"   💡 提示: 请检查 DEEPSEEK_API_KEY 环境变量或代码中的 API 密钥配置")
            print(f"   🔑 API 密钥状态: {'已设置' if hasattr(llm_agent, 'api_key') and llm_agent.api_key else '未设置'}")
    except Exception as e:
        print(f"⚠️  LLM 初始化失败: {e}")
        print("   将仅显示模型预测，无市场情报参考")
        llm_agent = None
else:
    print("ℹ️  LLM 市场情报未启用")

print("=" * 70)
print()

# ==================== 辅助函数 ====================

# 检查是否是交易日（周一到周五）
def is_trading_day(date=None):
    """检查指定日期是否是交易日（周一到周五）"""
    if date is None:
        date = datetime.date.today()
    return date.weekday() < 5  # 0-4 表示周一到周五


# 获取最近的交易日
def get_recent_trading_date(days_back=0):
    """获取最近的交易日，如果今天不是交易日，则返回最近的交易日"""
    current_date = datetime.date.today() - datetime.timedelta(days=days_back)
    # 如果今天不是交易日，往前找最近的交易日（最多往前找7天）
    for i in range(8):
        check_date = current_date - datetime.timedelta(days=i)
        if is_trading_day(check_date):
            return check_date
    return current_date  # 如果找不到，返回原日期


# 检查是否是交易时间（9:30-15:00）
def is_trading_time():
    """检查当前是否是交易时间（9:30-15:00）"""
    now = datetime.datetime.now()
    current_time = now.time()
    # 上午：9:30-11:30，下午：13:00-15:00
    morning_start = datetime.time(9, 30)
    morning_end = datetime.time(11, 30)
    afternoon_start = datetime.time(13, 0)
    afternoon_end = datetime.time(15, 0)
    
    return (morning_start <= current_time <= morning_end) or \
           (afternoon_start <= current_time <= afternoon_end)


# 转换股票代码格式
def convert_stock_code(code):
    """
    转换股票代码格式
    baostock: sh.600036
    tushare: 600036.SH
    akshare: 600036 (需要判断市场)
    """
    if '.' in code:
        market, num = code.split('.')
        return {
            'baostock': code,
            'tushare': f"{num}.{market.upper()}",
            'akshare': num,
            'market': 'sh' if market == 'sh' else 'sz'
        }
    else:
        # 假设是6位数字代码
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


# 使用 Tushare 获取5分钟K线数据
def fetch_tushare_5min(code_info, days=7):
    """使用 Tushare 获取5分钟K线数据"""
    try:
        ts_code = code_info['tushare']
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        # Tushare 获取5分钟K线
        # 注意：stk_mins 需要积分，如果没有积分可以使用 daily 接口
        try:
            df = pro.stk_mins(
                ts_code=ts_code,
                freq='5min',
                start_date=start_date + '0930',
                end_date=end_date + '1500'
            )
        except:
            # 如果没有积分，尝试使用日线数据（然后模拟5分钟数据）
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                # 将日线数据转换为5分钟格式（简化处理）
                df = df.rename(columns={'trade_date': 'date', 'close': 'close', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                df['time'] = df['date'] + '15000000'  # 使用收盘时间
                return df[['date', 'time', 'close', 'volume']]
            return None
        
        if df is None or len(df) == 0:
            return None
        
        # 转换列名
        if 'trade_time' in df.columns:
            df = df.rename(columns={
                'trade_time': 'time',
                'close': 'close',
                'vol': 'volume'
            })
        elif 'time' not in df.columns:
            # 如果没有 time 列，尝试其他列名
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'time'})
        
        # 提取日期
        if 'time' in df.columns:
            if isinstance(df['time'].iloc[0], str):
                # 如果 time 是字符串，尝试解析
                try:
                    df['time'] = pd.to_datetime(df['time'])
                except:
                    pass
            if pd.api.types.is_datetime64_any_dtype(df['time']):
                df['date'] = df['time'].dt.strftime('%Y-%m-%d')
                df['time'] = df['time'].dt.strftime('%Y%m%d%H%M%S')
        
        # 只保留需要的列
        if 'date' in df.columns and 'time' in df.columns:
            df = df[['date', 'time', 'close', 'volume']]
        else:
            return None
        
        return df
    except Exception as e:
        print(f"   [Tushare错误] {e}")
        import traceback
        print(f"   [详细] {traceback.format_exc()}")
        return None


# 使用 AkShare 获取5分钟K线数据
def fetch_akshare_5min(code_info, days=7):
    """使用 AkShare 获取5分钟K线数据"""
    try:
        symbol = code_info['akshare']
        market = code_info['market']
        
        # AkShare 获取5分钟K线
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        # 尝试使用股票分钟K线接口
        try:
            # 获取5分钟K线数据
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="5",
                adjust="qfq",
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) == 0:
                # 如果失败，尝试使用日线数据
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if df is not None and len(df) > 0:
                    # 将日线数据转换为5分钟格式（简化处理）
                    df = df.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['time'] = df['date'] + '15000000'  # 使用收盘时间
                    return df[['date', 'time', 'close', 'volume']]
                return None
            
            # 转换列名（AkShare 返回中文列名）
            column_mapping = {
                '时间': 'time',
                '收盘': 'close',
                '成交量': 'volume',
                '日期': 'date'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # 如果没有 time 列，尝试从其他列提取
            if 'time' not in df.columns:
                if 'date' in df.columns:
                    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
                else:
                    return None
            
            # 提取日期
            if 'time' in df.columns:
                if isinstance(df['time'].iloc[0], str):
                    # 如果 time 是字符串，尝试解析
                    try:
                        df['time'] = pd.to_datetime(df['time'])
                    except:
                        pass
                if pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['date'] = df['time'].dt.strftime('%Y-%m-%d')
                    df['time'] = df['time'].dt.strftime('%Y%m%d%H%M%S')
            
            # 只保留需要的列
            if 'date' in df.columns and 'time' in df.columns:
                df = df[['date', 'time', 'close', 'volume']]
            else:
                return None
            
            return df
        except Exception as e:
            print(f"   [AkShare错误] {e}")
            import traceback
            print(f"   [详细] {traceback.format_exc()}")
            return None
    except Exception as e:
        print(f"   [AkShare错误] {e}")
        import traceback
        print(f"   [详细] {traceback.format_exc()}")
        return None


# 使用 baostock 获取5分钟K线数据（备用）
def fetch_baostock_5min(code_info, days=7):
    """使用 baostock 获取5分钟K线数据（备用）"""
    try:
        bs_code = code_info['baostock']
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(
            bs_code, 
            "date,time,close,volume", 
            start_date=start_date, 
            end_date=end_date, 
            frequency='5', 
            adjustflag='3'
        )
        
        if rs.error_code != '0':
            return None
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        return df
    except Exception as e:
        print(f"   [baostock错误] {e}")
        return None


# 获取股票数据（多数据源支持）
def fetch_data_with_retry(max_retries=3, extend_days=0, try_today=True):
    """
    获取股票数据，支持多个数据源
    优先级：Tushare > AkShare > baostock
    支持扩展日期范围
    """
    code_info = convert_stock_code(STOCK_CODE)
    
    for attempt in range(max_retries):
        try:
            # 计算日期范围
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=7 + extend_days)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            # 优先使用 Tushare
            if DATA_SOURCE == "tushare" and TUSHARE_AVAILABLE:
                df = fetch_tushare_5min(code_info, days=7 + extend_days)
                if df is not None and len(df) > 0:
                    return df
            
            # 使用 AkShare
            if DATA_SOURCE == "akshare" and AKSHARE_AVAILABLE:
                df = fetch_akshare_5min(code_info, days=7 + extend_days)
                if df is not None and len(df) > 0:
                    return df
            
            # 使用 baostock（备用）
            if BAOSTOCK_AVAILABLE:
                df = fetch_baostock_5min(code_info, days=7 + extend_days)
                if df is not None and len(df) > 0:
                    return df
            
            # 如果都失败，等待后重试
            if attempt < max_retries - 1:
                print(f"尝试 {attempt+1}/{max_retries} 失败，等待重试...")
                time.sleep(5 + random.uniform(0, 5))
            else:
                raise Exception("所有数据源都获取失败")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                time.sleep(5 + random.uniform(0, 5))
            else:
                raise Exception(f"数据获取失败，已达最大重试次数: {e}")
    
    raise Exception("数据获取失败，已达最大重试次数")


# 动作映射函数（V7 模型：7个动作，根据 Discrete(7) 调整）
def map_action_to_operation(action):
    """将动作映射到具体操作（V7 模型，Discrete(7)）"""
    if action == 0: return "卖出 100%"
    elif action == 1: return "卖出 50%"
    elif action == 2: return "卖出 25%"
    elif action == 3: return "持有"
    elif action == 4: return "买入 25%"
    elif action == 5: return "买入 50%"
    elif action == 6: return "买入 100%"
    else: return "未知动作"


# 格式化市场情报显示（详细版）
def format_intelligence_detailed(intelligence):
    """格式化市场情报信息用于显示（详细版，包含所有7个维度）"""
    if not intelligence:
        return ""
    
    lines = []
    lines.append("   " + "=" * 64)
    lines.append("   📊 LLM 市场情报参考（决策辅助信息）")
    lines.append("   " + "=" * 64)
    
    # 1. 宏观经济数据
    macro_score = intelligence.get('macro_economic_score', 0)
    macro_icon = "📈" if macro_score > 0.1 else "📉" if macro_score < -0.1 else "➡️"
    lines.append(f"   1️⃣  宏观经济评分 {macro_icon}: {macro_score:+.3f}")
    lines.append(f"      └─ GDP、CPI、利率政策综合影响")
    if macro_score > 0.2:
        lines.append(f"      └─ 💡 宏观经济环境良好，有利于市场上涨")
    elif macro_score < -0.2:
        lines.append(f"      └─ ⚠️  宏观经济环境偏弱，需谨慎")
    
    # 2. 新闻和舆情分析
    sentiment_score = intelligence.get('market_sentiment_score', 0)
    sentiment_icon = "😊" if sentiment_score > 0.2 else "😐" if sentiment_score > -0.2 else "😟"
    lines.append(f"   2️⃣  市场情绪评分 {sentiment_icon}: {sentiment_score:+.3f}")
    lines.append(f"      └─ 新闻热点、负面消息、投资者情绪")
    if sentiment_score > 0.3:
        lines.append(f"      └─ 💡 市场情绪积极，正面消息较多")
    elif sentiment_score < -0.3:
        lines.append(f"      └─ ⚠️  市场情绪悲观，负面消息较多")
    
    # 3. 市场情绪指标（VIX）
    vix_level = intelligence.get('vix_level', 20)
    risk_level = intelligence.get('risk_level', 0.5)
    vix_icon = "🔴" if vix_level > 25 else "🟡" if vix_level > 18 else "🟢"
    lines.append(f"   3️⃣  恐慌指数 VIX {vix_icon}: {vix_level:.2f}")
    lines.append(f"      └─ 风险等级: {risk_level:.3f} ({'高风险' if risk_level > 0.7 else '中风险' if risk_level > 0.4 else '低风险'})")
    if vix_level > 25:
        lines.append(f"      └─ ⚠️  恐慌指数较高，市场波动可能加大")
    elif vix_level < 15:
        lines.append(f"      └─ 💡 恐慌指数较低，市场相对稳定")
    
    # 4. 资金流向数据
    capital_flow = intelligence.get('capital_flow_score', 0)
    flow_icon = "💰" if capital_flow > 0.2 else "💸" if capital_flow < -0.2 else "💵"
    lines.append(f"   4️⃣  资金流向评分 {flow_icon}: {capital_flow:+.3f}")
    lines.append(f"      └─ 外资、融资融券、北向资金流向")
    if capital_flow > 0.3:
        lines.append(f"      └─ 💡 资金净流入，市场资金面充裕")
    elif capital_flow < -0.3:
        lines.append(f"      └─ ⚠️  资金净流出，市场资金面紧张")
    
    # 5. 政策变化信息
    policy_impact = intelligence.get('policy_impact_score', 0)
    policy_icon = "📜" if policy_impact > 0.1 else "📋" if policy_impact > -0.1 else "📄"
    lines.append(f"   5️⃣  政策影响评分 {policy_icon}: {policy_impact:+.3f}")
    lines.append(f"      └─ 货币/财政/监管政策影响")
    if policy_impact > 0.2:
        lines.append(f"      └─ 💡 政策环境利好，支持市场发展")
    elif policy_impact < -0.2:
        lines.append(f"      └─ ⚠️  政策环境偏紧，需关注政策变化")
    
    # 6. 国际市场联动
    intl_corr = intelligence.get('international_correlation', 0.5)
    intl_icon = "🌍" if intl_corr > 0.6 else "🌎" if intl_corr > 0.4 else "🌏"
    lines.append(f"   6️⃣  国际联动系数 {intl_icon}: {intl_corr:.3f}")
    lines.append(f"      └─ 与美股、港股相关性")
    if intl_corr > 0.7:
        lines.append(f"      └─ 💡 与国际市场联动性强，关注海外市场走势")
    elif intl_corr < 0.4:
        lines.append(f"      └─ 💡 与国际市场联动性弱，主要受国内因素影响")
    
    # 7. 突发事件应对
    emergency_impact = intelligence.get('emergency_impact_score', 0)
    emergency_icon = "🚨" if abs(emergency_impact) > 0.3 else "⚡" if abs(emergency_impact) > 0.1 else "✅"
    lines.append(f"   7️⃣  突发事件影响 {emergency_icon}: {emergency_impact:+.3f}")
    lines.append(f"      └─ 地缘政治、疫情、自然灾害等")
    if emergency_impact < -0.5:
        lines.append(f"      └─ 🚨 重大负面事件，需高度警惕")
    elif emergency_impact > 0.3:
        lines.append(f"      └─ 💡 正面事件影响，可能带来机会")
    elif abs(emergency_impact) < 0.1:
        lines.append(f"      └─ ✅ 无重大突发事件影响")
    
    # 综合分析建议
    lines.append("   " + "-" * 64)
    lines.append("   💡 综合建议:")
    
    # 计算综合评分
    total_score = (
        macro_score * 0.2 +
        sentiment_score * 0.2 +
        (1 - risk_level) * 0.2 +  # 风险等级越低越好
        capital_flow * 0.15 +
        policy_impact * 0.15 +
        emergency_impact * 0.1
    )
    
    if total_score > 0.3:
        lines.append(f"      ✅ 整体市场环境积极，可考虑适度加仓")
    elif total_score > 0.1:
        lines.append(f"      ⚠️  市场环境中性偏积极，保持谨慎乐观")
    elif total_score > -0.1:
        lines.append(f"      ⚠️  市场环境中性，建议保持观望")
    elif total_score > -0.3:
        lines.append(f"      ⚠️  市场环境偏弱，建议减仓或保持低仓位")
    else:
        lines.append(f"      🚨 市场环境较差，建议大幅减仓或空仓")
    
    if 'reasoning' in intelligence and intelligence['reasoning']:
        lines.append(f"   📝 分析理由: {intelligence['reasoning']}")
    
    lines.append("   " + "=" * 64)
    
    return "\n".join(lines)


# ==================== 持仓状态管理 ====================


# 保存持仓状态
def save_portfolio_state(stock_code, shares_held, current_balance, last_price, initial_balance):
    """保存当前持仓状态到文件"""
    try:
        state = {
            'stock_code': stock_code,
            'shares_held': float(shares_held),
            'current_balance': float(current_balance),
            'last_price': float(last_price),
            'initial_balance': float(initial_balance),
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_assets': float(current_balance + shares_held * last_price) if last_price > 0 else float(current_balance)
        }
        
        with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ⚠️  保存持仓状态失败: {e}")
        return False


# 加载持仓状态
def load_portfolio_state():
    """从文件加载持仓状态"""
    try:
        if not os.path.exists(PORTFOLIO_STATE_FILE):
            return None
        
        with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # 验证状态文件是否匹配当前股票
        if state.get('stock_code') != STOCK_CODE:
            print(f"   ⚠️  持仓状态文件中的股票代码 ({state.get('stock_code')}) 与当前股票 ({STOCK_CODE}) 不匹配")
            print(f"   💡 提示: 将使用默认初始状态，或运行 update_portfolio.py 更新持仓状态")
            return None
        
        return state
    except Exception as e:
        print(f"   ⚠️  加载持仓状态失败: {e}")
        return None


# 显示持仓状态
def show_portfolio_state(state):
    """显示加载的持仓状态"""
    if not state:
        return
    
    print("   " + "=" * 64)
    print("   📋 已加载持仓状态")
    print("   " + "=" * 64)
    print(f"   股票代码: {state.get('stock_code', '未知')}")
    print(f"   持仓数量: {state.get('shares_held', 0):.2f} 股")
    print(f"   可用资金: {state.get('current_balance', 0):.2f} 元")
    if state.get('last_price', 0) > 0:
        position_value = state.get('shares_held', 0) * state.get('last_price', 0)
        total_assets = state.get('current_balance', 0) + position_value
        print(f"   持仓市值: {position_value:.2f} 元")
        print(f"   总资产: {total_assets:.2f} 元")
    print(f"   上次更新: {state.get('last_update', '未知')}")
    print("   " + "=" * 64)
    print()


# ==================== 图形化持仓编辑（Flask 网页） ====================

portfolio_state_mtime = os.path.getmtime(PORTFOLIO_STATE_FILE) if os.path.exists(PORTFOLIO_STATE_FILE) else None

flask_app = None


def create_portfolio_web_app():
    global flask_app
    app = Flask(__name__)

    TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>持仓编辑器 - RL 股票实盘</title>
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
    .footer { margin-top:24px; font-size:12px; color:#999; text-align:center;}
  </style>
</head>
<body>
  <div class="container">
    <h1>持仓编辑器（实时同步</h1>
    <p class="desc">修改后点击“保存持仓”，<strong>正在运行的 real_time_predict_v7.py 会自动读取最新持仓</strong>，无需停止脚本。</p>
    <form method="post">
      <label>股票代码（与脚本一致）</label>
      <input type="text" name="stock_code" value="{{ stock_code }}" readonly>

      <div class="row">
        <div>
          <label>持仓数量（股）</label>
          <input type="number" step="0.01" name="shares_held" value="{{ shares_held }}">
        </div>
        <div>
          <label>可用资金（元）</label>
          <input type="number" step="0.01" name="current_balance" value="{{ current_balance }}">
        </div>
      </div>

      <div class="row">
        <div>
          <label>最近成交价/成本价（元）</label>
          <input type="number" step="0.0001" name="last_price" value="{{ last_price }}">
        </div>
        <div>
          <label>初始资金基准（元）</label>
          <input type="number" step="0.01" name="initial_balance" value="{{ initial_balance }}">
        </div>
      </div>

      <button type="submit">💾 保存持仓</button>
      {% if msg %}
      <div class="status">{{ msg }}</div>
      {% endif %}
    </form>
    <div class="footer">
      打开方式：在浏览器中访问 http://{{ host }}:{{ port }}<br>
      注意：本页面仅在本机可访问，安全用于手动更新持仓。
    </div>
  </div>
</body>
</html>
"""

    @app.route("/", methods=["GET", "POST"])
    def index():
        msg = ""
        state = load_portfolio_state()
        # 默认值
        data = {
            "stock_code": STOCK_CODE,
            "shares_held": 0.0,
            "current_balance": 100000.0,
            "last_price": 0.0,
            "initial_balance": 100000.0,
        }
        if state:
            data.update({
                "stock_code": state.get("stock_code", STOCK_CODE),
                "shares_held": state.get("shares_held", 0.0),
                "current_balance": state.get("current_balance", 100000.0),
                "last_price": state.get("last_price", 0.0),
                "initial_balance": state.get("initial_balance", 100000.0),
            })

        if request.method == "POST":
            try:
                stock_code = request.form.get("stock_code", STOCK_CODE).strip()
                shares_held = float(request.form.get("shares_held") or 0)
                current_balance = float(request.form.get("current_balance") or 0)
                last_price = float(request.form.get("last_price") or 0)
                initial_balance = float(request.form.get("initial_balance") or 0)

                save_portfolio_state(stock_code, shares_held, current_balance, last_price, initial_balance)
                msg = "✅ 已保存持仓状态，实时预测脚本将在下一轮自动同步。"
                data.update({
                    "stock_code": stock_code,
                    "shares_held": shares_held,
                    "current_balance": current_balance,
                    "last_price": last_price,
                    "initial_balance": initial_balance,
                })
            except Exception as e:
                msg = f"❌ 保存失败: {e}"

        return app.response_class(
            TEMPLATE.replace("{{ host }}", WEB_EDITOR_HOST).replace("{{ port }}", str(WEB_EDITOR_PORT))
                    .replace("{{ stock_code }}", str(data["stock_code"]))
                    .replace("{{ shares_held }}", str(data["shares_held"]))
                    .replace("{{ current_balance }}", str(data["current_balance"]))
                    .replace("{{ last_price }}", str(data["last_price"]))
                    .replace("{{ initial_balance }}", str(data["initial_balance"]))
                    .replace("{{ msg }}", msg),
            mimetype="text/html"
        )

    flask_app = app
    return app


def start_portfolio_web_editor():
    """在后台线程启动简单网页，用于图形化编辑持仓"""
    if not FLASK_AVAILABLE or not ENABLE_WEB_EDITOR:
        return

    app = create_portfolio_web_app()

    def run():
        try:
            app.run(host=WEB_EDITOR_HOST, port=WEB_EDITOR_PORT, debug=False, use_reloader=False)
        except Exception as e:
            print(f"⚠️  持仓网页编辑器启动失败: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"✅ 持仓网页编辑器已启动: 在浏览器中打开 http://{WEB_EDITOR_HOST}:{WEB_EDITOR_PORT}")
    print(f"   可在脚本运行时实时修改持仓信息，无需停止 real_time_predict_v7.py")


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
        if portfolio_state_mtime is None or mtime > portfolio_state_mtime + 1e-6:
            state = load_portfolio_state()
            if state:
                current_balance = state.get('current_balance', current_balance)
                shares_held = state.get('shares_held', shares_held)
                last_price = state.get('last_price', last_price)
                initial_balance = state.get('initial_balance', initial_balance)
                portfolio_state_mtime = mtime
                print("\n   🔄 检测到外部更新的持仓状态，已实时同步内存状态")
                show_portfolio_state(state)
    except Exception as e:
        print(f"   ⚠️  检测持仓状态更新失败: {e}")

    return current_balance, shares_held, last_price, initial_balance


# ==================== 操作记录功能 ====================


# 初始化交易日志文件
def init_trade_log():
    """初始化交易日志文件，如果不存在则创建表头"""
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '时间戳', '日期', '时间', '股票代码', '操作类型', '操作比例', 
                '当前价格', '建议买入价格', '建议卖出价格', '预测数量', '预测金额', 
                '持仓数量', '可用资金', '总资产', '操作状态', '备注'
            ])
        print(f"✅ 创建交易日志文件: {TRADE_LOG_FILE}")
        print(f"   📁 文件位置: {os.path.abspath(TRADE_LOG_FILE)}")


# 记录交易操作
def log_trade_operation(
    stock_code, operation, current_price, shares_held, current_balance, 
    total_assets, status='待执行', note='', suggested_buy_price=None, suggested_sell_price=None
):
    """
    记录交易操作到CSV文件
    
    参数:
        stock_code: 股票代码
        operation: 操作类型（如"买入 100%"）
        current_price: 当前价格
        shares_held: 当前持仓数量
        current_balance: 当前可用资金
        total_assets: 总资产
        status: 操作状态（待执行/已执行/预测）
        note: 备注
        suggested_buy_price: 建议买入价格（可选）
        suggested_sell_price: 建议卖出价格（可选）
    """
    try:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        # 解析操作类型和比例
        if "买入" in operation:
            op_type = "买入"
            op_percentage = operation.split()[-1] if "%" in operation else "0%"
        elif "卖出" in operation:
            op_type = "卖出"
            op_percentage = operation.split()[-1] if "%" in operation else "0%"
        else:
            op_type = "持有"
            op_percentage = "0%"
        
        # 计算预测数量和金额（如果是买入/卖出）
        if "买入" in operation or "卖出" in operation:
            percentage = float(op_percentage[:-1]) / 100
            if "买入" in operation:
                predicted_amount = current_balance * percentage
                predicted_shares = predicted_amount / current_price if current_price > 0 else 0
                # 如果没有提供建议买入价格，使用当前价格
                if suggested_buy_price is None:
                    suggested_buy_price = current_price
            else:
                predicted_shares = shares_held * percentage
                predicted_amount = predicted_shares * current_price
                # 如果没有提供建议卖出价格，使用当前价格
                if suggested_sell_price is None:
                    suggested_sell_price = current_price
        else:
            predicted_shares = 0
            predicted_amount = 0
            # 非买卖操作时，如果需要建议价格，默认使用当前价格
            if suggested_buy_price is None:
                suggested_buy_price = current_price
            if suggested_sell_price is None:
                suggested_sell_price = current_price
        
        # 写入CSV文件
        with open(TRADE_LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, date, time_str, stock_code, op_type, op_percentage,
                f"{current_price:.2f}",  # 当前价格
                f"{suggested_buy_price:.2f}" if suggested_buy_price else "",  # 建议买入价格
                f"{suggested_sell_price:.2f}" if suggested_sell_price else "",  # 建议卖出价格
                f"{predicted_shares:.2f}",  # 预测数量
                f"{predicted_amount:.2f}",  # 预测金额
                f"{shares_held:.2f}",  # 持仓数量
                f"{current_balance:.2f}",  # 可用资金
                f"{total_assets:.2f}",  # 总资产
                status,  # 操作状态
                note  # 备注
            ])
        
        return True
    except Exception as e:
        print(f"   ⚠️  记录交易操作失败: {e}")
        import traceback
        print(f"   [详细错误] {traceback.format_exc()}")
        return False


# 读取待执行的操作汇总
def get_pending_operations():
    """读取待执行的操作汇总"""
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return []
        
        df = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig')
        # 筛选待执行或预测的操作
        pending = df[df['操作状态'].isin(['待执行', '预测'])].copy()
        
        if len(pending) == 0:
            return []
        
        # 按时间排序
        pending = pending.sort_values('时间戳')
        return pending.to_dict('records')
    except Exception as e:
        print(f"   ⚠️  读取待执行操作失败: {e}")
        return []


# 更新操作状态
def update_operation_status(timestamp, new_status, note=''):
    """更新操作状态"""
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return False
        
        # 读取CSV文件
        df = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig')
        
        # 更新状态
        mask = df['时间戳'] == timestamp
        if mask.any():
            df.loc[mask, '操作状态'] = new_status
            if note:
                df.loc[mask, '备注'] = note
            
            # 保存
            df.to_csv(TRADE_LOG_FILE, index=False, encoding='utf-8-sig')
            return True
        return False
    except Exception as e:
        print(f"   ⚠️  更新操作状态失败: {e}")
        return False


# 显示操作汇总
def show_trade_summary():
    """显示操作汇总"""
    pending_ops = get_pending_operations()
    
    if len(pending_ops) == 0:
        return "   ✅ 暂无待执行的操作"
    
    lines = []
    lines.append("   " + "=" * 64)
    lines.append(f"   📋 待执行操作汇总（共 {len(pending_ops)} 条）")
    lines.append("   " + "=" * 64)
    
    for i, op in enumerate(pending_ops, 1):
        op_type = op.get('操作类型', '未知')
        op_percentage = op.get('操作比例', '0%')
        current_price = op.get('当前价格', '0.00')
        suggested_buy_price = op.get('建议买入价格', '')
        suggested_sell_price = op.get('建议卖出价格', '')
        predicted_shares = op.get('预测数量', '0.00')
        predicted_amount = op.get('预测金额', '0.00')
        timestamp = op.get('时间戳', '未知')
        status = op.get('操作状态', '未知')
        
        icon = "🟢" if op_type == "买入" else "🔴" if op_type == "卖出" else "⚪"
        lines.append(f"   {i}. {icon} {op_type} {op_percentage}")
        lines.append(f"      时间: {timestamp}")
        lines.append(f"      当前价格: {current_price} 元")
        
        # 显示建议价格
        if op_type == "买入" and suggested_buy_price:
            lines.append(f"      建议买入价格: {suggested_buy_price} 元")
            if predicted_shares and float(predicted_shares) > 0:
                lines.append(f"      预测数量: {predicted_shares} 股")
                lines.append(f"      预测金额: {predicted_amount} 元")
        elif op_type == "卖出" and suggested_sell_price:
            lines.append(f"      建议卖出价格: {suggested_sell_price} 元")
            if predicted_shares and float(predicted_shares) > 0:
                lines.append(f"      预测数量: {predicted_shares} 股")
                lines.append(f"      预测金额: {predicted_amount} 元")
        
        lines.append(f"      状态: {status}")
        if i < len(pending_ops):
            lines.append("")
    
    lines.append("   " + "=" * 64)
    lines.append(f"   💡 提示: 查看完整记录请查看文件 {TRADE_LOG_FILE}")
    
    return "\n".join(lines)


# 显示最近的操作历史
def show_recent_trades(limit=10):
    """显示最近的操作历史"""
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return "   ℹ️  暂无操作记录"
        
        df = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig')
        if len(df) == 0:
            return "   ℹ️  暂无操作记录"
        
        # 按时间排序，取最近N条
        df = df.sort_values('时间戳', ascending=False).head(limit)
        
        lines = []
        lines.append("   " + "=" * 64)
        lines.append(f"   📜 最近操作历史（最近 {len(df)} 条）")
        lines.append("   " + "=" * 64)
        
        for _, row in df.iterrows():
            op_type = row.get('操作类型', '未知')
            op_percentage = row.get('操作比例', '0%')
            current_price = row.get('当前价格', '0.00')
            suggested_buy_price = row.get('建议买入价格', '')
            suggested_sell_price = row.get('建议卖出价格', '')
            timestamp = row.get('时间戳', '未知')
            status = row.get('操作状态', '未知')
            
            icon = "🟢" if op_type == "买入" else "🔴" if op_type == "卖出" else "⚪"
            status_icon = "✅" if status == "已执行" else "⏳" if (status == "待执行" or status == "预测") else "❌"
            
            price_info = f"当前: {current_price}"
            if op_type == "买入" and suggested_buy_price:
                price_info += f" | 建议买入: {suggested_buy_price}"
            elif op_type == "卖出" and suggested_sell_price:
                price_info += f" | 建议卖出: {suggested_sell_price}"
            
            lines.append(f"   {icon} {op_type} {op_percentage} | {price_info} | {status_icon} {status}")
            lines.append(f"      时间: {timestamp}")
        
        lines.append("   " + "=" * 64)
        
        return "\n".join(lines)
    except Exception as e:
        return f"   ⚠️  读取操作历史失败: {e}"


# ==================== 主循环 ====================

# 初始化交易日志
init_trade_log()

# 尝试迁移旧格式日志文件（如果存在且需要）
try:
    if os.path.exists(TRADE_LOG_FILE):
        # 检查是否需要迁移
        df_check = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig', nrows=1)
        if '建议买入价格' not in df_check.columns:
            print(f"   🔄 检测到旧格式日志文件，正在迁移...")
            # 执行迁移逻辑（简化版）
            df = pd.read_csv(TRADE_LOG_FILE, encoding='utf-8-sig')
            if '建议买入价格' not in df.columns:
                df['建议买入价格'] = ''
            if '建议卖出价格' not in df.columns:
                df['建议卖出价格'] = ''
            if '预测数量' not in df.columns:
                df['预测数量'] = df.get('数量', 0.0)
            if '预测金额' not in df.columns:
                df['预测金额'] = df.get('金额', 0.0)
            if '当前价格' not in df.columns:
                df['当前价格'] = df.get('价格', 0.0)
            
            # 填充建议价格
            for idx, row in df.iterrows():
                if row['操作类型'] == '买入' and (pd.isna(row.get('建议买入价格', '')) or row.get('建议买入价格', '') == ''):
                    df.at[idx, '建议买入价格'] = row.get('当前价格', row.get('价格', 0.0))
                elif row['操作类型'] == '卖出' and (pd.isna(row.get('建议卖出价格', '')) or row.get('建议卖出价格', '') == ''):
                    df.at[idx, '建议卖出价格'] = row.get('当前价格', row.get('价格', 0.0))
            
            # 重新排列列
            new_order = ['时间戳', '日期', '时间', '股票代码', '操作类型', '操作比例',
                        '当前价格', '建议买入价格', '建议卖出价格', '预测数量', '预测金额',
                        '持仓数量', '可用资金', '总资产', '操作状态', '备注']
            existing_cols = [col for col in new_order if col in df.columns]
            df = df[existing_cols]
            df.to_csv(TRADE_LOG_FILE, index=False, encoding='utf-8-sig')
            print(f"   ✅ 日志文件格式已更新")
except Exception as e:
    print(f"   ⚠️  日志文件迁移跳过: {e}")

# 尝试加载持仓状态
portfolio_state = load_portfolio_state()
if portfolio_state:
    show_portfolio_state(portfolio_state)
    # 从状态文件加载持仓信息
    initial_balance = portfolio_state.get('initial_balance', 100000.0)
    current_balance = portfolio_state.get('current_balance', initial_balance)
    shares_held = portfolio_state.get('shares_held', 0.0)
    last_price = portfolio_state.get('last_price', 0.0)
    print(f"✅ 已从 {PORTFOLIO_STATE_FILE} 加载持仓状态")
    print(f"   持仓数量: {shares_held:.2f} 股")
    print(f"   可用资金: {current_balance:.2f} 元")
    print()
else:
    # 使用默认初始状态
    initial_balance = 100000.0  # 初始资金
    current_balance = initial_balance
    shares_held = 0.0  # 当前持股数
    last_price = 0.0  # 上次价格，用于计算盈亏
    print(f"ℹ️  使用默认初始状态（未找到持仓状态文件或股票代码不匹配）")
    print(f"   初始资金: {initial_balance:.2f} 元")
    print(f"   持仓数量: {shares_held:.2f} 股")
    print()

# 启动图形化持仓编辑器
if ENABLE_WEB_EDITOR and FLASK_AVAILABLE:
    start_portfolio_web_editor()
elif ENABLE_WEB_EDITOR and not FLASK_AVAILABLE:
    print("⚠️  已启用图形化持仓管理，但未安装 Flask，无法启动网页编辑器。")
    print("   请运行: pip install flask")

consecutive_empty_count = 0  # 连续空数据计数
max_empty_before_extend = 3  # 连续3次空数据后扩展日期范围
last_day = None  # 上一个交易日
last_action = None  # 上一个动作，用于检测变化
last_price_value = None  # 上次价格值，用于检测价格变化
last_data_time = None  # 上次数据时间，用于检测数据更新

last_shares_held = shares_held  # 上次持仓数量，用于检测仓位变动
daily_pnl = 0.0  # 每日盈亏
daily_pnl_history = []  # 存储每日盈亏记录

print("🚀 开始实时预测循环...")
print("📌 模型预测基于 V7 (126维价格序列)")
print("📌 LLM 情报仅作为参考，不影响模型预测")
print(f"📌 数据源: {DATA_SOURCE.upper()} ({'支持实时数据' if DATA_SOURCE in ['tushare', 'akshare'] else '有延迟'})")
print(f"📌 操作记录: {TRADE_LOG_FILE}")
print(f"📌 持仓管理: 支持通过网页实时修改（http://{WEB_EDITOR_HOST}:{WEB_EDITOR_PORT}）")
print()

while True:
    try:
        current_time = datetime.datetime.now()
        is_weekend = current_time.weekday() >= 5
        is_trading = is_trading_time()
        
        # 如果连续多次获取不到数据，尝试扩展日期范围
        extend_days = min(consecutive_empty_count // max_empty_before_extend, 5)
        
        # 尝试获取数据（优先今天的数据）
        df = fetch_data_with_retry(extend_days=extend_days, try_today=True)
        
        if not df.empty and len(df) > 0:
            # 重置连续空数据计数
            consecutive_empty_count = 0
            
            df = df.sort_values('time')  # 按时间排序
            recent_closes = df['close'].astype(float).values  # 所有 close
            
            # 构建 V7 模型观察向量（126维价格序列）
            # 如果实时数据不足，用历史数据补充（保留实时数据）
            if len(recent_closes) < 126:
                # 需要补充的数据量
                need_more = 126 - len(recent_closes)
                realtime_count = len(recent_closes)
                
                # 优先从已获取的df中提取历史数据来补充
                try:
                    # 从df中获取所有数据的收盘价
                    all_closes = df['close'].astype(float).values
                    
                    if len(all_closes) >= 126:
                        # 如果df中有足够的数据，直接使用最后126条
                        recent_closes = all_closes[-126:]
                        print(f"✅ 数据使用: 从 {len(df)} 条数据中提取最后 126 条（包含实时数据 {realtime_count} 条）")
                    elif len(all_closes) > realtime_count:
                        # 如果df中有更多数据，使用所有数据，不足部分用历史数据补充
                        # 获取df中更早的数据（排除当前实时数据）
                        earlier_closes = all_closes[:-realtime_count] if realtime_count > 0 else all_closes
                        available_history = len(earlier_closes)
                        
                        if available_history >= need_more:
                            # 使用更早数据的最后部分来补充
                            history_supplement = earlier_closes[-need_more:]
                            recent_closes = np.concatenate([history_supplement, recent_closes])
                            print(f"✅ 数据补充: 实时数据 {realtime_count} 条 + 历史数据 {need_more} 条 = {len(recent_closes)} 条")
                        else:
                            # 历史数据不足，用历史数据的平均值填充剩余部分
                            if available_history > 0:
                                avg_value = np.mean(earlier_closes)
                                remaining = need_more - available_history
                                history_supplement = earlier_closes
                                padding = np.full(remaining, avg_value)
                                recent_closes = np.concatenate([padding, history_supplement, recent_closes])
                                print(f"✅ 数据补充: 实时数据 {realtime_count} 条 + 历史数据 {available_history} 条 + 平均值填充 {remaining} 条")
                            else:
                                # 完全没有历史数据，用最后值填充
                                last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                padding = np.full(need_more, last_value)
                                recent_closes = np.concatenate([padding, recent_closes])
                                print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                    else:
                        # df中的数据就是实时数据，需要获取更多历史数据
                        # 用最后值填充
                        last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                        padding = np.full(need_more, last_value)
                        recent_closes = np.concatenate([padding, recent_closes])
                        print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                except Exception as e:
                    # 出错时用最后值填充
                    last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                    padding = np.full(need_more, last_value)
                    recent_closes = np.concatenate([padding, recent_closes])
                    print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条（错误: {e}）")
            
            # 确保取最后126条
            obs = np.array(recent_closes[-126:], dtype=np.float32)
            
            # 计算技术指标用于决策分析
            price_trend = None
            price_volatility = None
            recent_change = None
            if len(recent_closes) >= 20:
                # 计算价格趋势（最近20个数据点的趋势）
                recent_20 = recent_closes[-20:]
                price_trend = (recent_20[-1] - recent_20[0]) / recent_20[0] * 100  # 百分比变化
                
                # 计算波动率
                price_volatility = np.std(recent_20) / np.mean(recent_20) * 100
                
                # 最近变化
                if len(recent_closes) >= 2:
                    recent_change = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2] * 100
            
            # V7 模型预测（仅使用价格序列）
            action, _states = model.predict(obs, deterministic=True)
            operation = map_action_to_operation(action)
            volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
            
            # 获取动作概率分布（用于分析决策信心）
            action_probs = None
            try:
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                action_probs = model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()[0]
            except:
                pass  # 如果获取失败就跳过
            
            # 获取最新数据的日期和时间
            latest_date = df['date'].iloc[-1] if 'date' in df.columns else '未知'
            latest_time = df['time'].iloc[-1] if 'time' in df.columns else '未知'
            current_price = recent_closes[-1]
            
            # 在使用前，先检查 portfolio_state.json 是否被外部修改，若有则实时同步
            current_balance, shares_held, last_price, initial_balance = refresh_portfolio_from_file_if_changed(
                current_balance, shares_held, last_price, initial_balance
            )
            
            # 检测数据是否更新
            data_updated = (last_data_time != latest_time or last_price_value != current_price)
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
            
            # 获取市场情报（仅作为参考，不输入模型）
            # 强制使用真实 API，不使用缓存中的模拟数据
            intelligence = None
            intelligence_source = "未知"
            if llm_agent and latest_date != '未知':
                try:
                    # 检查是否是模拟模式
                    if hasattr(llm_agent, 'mock_mode') and llm_agent.mock_mode:
                        print(f"   ⚠️  [LLM警告] 当前为模拟模式，API 密钥可能未正确配置")
                        print(f"   💡 提示: 请检查 DEEPSEEK_API_KEY 环境变量或代码中的 API 密钥配置")
                        print(f"   🔑 API 密钥状态: {'已设置' if hasattr(llm_agent, 'api_key') and llm_agent.api_key else '未设置'}")
                    
                    # 强制刷新市场情报（不使用缓存），确保获取真实数据
                    # 如果缓存中有模拟数据，强制刷新可以获取真实数据
                    force_refresh = True  # 强制刷新，确保使用真实 API
                    print(f"   🔄 正在从 DeepSeek API 获取市场情报（强制刷新）...")
                    intelligence = llm_agent.get_market_intelligence(
                        latest_date, 
                        force_refresh=force_refresh
                    )
                    
                    # 判断数据来源
                    if intelligence:
                        source = intelligence.get('source', 'unknown')
                        if source == 'mock_data':
                            intelligence_source = "⚠️ 模拟数据（API 可能未正确配置）"
                            print(f"   ⚠️  [LLM警告] 获取到模拟数据，API 可能未正确调用")
                            print(f"   💡 建议: 检查 API 密钥配置或网络连接")
                        elif source == 'deepseek' or source == 'grok':
                            intelligence_source = "✅ 真实 API 数据"
                        else:
                            intelligence_source = f"缓存 ({source})"
                except Exception as e:
                    print(f"   [LLM错误] 获取市场情报失败: {e}")
                    import traceback
                    print(f"   [详细错误] {traceback.format_exc()}")
            
            if latest_date == today_str:
                data_status = "🟢 实时数据（今日）"
                data_status_detail = f"✅ 已获取到 {today_str} 的实时数据（数据源: {DATA_SOURCE.upper()}）"
            else:
                # 计算数据日期与今天的差异
                try:
                    data_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d').date()
                    days_diff = (datetime.date.today() - data_date).days
                    if days_diff == 1:
                        data_status = "🟡 昨日数据"
                        data_status_detail = f"ℹ️  当前时间: {current_time_str}, 数据日期: {latest_date}（{days_diff}天前）"
                    else:
                        data_status = "🟡 历史数据"
                        data_status_detail = f"ℹ️  当前时间: {current_time_str}, 数据日期: {latest_date}（{days_diff}天前）"
                except:
                    data_status = "🟡 历史数据"
                    data_status_detail = f"ℹ️  数据日期: {latest_date}"
            
            print(f"   数据状态: {data_status}")
            print(f"   {data_status_detail}")
            print(f"   数据时间: {latest_time}, 数据条数: {len(df)}")
            print(f"   模型: V7 (126维价格序列)")
            
            # 如果是历史数据，给出原因说明
            if latest_date != today_str:
                if is_weekend:
                    print(f"   💡 原因: 今天是周末（非交易日）")
                elif not is_trading:
                    print(f"   💡 原因: 当前非交易时间（交易时间: 9:30-11:30, 13:00-15:00）")
                else:
                    print(f"   💡 原因: 数据源可能尚未更新今日数据，或今日无交易")
            
            if not data_updated:
                print(f"   ⚠️  提示: 数据与上次相同，可能是非交易时间或数据源未更新")
            
            # 显示详细的市场情报参考（如果可用）
            if intelligence:
                print()
                print(format_intelligence_detailed(intelligence))
                print(f"   📌 数据来源: {intelligence_source} ({intelligence.get('source', 'unknown')})")
            else:
                print("   ℹ️  暂无市场情报参考（LLM 未启用或数据获取失败）")
            
            # 显示预测结果
            print("=" * 70)
            
            # 检测操作变化
            action_changed = (last_action is not None and operation != last_action)
            
            # 数据更新状态提示
            if not data_updated:
                print(f"⚠️  数据未更新（与上次相同）")
            
            # 计算建议价格（用于显示）
            dyn_buy_offset, dyn_sell_offset = get_dynamic_offsets(price_volatility)
            suggested_buy_price = current_price * (1 + dyn_buy_offset) if "买入" in operation else None
            suggested_sell_price = current_price * (1 + dyn_sell_offset) if "卖出" in operation else None
            
            if action_changed:
                print(f"⚠️  动作变化！从 {last_action} 变为 {operation}")
                # 用颜色突出（ANSI 红色）
                price_info = f"当前价格: {current_price:.2f}"
                if suggested_buy_price:
                    price_info += f" | 建议买入价格: {suggested_buy_price:.2f} (偏移: {dyn_buy_offset*100:+.2f}%)"
                if suggested_sell_price:
                    price_info += f" | 建议卖出价格: {suggested_sell_price:.2f} (偏移: {dyn_sell_offset*100:+.2f}%)"
                print(f"\033[91m✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, {price_info}, 成交量: {volume:.0f}, 预测动作: {operation}\033[0m")
            else:
                price_info = f"当前价格: {current_price:.2f}"
                if suggested_buy_price:
                    price_info += f" | 建议买入价格: {suggested_buy_price:.2f} (偏移: {dyn_buy_offset*100:+.2f}%)"
                if suggested_sell_price:
                    price_info += f" | 建议卖出价格: {suggested_sell_price:.2f} (偏移: {dyn_sell_offset*100:+.2f}%)"
                print(f"✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, {price_info}, 成交量: {volume:.0f}, 预测动作: {operation}")
            
            # 先计算总资产，供后续决策分析和持仓信息使用
            total_assets = current_balance + shares_held * current_price
            position_value = shares_held * current_price  # 持仓市值
            
            # 显示决策分析
            print()
            print("   " + "=" * 64)
            print("   🔍 模型决策分析")
            print("   " + "=" * 64)
            
            # 价格趋势分析
            if price_trend is not None:
                trend_icon = "📈" if price_trend > 0 else "📉" if price_trend < 0 else "➡️"
                print(f"   价格趋势（近20点）{trend_icon}: {price_trend:+.2f}%")
                if price_trend > 2:
                    print(f"      └─ 💡 近期上涨趋势明显，可能是买入信号")
                elif price_trend < -2:
                    print(f"      └─ ⚠️  近期下跌趋势，需谨慎")
                else:
                    print(f"      └─ ➡️  价格相对稳定")
            
            # 波动率分析
            if price_volatility is not None:
                vol_level = "高" if price_volatility > 2 else "中" if price_volatility > 1 else "低"
                print(f"   价格波动率: {price_volatility:.2f}% ({vol_level})")
                # 显示动态偏移说明
                if "买入" in operation or "卖出" in operation:
                    if price_volatility < 0.2:
                        offset_factor = 0.5
                        offset_desc = "缩小一半（波动小，更容易成交）"
                    elif price_volatility < 0.5:
                        offset_factor = 1.0
                        offset_desc = "基础偏移（正常波动）"
                    else:
                        offset_factor = 2.0
                        offset_desc = "放大一倍（波动大，给更多空间）"
                    print(f"      └─ 动态偏移策略: {offset_desc}")
            
            # 建议价格显示
            if "买入" in operation and suggested_buy_price:
                price_diff = suggested_buy_price - current_price
                price_diff_pct = (price_diff / current_price) * 100
                print(f"   💰 建议买入价格: {suggested_buy_price:.2f} 元 (当前价格: {current_price:.2f} 元, 偏移: {price_diff_pct:+.2f}%)")
            elif "卖出" in operation and suggested_sell_price:
                price_diff = suggested_sell_price - current_price
                price_diff_pct = (price_diff / current_price) * 100
                print(f"   💰 建议卖出价格: {suggested_sell_price:.2f} 元 (当前价格: {current_price:.2f} 元, 偏移: {price_diff_pct:+.2f}%)")
            
            # 最近变化
            if recent_change is not None:
                change_icon = "📈" if recent_change > 0 else "📉" if recent_change < 0 else "➡️"
                print(f"   最近变化 {change_icon}: {recent_change:+.2f}%")
            
            # 动作概率分析
            if action_probs is not None:
                action_names = ["卖出100%", "卖出50%", "卖出25%", "持有", "买入25%", "买入50%", "买入100%"]
                max_prob_idx = np.argmax(action_probs)
                max_prob = action_probs[max_prob_idx] * 100
                print(f"   决策信心: {max_prob:.1f}% (选择: {action_names[max_prob_idx]})")
                
                # 显示前3个最可能的动作
                top3_indices = np.argsort(action_probs)[-3:][::-1]
                print(f"   前3个可能动作:")
                for i, idx in enumerate(top3_indices, 1):
                    prob = action_probs[idx] * 100
                    print(f"      {i}. {action_names[idx]}: {prob:.1f}%")
            
            # 当前持仓状态对决策的影响
            position_ratio = (position_value / total_assets * 100) if total_assets > 0 else 0
            if position_ratio == 0:
                print(f"   持仓状态: 空仓 (0%)")
                print(f"      └─ 💡 当前空仓，模型可能认为这是买入机会")
            elif position_ratio < 30:
                print(f"   持仓状态: 低仓位 ({position_ratio:.1f}%)")
                print(f"      └─ 💡 仓位较低，模型可能建议加仓")
            elif position_ratio > 70:
                print(f"   持仓状态: 高仓位 ({position_ratio:.1f}%)")
                print(f"      └─ ⚠️  仓位较高，模型可能建议减仓")
            else:
                print(f"   持仓状态: 中等仓位 ({position_ratio:.1f}%)")
            
            # 决策原因推测
            print()
            print("   💡 决策原因推测:")
            if "买入" in operation and position_ratio == 0:
                print(f"      ✅ 空仓状态，模型识别到买入机会")
                if price_trend and price_trend > 0:
                    print(f"      ✅ 价格呈上涨趋势，支持买入决策")
                if price_volatility and price_volatility < 2:
                    print(f"      ✅ 波动率较低，风险可控")
            elif "买入" in operation:
                print(f"      ✅ 模型认为当前价格具有投资价值")
                if price_trend and price_trend < 0:
                    print(f"      ⚠️  虽然价格下跌，但模型可能认为已到买入时机")
            elif "卖出" in operation:
                print(f"      ⚠️  模型建议卖出，可能是风险控制或获利了结")
            elif "持有" in operation:
                print(f"      ➡️  模型建议持有，等待更好的交易时机")
            
            print("   " + "=" * 64)
            
            # 显示当前持仓信息
            print()
            print("   " + "=" * 64)
            print("   💼 当前持仓信息（已实时同步外部修改）")
            print("   " + "=" * 64)
            print(f"   持仓数量: {shares_held:.2f} 股")
            print(f"   持仓市值: {position_value:.2f} 元 ({position_ratio:.1f}%)")
            print(f"   可用资金: {current_balance:.2f} 元 ({100-position_ratio:.1f}%)")
            print(f"   总资产: {total_assets:.2f} 元")
            if shares_held > 0 and last_price > 0:
                # 计算持仓盈亏
                cost_basis = last_price  # 简化：使用上次价格作为成本价
                pnl = (current_price - cost_basis) * shares_held
                pnl_ratio = (current_price / cost_basis - 1) * 100 if cost_basis > 0 else 0
                pnl_icon = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
                print(f"   持仓盈亏: {pnl_icon} {pnl:+.2f} 元 ({pnl_ratio:+.2f}%)")
            print("   " + "=" * 64)
            
            # 初始化仓位变动标记（在执行交易前）
            position_changed = False  # 标记仓位是否变动
            
            # 更新状态变量（在执行交易前保存）
            last_action = operation  # 更新上次动作
            last_price_value = current_price  # 更新上次价格值
            last_data_time = latest_time  # 更新上次数据时间
            
            # 模拟交易执行（仅当操作变化且是买入/卖出时执行，避免重复执行）
            trade_amount = 0.0  # 交易金额
            trade_shares = 0.0  # 交易数量
            
            if action_changed and ("买入" in operation or "卖出" in operation):
                if "买入" in operation:
                    buy_percentage = float(operation.split()[-1][:-1]) / 100  # e.g., 25% -> 0.25
                    buy_amount = current_balance * buy_percentage
                    shares_bought = buy_amount / current_price if current_price > 0 else 0
                    
                    # 执行买入
                    shares_held += shares_bought
                    current_balance -= buy_amount
                    position_changed = True
                    trade_amount = buy_amount
                    trade_shares = shares_bought
                    
                    # 显示执行买入信息（包含建议价格）
                    print(f"   💰 执行买入: {buy_percentage*100:.0f}%, 金额: {buy_amount:.2f} 元, 数量: {shares_bought:.2f} 股")
                    if suggested_buy_price:
                        print(f"      💡 建议买入价格: {suggested_buy_price:.2f} 元 (当前执行价格: {current_price:.2f} 元)")
                    
                elif "卖出" in operation:
                    sell_percentage = float(operation.split()[-1][:-1]) / 100
                    shares_sold = shares_held * sell_percentage
                    sell_amount = shares_sold * current_price
                    
                    # 执行卖出
                    shares_held -= shares_sold
                    current_balance += sell_amount
                    position_changed = True
                    trade_amount = sell_amount
                    trade_shares = shares_sold
                    
                    # 显示执行卖出信息（包含建议价格）
                    print(f"   💰 执行卖出: {sell_percentage*100:.0f}%, 金额: {sell_amount:.2f} 元, 数量: {shares_sold:.2f} 股")
                    if suggested_sell_price:
                        print(f"      💡 建议卖出价格: {suggested_sell_price:.2f} 元 (当前执行价格: {current_price:.2f} 元)")
            
            # 记录预测操作（动作变化时记录，包含建议价格）
            if action_changed and ("买入" in operation or "卖出" in operation):
                # 根据波动率动态计算偏移
                dyn_buy_offset, dyn_sell_offset = get_dynamic_offsets(price_volatility)
                # 计算建议价格（买入稍微低一点，卖出稍微高一点）
                suggested_buy_price = current_price * (1 + dyn_buy_offset) if "买入" in operation else None
                suggested_sell_price = current_price * (1 + dyn_sell_offset) if "卖出" in operation else None
                
                # 记录预测操作（状态为"预测"）
                note = f"模型预测: {operation}"
                log_trade_operation(
                    STOCK_CODE, operation, current_price, shares_held, 
                    current_balance, total_assets, status='预测', note=note,
                    suggested_buy_price=suggested_buy_price,
                    suggested_sell_price=suggested_sell_price
                )
                print(f"   📝 预测操作已记录到日志: {TRADE_LOG_FILE}")
            
            # 只在仓位真正变动时记录到日志（已执行的操作）
            if position_changed:
                # 重新计算总资产（交易后）
                total_assets_after = current_balance + shares_held * current_price
                
                # 根据波动率动态计算偏移（执行时也使用同样规则）
                dyn_buy_offset, dyn_sell_offset = get_dynamic_offsets(price_volatility)
                suggested_buy_price = current_price * (1 + dyn_buy_offset) if "买入" in operation else None
                suggested_sell_price = current_price * (1 + dyn_sell_offset) if "卖出" in operation else None
                
                # 记录操作到日志（使用交易后的持仓信息）
                note = f"仓位变动: {operation}"
                log_trade_operation(
                    STOCK_CODE, operation, current_price, shares_held, 
                    current_balance, total_assets_after, status='已执行', note=note,
                    suggested_buy_price=suggested_buy_price,
                    suggested_sell_price=suggested_sell_price
                )
                print(f"   📝 仓位变动已记录到日志: {TRADE_LOG_FILE}")
                
                # 更新上次持仓数量
                last_shares_held = shares_held
                
                # 保存持仓状态（仓位变动后）
                save_portfolio_state(STOCK_CODE, shares_held, current_balance, current_price, initial_balance)
            
            # 定期保存持仓状态（即使没有仓位变动，也定期保存）
            if random.randint(1, 10) == 1:  # 10% 概率保存
                save_portfolio_state(STOCK_CODE, shares_held, current_balance, current_price, initial_balance)
            
            # 显示操作汇总（仅在有待执行操作时显示）
            pending_summary = show_trade_summary()
            if "暂无待执行的操作" not in pending_summary:
                print()
                print(pending_summary)
                print()
            
            # 显示最近操作历史（每10次循环显示一次，或仓位变动时显示）
            if position_changed or random.randint(1, 10) == 1:  # 仓位变动时或10% 概率显示
                print(show_recent_trades(limit=5))
                print()
            
            # 检查是否是收盘时间或新一天，计算每日盈亏
            current_day = latest_date
            if last_day is not None and current_day != last_day:
                # 计算每日盈亏
                current_net_worth = current_balance + shares_held * last_price
                daily_pnl = current_net_worth - initial_balance  # 相对初始资金的盈亏
                daily_pnl_history.append((last_day, daily_pnl))
                print(f"📊 每日收盘盈亏 ({last_day}): {daily_pnl:.2f} 元 (净值: {current_net_worth:.2f} 元)")
                initial_balance = current_net_worth  # 更新基准
                
            last_price = current_price  # 更新上次价格
            last_day = current_day  # 更新上次日期
            
            # 根据是否在交易时间决定等待时间
            if is_trading:
                wait_time = 60  # 交易时间内等待1分钟
            else:
                wait_time = 120  # 非交易时间等待2分钟
            
            time.sleep(wait_time + random.uniform(0, 30))
        else:
            consecutive_empty_count += 1
            
            # 根据情况给出不同的提示
            if is_weekend:
                reason = "周末（非交易日）"
                wait_time = 300  # 周末等待5分钟
            elif not is_trading:
                reason = "非交易时间"
                wait_time = 120  # 非交易时间等待2分钟
            else:
                reason = "可能数据源暂时无数据"
                wait_time = 60  # 交易时间等待1分钟
            
            print(f"⏸️  时间: {time.ctime()}, 未找到数据 - {reason}")
            if extend_days > 0:
                print(f"   已扩展日期范围至 {extend_days} 天前")
            print(f"   等待 {wait_time} 秒后重试...")
            print()
            
            time.sleep(wait_time + random.uniform(0, 30))
            continue  # 跳过后续的 sleep，因为已经 sleep 了
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在退出...")
        break
    except Exception as e:
        consecutive_empty_count += 1
        print(f"❌ 时间: {time.ctime()}, 错误: {e}")
        print(f"   等待 60 秒后重试...")
        print()
        time.sleep(60 + random.uniform(0, 30))
        continue  # 跳过后续的 sleep

# 清理资源
if DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
    bs.logout()

print("\n✅ 程序已退出")


