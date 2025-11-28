"""
V3 实时预测脚本 - 集成 V8/V9 LLM 市场情报增强
支持：
1. 宏观经济数据 - GDP、CPI、利率政策分析
2. 新闻和舆情分析 - 市场热点、负面消息识别
3. 市场情绪指标 - 恐慌指数 VIX、投资者情绪
4. 资金流向数据 - 外资、融资融券、北向资金
5. 政策变化信息 - 货币/财政/监管政策影响
6. 国际市场联动 - 美股、港股相关性分析
7. 突发事件应对 - 地缘政治、疫情、自然灾害
"""

import os
import sys
import random
import warnings
import baostock as bs
import numpy as np

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

import time
import pandas as pd
import datetime  # 用于日期计算

# 导入 LLM 市场情报模块
try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入 llm_market_intelligence 模块，将仅使用技术指标")
    LLM_AVAILABLE = False

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# ==================== 配置参数 ====================
MODEL_PATH = "ppo_stock_v7.zip"  # 默认模型路径，可修改为 v8 或 v9
STOCK_CODE = 'sh.600036'  # 股票代码
LLM_PROVIDER = "deepseek"  # LLM 提供商：deepseek 或 grok
ENABLE_LLM = True  # 是否启用 LLM 市场情报
DEEPSEEK_API_KEY = "sk-167914945f7945d498e09a7f186c101d"  # DeepSeek API 密钥

# ==================== 初始化 ====================
print("=" * 70)
print("V3 实时预测系统 - LLM 市场情报增强版")
print("=" * 70)

# baostock 登录
bs.login()
print("✅ baostock 登录成功！")

# 加载模型（在抑制警告的上下文中）
with SuppressGymWarning():
    if not os.path.exists(MODEL_PATH):
        # 尝试查找其他版本的模型
        possible_models = [
            "ppo_stock_v8.zip",
            "ppo_stock_v9.zip",
            "ppo_stock_v7.zip",
            "models_v7/best/best_model.zip"
        ]
        for model_file in possible_models:
            if os.path.exists(model_file):
                MODEL_PATH = model_file
                break
        else:
            raise FileNotFoundError(f"未找到模型文件，请检查: {MODEL_PATH}")
    
    model = PPO.load(MODEL_PATH)
    
    # 检测模型版本（通过观察空间维度）
    obs_shape = model.observation_space.shape
    if len(obs_shape) == 1:
        if obs_shape[0] == 29:
            MODEL_VERSION = "V8"  # V8: 29维（21技术指标 + 8 LLM特征）
        elif obs_shape[0] > 100:
            MODEL_VERSION = "V9"  # V9: 历史窗口 + LLM特征
        elif obs_shape[0] == 126:
            MODEL_VERSION = "V7"  # V7: 126维价格序列
        else:
            MODEL_VERSION = "UNKNOWN"
    else:
        MODEL_VERSION = "UNKNOWN"
    
    print(f"✅ 模型加载成功！")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   模型版本: {MODEL_VERSION}")
    print(f"   观察空间: {obs_shape}")

# 初始化 LLM 市场情报代理
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
        print(f"✅ LLM 市场情报代理初始化成功！")
        print(f"   提供商: {LLM_PROVIDER.upper()}")
        print(f"   API 密钥: 已配置（使用真实 DeepSeek API）")
        print(f"   数据模式: 真实数据（非模拟）")
        print(f"   缓存: 已启用")
    except Exception as e:
        print(f"⚠️  LLM 初始化失败: {e}")
        print("   将仅使用技术指标进行预测")
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

# 重试函数
def fetch_data_with_retry(max_retries=3, extend_days=0):
    """获取股票数据，支持扩展日期范围"""
    for attempt in range(max_retries):
        try:
            # 获取最近的交易日作为结束日期
            end_date_obj = get_recent_trading_date(extend_days)
            end_date = end_date_obj.strftime('%Y-%m-%d')
            # 开始日期：往前推7天
            start_date = (end_date_obj - datetime.timedelta(days=7)).strftime('%Y-%m-%d')

            rs = bs.query_history_k_data_plus(
                STOCK_CODE, 
                "date,time,close,volume", 
                start_date=start_date, 
                end_date=end_date, 
                frequency='5', 
                adjustflag='3'
            )
            
            # 检查错误码
            if rs.error_code != '0':
                error_msg = f"baostock 错误: {rs.error_msg}"
                if attempt < max_retries - 1:
                    print(f"尝试 {attempt+1}/{max_retries} 失败: {error_msg}")
                    time.sleep(5 + random.uniform(0, 5))
                    continue
                else:
                    raise Exception(error_msg)
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                time.sleep(5 + random.uniform(0, 5))
            else:
                raise Exception(f"数据获取失败，已达最大重试次数: {e}")
    raise Exception("数据获取失败，已达最大重试次数")

# 动作映射函数（根据模型版本调整）
def map_action_to_operation(action, model_version="V7"):
    """将动作映射到具体操作"""
    if model_version in ["V8", "V9"]:
        # V8/V9: 7个动作
        if action == 0: return "持有"
        elif action == 1: return "买入 25%"
        elif action == 2: return "买入 50%"
        elif action == 3: return "买入 100%"
        elif action == 4: return "卖出 25%"
        elif action == 5: return "卖出 50%"
        elif action == 6: return "卖出 100%"
        else: return "未知动作"
    else:
        # V7: 9个动作
        if action == 0: return "卖出 100%"
        elif action == 1: return "卖出 50%"
        elif action == 2: return "卖出 25%"
        elif action == 3: return "持有"
        elif action == 4: return "持有"
        elif action == 5: return "持有"
        elif action == 6: return "买入 25%"
        elif action == 7: return "买入 50%"
        elif action == 8: return "买入 100%"
        else: return "未知动作"

# 构建观察向量（根据模型版本）
def build_observation(recent_closes, date_str=None, model_version="V7"):
    """
    构建观察向量
    
    Args:
        recent_closes: 价格序列
        date_str: 日期字符串（用于获取 LLM 情报）
        model_version: 模型版本
    """
    if model_version == "V7":
        # V7: 仅使用价格序列（126维）
        if len(recent_closes) < 126:
            last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
            padding = np.full(126 - len(recent_closes), last_value)
            recent_closes = np.concatenate((padding, recent_closes))
        obs = np.array(recent_closes[-126:], dtype=np.float32)
        
    elif model_version == "V8":
        # V8: 需要 29 维（21技术指标 + 8 LLM特征）
        # 这里简化处理：使用价格序列的前21个值作为技术指标占位
        # 实际应用中应该计算真实的技术指标
        if len(recent_closes) < 21:
            last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
            padding = np.full(21 - len(recent_closes), last_value)
            recent_closes = np.concatenate((padding, recent_closes))
        
        # 技术指标部分（简化版，实际应计算 MA、RSI、MACD 等）
        tech_features = recent_closes[-21:] / recent_closes[-1] - 1.0  # 归一化
        
        # LLM 市场情报部分（8维）
        llm_features = np.zeros(8, dtype=np.float32)
        if llm_agent and date_str:
            try:
                intelligence = llm_agent.get_market_intelligence(date_str, force_refresh=False)
                llm_features = np.array(llm_agent.get_feature_vector(intelligence), dtype=np.float32)
            except Exception as e:
                print(f"   [LLM警告] 获取市场情报失败: {e}")
        
        obs = np.concatenate([tech_features, llm_features]).astype(np.float32)
        
    elif model_version == "V9":
        # V9: 历史窗口 + 持仓信息 + LLM特征
        # 这里需要更复杂的处理，暂时使用简化版本
        # 实际应用中应该使用完整的环境来构建观察
        if len(recent_closes) < 126:
            last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
            padding = np.full(126 - len(recent_closes), last_value)
            recent_closes = np.concatenate((padding, recent_closes))
        obs = np.array(recent_closes[-126:], dtype=np.float32)
        
    else:
        # 默认：使用价格序列
        if len(recent_closes) < 126:
            last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
            padding = np.full(126 - len(recent_closes), last_value)
            recent_closes = np.concatenate((padding, recent_closes))
        obs = np.array(recent_closes[-126:], dtype=np.float32)
    
    return obs

# 格式化市场情报显示
def format_intelligence(intelligence):
    """格式化市场情报信息用于显示"""
    if not intelligence:
        return ""
    
    lines = []
    lines.append(f"   📊 市场情报:")
    lines.append(f"      宏观经济: {intelligence.get('macro_economic_score', 0):+.3f}")
    lines.append(f"      市场情绪: {intelligence.get('market_sentiment_score', 0):+.3f}")
    lines.append(f"      风险等级: {intelligence.get('risk_level', 0.5):.3f}")
    lines.append(f"      政策影响: {intelligence.get('policy_impact_score', 0):+.3f}")
    lines.append(f"      突发事件: {intelligence.get('emergency_impact_score', 0):+.3f}")
    lines.append(f"      资金流向: {intelligence.get('capital_flow_score', 0):+.3f}")
    lines.append(f"      国际联动: {intelligence.get('international_correlation', 0.5):.3f}")
    lines.append(f"      VIX水平: {intelligence.get('vix_level', 20):.2f}")
    
    if 'reasoning' in intelligence:
        lines.append(f"      分析理由: {intelligence['reasoning']}")
    
    return "\n".join(lines)

# ==================== 主循环 ====================
consecutive_empty_count = 0  # 连续空数据计数
max_empty_before_extend = 3  # 连续3次空数据后扩展日期范围
last_day = None  # 上一个交易日
last_action = None  # 上一个动作，用于检测变化

# 模拟持仓和盈亏统计
initial_balance = 100000.0  # 初始资金
current_balance = initial_balance
shares_held = 0.0  # 当前持股数
last_price = 0.0  # 上次价格，用于计算盈亏
daily_pnl = 0.0  # 每日盈亏
daily_pnl_history = []  # 存储每日盈亏记录

print("🚀 开始实时预测循环...\n")

while True:
    try:
        current_time = datetime.datetime.now()
        is_weekend = current_time.weekday() >= 5
        is_trading = is_trading_time()
        
        # 如果连续多次获取不到数据，尝试扩展日期范围
        extend_days = min(consecutive_empty_count // max_empty_before_extend, 5)
        
        df = fetch_data_with_retry(extend_days=extend_days)
        
        if not df.empty and len(df) > 0:
            # 重置连续空数据计数
            consecutive_empty_count = 0
            
            df = df.sort_values('time')  # 按时间排序
            recent_closes = df['close'].astype(float).values  # 所有 close
            
            # 获取最新数据的日期和时间
            latest_date = df['date'].iloc[-1] if 'date' in df.columns else '未知'
            latest_time = df['time'].iloc[-1] if 'time' in df.columns else '未知'
            
            # 构建观察向量
            obs = build_observation(recent_closes, latest_date, MODEL_VERSION)
            
            # 模型预测
            action, _states = model.predict(obs, deterministic=True)
            operation = map_action_to_operation(action, MODEL_VERSION)
            current_price = recent_closes[-1]
            volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
            
            # 获取市场情报（用于显示）
            intelligence = None
            if llm_agent and latest_date != '未知':
                try:
                    intelligence = llm_agent.get_market_intelligence(latest_date, force_refresh=False)
                except Exception as e:
                    pass  # 静默失败，不影响主流程
            
            # 检查动作是否变化
            if last_action is not None and operation != last_action:
                print(f"⚠️  动作变化！从 {last_action} 变为 {operation}")
                # 用颜色突出（ANSI 红色）
                print(f"\033[91m✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, 价格: {current_price:.2f}, 成交量: {volume:.0f}, 预测动作: {operation}\033[0m")
            else:
                print(f"✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, 价格: {current_price:.2f}, 成交量: {volume:.0f}, 预测动作: {operation}")
            
            print(f"   数据日期: {latest_date}, 数据时间: {latest_time}, 数据条数: {len(df)}")
            
            # 显示市场情报（如果可用）
            if intelligence:
                print(format_intelligence(intelligence))
            
            last_action = operation  # 更新上次动作
            
            # 模拟交易执行
            if "买入" in operation:
                buy_percentage = float(operation.split()[-1][:-1]) / 100  # e.g., 25% -> 0.25
                buy_amount = current_balance * buy_percentage
                shares_bought = buy_amount / current_price
                shares_held += shares_bought
                current_balance -= buy_amount
            elif "卖出" in operation:
                sell_percentage = float(operation.split()[-1][:-1]) / 100
                shares_sold = shares_held * sell_percentage
                sell_amount = shares_sold * current_price
                shares_held -= shares_sold
                current_balance += sell_amount
            
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
            
            print()  # 空行分隔
            
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

# baostock 登出
bs.logout()
print("\n✅ 程序已退出")

