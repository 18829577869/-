"""
V4 实时预测脚本 - V7 模型 + LLM 市场情报参考
使用 V7 模型（126维价格序列）进行预测
LLM 市场情报作为决策参考信息显示，不输入到模型中

支持的市场情报参考：
1. ✅ 宏观经济数据 - GDP、CPI、利率政策分析
2. ✅ 新闻和舆情分析 - 市场热点、负面消息识别
3. ✅ 市场情绪指标 - 恐慌指数 VIX、投资者情绪
4. ✅ 资金流向数据 - 外资、融资融券、北向资金
5. ✅ 政策变化信息 - 货币/财政/监管政策影响
6. ✅ 国际市场联动 - 美股、港股相关性分析
7. ✅ 突发事件应对 - 地缘政治、疫情、自然灾害

⚠️ 数据源说明：
   baostock 是免费的历史数据源，有以下限制：
   - 不支持实时数据，通常有 1-2 天延迟
   - 5分钟K线数据可能无法获取当天数据
   - 如需实时数据，建议使用 Tushare 或 AkShare（需要注册和积分）
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
MODEL_PATH = "ppo_stock_v7.zip"  # 使用 V7 模型
STOCK_CODE = 'sh.600036'  # 股票代码
LLM_PROVIDER = "deepseek"  # LLM 提供商：deepseek 或 grok
ENABLE_LLM = True  # 是否启用 LLM 市场情报（作为参考）
DEEPSEEK_API_KEY = "sk-167914945f7945d498e09a7f186c101d"  # DeepSeek API 密钥

# ⚠️ 数据源说明：
# baostock 是免费的历史数据源，但有以下限制：
# 1. 不支持实时数据，通常有 1-2 天延迟
# 2. 5分钟K线数据可能无法获取当天数据
# 3. 如需实时数据，建议使用 Tushare 或 AkShare（需要注册和积分）
#
# 替代方案（获取实时数据）：
# - Tushare: https://tushare.pro/ (需要注册，部分功能需要积分)
# - AkShare: https://akshare.akfamily.xyz/ (免费，但可能有访问限制)
# - 券商API: 部分券商提供实时行情API（需要开户）

# ==================== 初始化 ====================
print("=" * 70)
print("V4 实时预测系统 - V7 模型 + LLM 市场情报参考版")
print("=" * 70)
print("📌 模型: V7 (126维价格序列)")
print("📌 LLM 情报: 作为决策参考，不输入模型")
print("=" * 70)

# baostock 登录
bs.login()
print("✅ baostock 登录成功！")

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
            print(f"   ⚠️  警告: 当前为模拟模式，将使用模拟数据")
            print(f"   💡 提示: 请检查 API 密钥是否正确配置")
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

# 重试函数 - 获取最新数据（不指定具体日期）
def fetch_data_with_retry(max_retries=3, extend_days=0, try_today=True):
    """
    获取股票数据，优先获取最新数据
    使用较大的日期范围，然后只取最新的数据
    """
    for attempt in range(max_retries):
        try:
            today = datetime.date.today()
            today_str = today.strftime('%Y-%m-%d')
            
            # 使用一个较大的日期范围（最近30天），确保能获取到最新数据
            # 不指定具体的结束日期，使用今天作为结束日期
            end_date = today_str
            start_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            
            # 获取最近30天的5分钟K线数据
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
            
            if not data_list:
                # 如果没有数据，尝试使用最近交易日
                if attempt < max_retries - 1:
                    extend_days += 1
                    continue
                else:
                    raise Exception("未获取到任何数据")
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 按日期和时间排序，获取最新的数据
            if 'date' in df.columns and 'time' in df.columns:
                df = df.sort_values(['date', 'time'])
                # 确保返回至少126条数据（模型需要的维度）
                # 优先使用最新日期的数据，如果不足则用历史数据补充
                if len(df) >= 126:
                    # 数据充足，返回最后126条（包含最新数据）
                    return df.tail(126)
                else:
                    # 数据不足，返回所有可用数据（后续会用历史数据补充）
                    return df
            
            # 如果没有日期列，直接返回最后的数据
            if len(df) >= 126:
                return df.tail(126)
            else:
                return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                time.sleep(5 + random.uniform(0, 5))
            else:
                raise Exception(f"数据获取失败，已达最大重试次数: {e}")
    raise Exception("数据获取失败，已达最大重试次数")

# 动作映射函数（V7 模型：9个动作）
def map_action_to_operation(action):
    """将动作映射到具体操作（V7 模型）"""
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

# ==================== 主循环 ====================
consecutive_empty_count = 0  # 连续空数据计数
max_empty_before_extend = 3  # 连续3次空数据后扩展日期范围
last_day = None  # 上一个交易日
last_action = None  # 上一个动作，用于检测变化
last_price_value = None  # 上次价格值，用于检测价格变化
last_data_time = None  # 上次数据时间，用于检测数据更新

# 模拟持仓和盈亏统计
initial_balance = 100000.0  # 初始资金
current_balance = initial_balance
shares_held = 0.0  # 当前持股数
last_price = 0.0  # 上次价格，用于计算盈亏
daily_pnl = 0.0  # 每日盈亏
daily_pnl_history = []  # 存储每日盈亏记录

print("🚀 开始实时预测循环...")
print("📌 模型预测基于 V7 (126维价格序列)")
print("📌 LLM 情报仅作为参考，不影响模型预测")
print("📌 数据更新: 优先获取实时数据，如无则使用最近交易日数据")
print()
print("⚠️  数据源说明:")
print("   baostock 是免费历史数据源，不支持实时数据")
print("   - 通常有 1-2 天延迟")
print("   - 5分钟K线数据可能无法获取当天数据")
print("   - 如需实时数据，建议使用 Tushare 或 AkShare")
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
            
            # 获取最新数据的日期和时间
            latest_date = df['date'].iloc[-1] if 'date' in df.columns else '未知'
            latest_time = df['time'].iloc[-1] if 'time' in df.columns else '未知'
            current_price = recent_closes[-1]
            
            # 检测数据是否更新
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            is_realtime_data = (latest_date == today_str)
            data_updated = (last_data_time != latest_time or last_price_value != current_price)
            
            # 构建 V7 模型观察向量（126维价格序列）
            # 如果实时数据不足，用历史数据补充（保留实时数据）
            if len(recent_closes) < 126:
                # 需要补充的数据量
                need_more = 126 - len(recent_closes)
                realtime_count = len(recent_closes)
                
                # 优先从已获取的df中提取历史数据来补充
                # fetch_data_with_retry 已经获取了最近30天的数据，我们可以直接使用
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
                        # 获取更早的数据（往前推更多天）
                        latest_date_in_df = df['date'].iloc[0] if len(df) > 0 and 'date' in df.columns else None
                        if latest_date_in_df:
                            try:
                                date_obj = datetime.datetime.strptime(latest_date_in_df, '%Y-%m-%d').date()
                                end_date = (date_obj - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                                start_date = (date_obj - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
                                
                                rs_history = bs.query_history_k_data_plus(
                                    STOCK_CODE, 
                                    "date,time,close,volume", 
                                    start_date=start_date, 
                                    end_date=end_date, 
                                    frequency='5', 
                                    adjustflag='3'
                                )
                                
                                if rs_history.error_code == '0':
                                    history_list = []
                                    while rs_history.next():
                                        history_list.append(rs_history.get_row_data())
                                    
                                    if history_list:
                                        df_history = pd.DataFrame(history_list, columns=rs_history.fields)
                                        if 'close' in df_history.columns:
                                            df_history = df_history.sort_values(['date', 'time']) if 'date' in df_history.columns else df_history
                                            history_closes = df_history['close'].astype(float).values
                                            
                                            if len(history_closes) >= need_more:
                                                history_supplement = history_closes[-need_more:]
                                                recent_closes = np.concatenate([history_supplement, recent_closes])
                                                print(f"✅ 数据补充: 实时数据 {realtime_count} 条 + 历史数据 {need_more} 条 = {len(recent_closes)} 条")
                                            else:
                                                # 用历史数据的平均值填充剩余部分
                                                if len(history_closes) > 0:
                                                    avg_value = np.mean(history_closes)
                                                    remaining = need_more - len(history_closes)
                                                    padding = np.full(remaining, avg_value)
                                                    recent_closes = np.concatenate([padding, history_closes, recent_closes])
                                                    print(f"✅ 数据补充: 实时数据 {realtime_count} 条 + 历史数据 {len(history_closes)} 条 + 平均值填充 {remaining} 条")
                                                else:
                                                    # 用最后值填充
                                                    last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                                    padding = np.full(need_more, last_value)
                                                    recent_closes = np.concatenate([padding, recent_closes])
                                                    print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                                        else:
                                            # 用最后值填充
                                            last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                            padding = np.full(need_more, last_value)
                                            recent_closes = np.concatenate([padding, recent_closes])
                                            print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                                    else:
                                        # 用最后值填充
                                        last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                        padding = np.full(need_more, last_value)
                                        recent_closes = np.concatenate([padding, recent_closes])
                                        print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                                else:
                                    # 用最后值填充
                                    last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                    padding = np.full(need_more, last_value)
                                    recent_closes = np.concatenate([padding, recent_closes])
                                    print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条")
                            except Exception as e:
                                # 用最后值填充
                                last_value = recent_closes[-1] if len(recent_closes) > 0 else 0.0
                                padding = np.full(need_more, last_value)
                                recent_closes = np.concatenate([padding, recent_closes])
                                print(f"⚠️  数据补充: 实时数据 {realtime_count} 条 + 最后值填充 {need_more} 条（错误: {e}）")
                        else:
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
            
            # V7 模型预测（仅使用价格序列）
            action, _states = model.predict(obs, deterministic=True)
            operation = map_action_to_operation(action)
            volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
            
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
            
            # 显示预测结果
            print("=" * 70)
            
            # 数据更新状态提示
            if not data_updated:
                print(f"⚠️  数据未更新（与上次相同）")
            
            if last_action is not None and operation != last_action:
                print(f"⚠️  动作变化！从 {last_action} 变为 {operation}")
                # 用颜色突出（ANSI 红色）
                print(f"\033[91m✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, 价格: {current_price:.2f}, 成交量: {volume:.0f}, 预测动作: {operation}\033[0m")
            else:
                print(f"✅ 时间: {time.ctime()}, 股票: {STOCK_CODE}, 价格: {current_price:.2f}, 成交量: {volume:.0f}, 预测动作: {operation}")
            
            # 数据状态信息
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
            
            if is_realtime_data:
                data_status = "🟢 实时数据（今日）"
                data_status_detail = f"✅ 已获取到 {today_str} 的实时数据"
            else:
                # 计算数据日期与今天的差异
                try:
                    data_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d').date()
                    days_diff = (datetime.date.today() - data_date).days
                    if days_diff == 0:
                        data_status = "🟡 今日数据（可能未更新）"
                        data_status_detail = f"⚠️  数据日期为今天，但可能不是最新数据"
                    elif days_diff == 1:
                        data_status = "🟡 昨日数据"
                        data_status_detail = f"ℹ️  当前时间: {current_time_str}，数据日期: {latest_date}（{days_diff}天前）"
                    else:
                        data_status = "🟡 历史数据"
                        data_status_detail = f"ℹ️  当前时间: {current_time_str}，数据日期: {latest_date}（{days_diff}天前）"
                except:
                    data_status = "🟡 历史数据"
                    data_status_detail = f"ℹ️  数据日期: {latest_date}"
            
            print(f"   数据状态: {data_status}")
            print(f"   {data_status_detail}")
            print(f"   数据时间: {latest_time}, 数据条数: {len(df)}")
            print(f"   模型: V7 (126维价格序列)")
            
            # 如果是历史数据，给出原因说明
            if not is_realtime_data:
                if is_weekend:
                    print(f"   💡 原因: 今天是周末（非交易日）")
                elif not is_trading:
                    print(f"   💡 原因: 当前非交易时间（交易时间: 9:30-11:30, 13:00-15:00）")
                else:
                    print(f"   💡 原因: baostock 数据源限制（通常有 1-2 天延迟，不支持实时数据）")
                    print(f"   📌 说明: 这是 baostock 免费数据源的限制，不是代码问题")
                    print(f"   💡 建议: 如需实时数据，可使用 Tushare 或 AkShare（需要注册）")
            
            if not data_updated:
                print(f"   ⚠️  提示: 数据与上次相同，可能是非交易时间或数据源未更新")
            
            # 显示详细的市场情报参考（如果可用）
            if intelligence:
                print()
                print(format_intelligence_detailed(intelligence))
                print(f"   📌 数据来源: {intelligence_source} ({intelligence.get('source', 'unknown')})")
            else:
                print("   ℹ️  暂无市场情报参考（LLM 未启用或数据获取失败）")
            
            print()
            
            # 更新状态变量
            last_action = operation  # 更新上次动作
            last_price_value = current_price  # 更新上次价格值
            last_data_time = latest_time  # 更新上次数据时间
            
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

