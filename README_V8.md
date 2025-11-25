# V8 版本 - LLM 增强的深度强化学习股票交易系统

## 🌟 核心创新

V8 版本在 V7 的基础上，集成了**大语言模型（LLM）**作为市场情报分析师，为强化学习模型提供宏观经济、新闻舆情、市场情绪等关键信息。

### 新增能力

1. **宏观经济分析**: GDP、CPI、利率政策影响评估
2. **新闻舆情分析**: 市场热点、负面消息识别
3. **市场情绪指标**: 恐慌指数 VIX、投资者情绪
4. **资金流向分析**: 外资、融资融券、北向资金
5. **政策变化跟踪**: 货币政策、财政政策、监管政策
6. **国际市场联动**: 与美股、港股的相关性
7. **突发事件应对**: 地缘政治、疫情、自然灾害

---

## 📊 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    PPO 强化学习模型                       │
│                                                           │
│  输入: 29维观察空间                                        │
│  ├─ 21维技术指标 (价格、成交量、RSI、MACD等)               │
│  └─ 8维LLM市场情报 (宏观、情绪、风险、政策等)              │
│                                                           │
│  输出: 7个离散动作                                         │
│  ├─ 0: 持有                                               │
│  ├─ 1-3: 买入 25%/50%/100%                                │
│  └─ 4-6: 卖出 25%/50%/100%                                │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
                   ┌────────┴────────┐
                   │                 │
        ┌──────────▼──────┐  ┌──────▼─────────┐
        │  技术指标提取    │  │  LLM情报代理    │
        │                 │  │                 │
        │ • MA5/10/20     │  │ • DeepSeek API  │
        │ • RSI/MACD      │  │ • Grok API      │
        │ • 布林带         │  │ • 智能缓存       │
        │ • 波动率         │  │ • 模拟模式       │
        └─────────────────┘  └─────────────────┘
                   │                 │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   历史股票数据   │
                   │                 │
                   │ • Baostock      │
                   │ • AkShare       │
                   └─────────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install stable-baselines3[extra]
pip install gymnasium pandas numpy
pip install requests  # LLM API 调用

# 可选：如果使用真实 LLM API
export DEEPSEEK_API_KEY="your_deepseek_api_key"
# 或
export GROK_API_KEY="your_grok_api_key"
```

### 2. 数据准备

#### 选项 A: 使用已有数据（推荐）

如果您已经运行过 V7，数据已经在 `stockdata_v7/` 目录中：

```bash
# 检查数据
ls stockdata_v7/train/
ls stockdata_v7/test/
```

#### 选项 B: 重新下载数据

```bash
# 使用 V7 的数据获取脚本
python get_etf_data_akshare.py
```

### 3. 批量生成市场情报（可选）

为了避免训练时频繁调用 LLM API，可以预先批量生成历史市场情报缓存：

```python
from llm_market_intelligence import MarketIntelligenceAgent

# 使用模拟模式（免费，训练时推荐）
agent = MarketIntelligenceAgent(provider="deepseek", enable_cache=True)

# 生成 2020-2025 的市场情报
agent.batch_generate_intelligence(
    start_date="2020-01-01",
    end_date="2025-11-24",
    use_mock=True  # 使用模拟数据，避免 API 成本
)
```

**注意**: 模拟数据会根据日期生成具有合理波动和趋势性的数据，适合训练使用。

### 4. 开始训练

#### 模式 A: 使用模拟 LLM 数据（推荐用于训练）

```bash
# 默认使用模拟数据（无需 API 密钥）
python train_v8.py
```

训练参数：
- **总步数**: 300万步
- **并行环境**: 8个
- **训练时间**: 约 2-4 小时（取决于硬件）

#### 模式 B: 使用真实 LLM API

```bash
# 设置 API 密钥
export DEEPSEEK_API_KEY="your_key_here"

# 修改 train_v8.py 中的 LLM_API_KEY
# 然后运行
python train_v8.py
```

**成本估算**（DeepSeek）:
- 每次调用约 0.001 元
- 训练期间每天调用 1 次 × 2000 天 ≈ 2 元
- 实盘每日调用 1 次 ≈ 0.001 元/天

### 5. 监控训练

```bash
# 实时监控训练指标
tensorboard --logdir=./logs_v8/

# 浏览器打开: http://localhost:6006
```

关键指标：
- `eval/mean_reward`: 评估奖励（应逐渐上升）
- `train/value_loss`: 价值网络损失
- `rollout/ep_rew_mean`: 回合平均奖励

### 6. 评估模型

训练完成后自动评估，或手动评估：

```bash
python train_v8.py eval
```

输出示例：

```
✓ sh.600036.招商银行
  最终净值: 112,450 元
  总收益率: +12.45%
  最大回撤: 5.30%
  夏普比率: 1.58
  交易次数: 8
  胜率: 62.50%
  风险事件: 0

平均收益率: +13.20%
平均最大回撤: 6.10%
平均夏普比率: 1.52
```

---

## 🎯 LLM 市场情报说明

### 情报维度

| 维度 | 范围 | 说明 | 影响 |
|------|------|------|------|
| **宏观经济评分** | -1 到 1 | GDP、CPI、利率综合 | 影响整体仓位 |
| **市场情绪评分** | -1 到 1 | 投资者情绪、VIX | 影响买卖决策 |
| **风险等级** | 0 到 1 | 整体市场风险 | 高风险时降低买入 |
| **政策影响评分** | -1 到 1 | 货币/财政政策 | 影响长期持仓 |
| **突发事件影响** | -1 到 1 | 地缘、疫情等 | 触发止损保护 |
| **资金流向评分** | -1 到 1 | 外资、融资融券 | 影响买卖时机 |
| **国际联动系数** | 0 到 1 | 与美股/港股相关性 | 参考外围市场 |
| **VIX水平** | 10-40 | 恐慌指数 | 风险预警 |

### 奖励函数集成

V8 的奖励函数在 V7 的基础上，增加了 LLM 信号：

```python
奖励 = 基础收益奖励 
     + 回撤惩罚（-10/-3/-0.5）
     + LLM情绪与动作匹配奖励（±0.2）
     + LLM宏观环境奖励（±0.1）
     + LLM突发事件应对奖励（+0.15）
     + 交易激励（+0.05）
     + 盈利奖励（+0.5/+1.0）
```

**示例**:
- 在积极市场（情绪>0.3，风险<0.5）买入 → +0.2 奖励
- 在高风险期（风险>0.6）卖出 → +0.2 奖励
- 在突发负面事件时降低仓位 → +0.15 奖励

---

## 📁 文件结构

```
RL-Stock/
├── llm_market_intelligence.py    # LLM 市场情报代理
├── stock_env_v8.py                # V8 交易环境
├── train_v8.py                    # V8 训练脚本
├── README_V8.md                   # 本文档
├── V8_配置指南.md                 # 详细配置说明
│
├── market_intelligence_cache/     # LLM 情报缓存（自动生成）
│   ├── 2024-01-01_deepseek.json
│   ├── 2024-01-02_deepseek.json
│   └── ...
│
├── stockdata_v7/                  # 股票数据（复用 V7）
│   ├── train/
│   └── test/
│
├── models_v8/                     # 模型检查点（自动生成）
│   ├── ppo_v8_checkpoint_100000_steps.zip
│   └── ...
│
├── logs_v8/                       # 训练日志（自动生成）
│   └── PPO_1/
│
└── ppo_stock_v8.zip              # 最终训练模型
```

---

## ⚙️ 配置选项

### LLM 提供商

#### DeepSeek（推荐）

**优点**:
- 兼容 OpenAI API 格式
- 价格便宜（0.001元/千tokens）
- 中文支持好
- 金融知识丰富

**配置**:

```python
# 在 train_v8.py 中
LLM_PROVIDER = "deepseek"
LLM_API_KEY = "your_deepseek_api_key"  # 或设置环境变量
```

**获取 API Key**:
1. 访问 https://platform.deepseek.com
2. 注册账号
3. 创建 API Key
4. 充值（建议 10 元起）

#### Grok（备选）

**优点**:
- X.AI 出品，实时性强
- 可访问 X（Twitter）数据
- 对突发事件响应快

**配置**:

```python
# 在 train_v8.py 中
LLM_PROVIDER = "grok"
LLM_API_KEY = "your_grok_api_key"
```

#### 模拟模式（免费）

如果不想使用 LLM API，可以使用模拟数据：

```python
# 在 train_v8.py 中
LLM_API_KEY = None  # 自动进入模拟模式
```

模拟数据特点：
- 根据日期生成确定性数据
- 包含合理的趋势和波动
- 完全免费
- 适合模型训练

### 调整 LLM 权重

如果您认为 LLM 信号过强或过弱，可以调整权重：

```python
# 在 stock_env_v8.py 的 __init__ 中
env = StockTradingEnvV8(
    ...,
    llm_weight=0.3  # 默认 0.3，范围 0.0-1.0
)
```

建议值：
- `0.0`: 完全忽略 LLM（退化为 V7）
- `0.1-0.2`: 轻微参考
- `0.3-0.5`: 平衡参考（推荐）
- `0.6-1.0`: 重度依赖 LLM

---

## 🔬 训练策略

### 1. 两阶段训练（推荐）

**阶段 1: 基础训练（模拟 LLM）**

```bash
# 使用模拟数据快速训练 200 万步
python train_v8.py
```

**阶段 2: 精调（真实 LLM）**

```bash
# 加载预训练模型，使用真实 API 精调 50 万步
# 修改 train_v8.py:
# model = PPO.load("ppo_stock_v8.zip", env=env)
# TOTAL_TIMESTEPS = 500_000
```

### 2. 增量训练

如果训练中断，可以继续训练：

```python
# 在 train_v8.py 的 train() 函数中修改：
model = PPO.load("ppo_stock_v8.zip", env=env)
model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
```

### 3. 超参数调优

根据您的股票池特点，调整参数：

**高波动股票**（如小盘股、科技股）:
- 增大 `ent_coef` 到 0.02（更多探索）
- 降低 `clip_range` 到 0.15（更保守）
- 增加 `llm_weight` 到 0.5（更依赖风险信号）

**低波动股票**（如银行、公用事业）:
- 降低 `ent_coef` 到 0.005（更少探索）
- 增加 `clip_range` 到 0.25（更激进）
- 降低 `llm_weight` 到 0.2（更依赖技术指标）

---

## 📈 性能优化

### 缓存管理

LLM 情报会自动缓存到 `market_intelligence_cache/`，加速训练：

```bash
# 查看缓存统计
ls market_intelligence_cache/ | wc -l

# 清理缓存（重新生成）
rm -rf market_intelligence_cache/
```

### 并行环境数

根据您的 CPU 核心数调整：

```python
# 在 train_v8.py 中
N_ENVS = 8  # 推荐: CPU核心数 - 2
```

- 4核CPU → `N_ENVS = 2-4`
- 8核CPU → `N_ENVS = 6-8`
- 16核CPU → `N_ENVS = 12-16`

### GPU 加速

PPO 支持 GPU，但对于本项目的网络规模，CPU 通常已足够快。如果想使用 GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 训练会自动使用 GPU
```

---

## 🎓 进阶使用

### 自定义 LLM Prompt

编辑 `llm_market_intelligence.py` 中的 `_call_llm_api()` 函数，修改 prompt：

```python
prompt = f"""你是资深金融分析师。请分析 {date} 的A股市场：

【您的自定义分析维度】
1. ...
2. ...

返回 JSON:
{{
  "your_custom_score": 0.5,
  ...
}}"""
```

然后在 `stock_env_v8.py` 中使用这些自定义字段。

### 实盘部署

1. **每日更新情报**:

```python
from llm_market_intelligence import MarketIntelligenceAgent
from datetime import datetime

agent = MarketIntelligenceAgent(provider="deepseek")
today = datetime.now().strftime('%Y-%m-%d')
intelligence = agent.get_market_intelligence(today, force_refresh=True)
```

2. **实盘交易接口**:

```python
# 加载训练好的模型
model = PPO.load("ppo_stock_v8.zip")

# 获取实时数据（您需要对接券商 API）
live_data = get_realtime_market_data()

# 创建单步环境
env = StockTradingEnvV8(...)
obs = env.process_live_data(live_data)

# 预测动作
action, _ = model.predict(obs, deterministic=True)

# 执行交易
execute_trade(action)
```

### 多模型集成

训练多个模型并投票：

```python
models = [
    PPO.load("ppo_stock_v8_model1.zip"),
    PPO.load("ppo_stock_v8_model2.zip"),
    PPO.load("ppo_stock_v8_model3.zip"),
]

actions = [model.predict(obs)[0] for model in models]
final_action = np.bincount(actions).argmax()  # 多数投票
```

---

## ⚠️ 风险提示

1. **模拟环境 ≠ 实盘**
   - 回测收益不代表实盘收益
   - 滑点、成交延迟、市场冲击未完全模拟

2. **LLM 的局限性**
   - LLM 的分析基于训练数据，可能滞后
   - 极端事件（如黑天鹅）难以预测
   - 模拟模式仅用于训练，实盘需真实 API

3. **监管合规**
   - 请确保您的交易符合当地法规
   - 部分算法交易可能需要报备

4. **资金管理**
   - 建议小资金测试（1-10万）
   - 设置止损线（如 -20%）
   - 不要投入全部资金

---

## 🤝 版本对比

| 特性 | V7 | V8 |
|------|----|----|
| 观察空间 | 21 维 | **29 维**（+8 LLM） |
| 市场感知 | 技术指标 | **技术指标 + LLM 情报** |
| 风险管理 | 回撤控制 | **回撤 + 情绪 + 宏观** |
| 突发事件 | 无法应对 | **LLM 分析突发影响** |
| 国际联动 | 无 | **美股/港股联动** |
| 政策敏感 | 弱 | **强（LLM 分析政策）** |
| 训练成本 | 无额外成本 | **+2元**（可用模拟免费） |

---

## 📞 常见问题

**Q1: 训练时出现 "未设置 API 密钥" 警告？**

A: 这是正常的，系统会自动使用模拟数据。如果您想使用真实 LLM，请设置环境变量：

```bash
export DEEPSEEK_API_KEY="your_key"
```

**Q2: 如何验证 LLM 是否生效？**

A: 检查 `market_intelligence_cache/` 目录，查看缓存的 JSON 文件中 `"source"` 字段：
- `"mock_data"`: 模拟模式
- `"deepseek"` 或 `"grok"`: 真实 API

**Q3: 训练很慢怎么办？**

A: 
1. 减少 `TOTAL_TIMESTEPS`（如 150 万步）
2. 减少 `N_ENVS`（如 4 个）
3. 使用更少的股票

**Q4: 如何选择 DeepSeek 还是 Grok？**

A:
- **DeepSeek**: 适合中国A股，中文支持好，价格便宜，推荐大多数用户
- **Grok**: 适合需要实时X（Twitter）数据的场景，对国际市场更敏感

**Q5: 模拟数据和真实 LLM 差距大吗？**

A: 训练阶段差距不大（模拟数据已包含合理趋势）。但实盘时建议使用真实 LLM，因为需要感知最新的市场变化。

---

## 📚 相关资源

- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [Grok API 文档](https://docs.x.ai/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)

---

## 🎉 更新日志

### V8.0 (2025-11-24)

**新增**:
- ✨ 集成 LLM 市场情报分析
- ✨ 支持 DeepSeek 和 Grok API
- ✨ 8 维市场情报特征
- ✨ 智能缓存机制
- ✨ 模拟数据模式（免费训练）

**改进**:
- 🔧 观察空间扩展到 29 维
- 🔧 奖励函数集成 LLM 信号
- 🔧 增强风险管理能力

**优化**:
- ⚡ 批量情报生成
- ⚡ 自动缓存加载
- ⚡ 降低 API 调用成本

---

## 📄 许可

本项目仅供学习和研究使用。实盘交易风险自负。

---

**祝您交易顺利！** 📈



