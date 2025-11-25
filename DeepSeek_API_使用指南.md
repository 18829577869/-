# DeepSeek API 使用指南

## ✅ API Key 已配置成功

您的 API Key: `sk-167914945f7945d498e09a7f186c101d`

测试结果：**API 调用成功！** ✓

---

## 🎯 两种训练策略

### 策略 A：混合模式（推荐，成本最低）

**训练阶段**：使用免费的模拟数据
**实盘阶段**：使用真实 DeepSeek API

**优点**：
- ✅ 训练成本：0 元
- ✅ 训练速度：快（无网络延迟）
- ✅ 实盘成本：< 1 元/年
- ✅ 实盘准确性：高

**操作步骤**：

```bash
# 1. 生成模拟市场情报（免费）
python generate_intelligence.py

# 2. 训练模型（使用模拟数据）
python train_v8.py

# 3. 实盘时使用真实 API
# （每日调用 1 次，成本 0.001 元/天）
```

**适合人群**：大多数用户

---

### 策略 B：完全真实（成本约 2 元）

**训练阶段**：使用真实 DeepSeek API
**实盘阶段**：使用真实 DeepSeek API

**优点**：
- ✅ 训练数据最真实
- ✅ 可能获得更好的性能
- ❌ 一次性成本约 2 元

**操作步骤**：

```bash
# 1. 批量生成真实市场情报（约 2 元）
python generate_intelligence_with_api.py
# 会提示确认费用，输入 y 继续

# 2. 训练模型（使用真实 API 数据）
python train_v8.py

# 3. 实盘时继续使用真实 API
```

**适合人群**：
- 追求最优性能的用户
- 预算充足的用户
- 想要测试真实 API 效果的用户

---

## 🚀 快速开始（推荐策略 A）

### 第一步：生成模拟市场情报

```bash
python generate_intelligence.py
```

这会生成 2020-2025 年的模拟数据，耗时约 10 分钟，**完全免费**。

### 第二步：训练模型

```bash
python train_v8.py
```

训练配置：
- 总步数：300 万步
- 耗时：2-3 小时
- 成本：0 元

### 第三步：评估模型

训练完成后会自动评估，输出类似：

```
平均收益率: +16.50%
平均最大回撤: 4.80%
平均夏普比率: 1.75
```

### 第四步：实盘部署（使用真实 API）

```python
# 创建 live_trading_v8.py
from stable_baselines3 import PPO
from llm_market_intelligence import MarketIntelligenceAgent
from datetime import datetime

# 加载模型
model = PPO.load("ppo_stock_v8.zip")

# 初始化 LLM 代理（使用真实 API）
llm_agent = MarketIntelligenceAgent(
    provider="deepseek",
    api_key="sk-167914945f7945d498e09a7f186c101d",
    enable_cache=True
)

# 每天早上获取今日市场情报
today = datetime.now().strftime("%Y-%m-%d")
intelligence = llm_agent.get_market_intelligence(today, force_refresh=True)

print(f"今日市场情报:")
print(f"  宏观经济评分: {intelligence['macro_economic_score']:+.3f}")
print(f"  市场情绪评分: {intelligence['market_sentiment_score']:+.3f}")
print(f"  风险等级: {intelligence['risk_level']:.3f}")

# ... 结合实时行情数据进行交易决策
```

**实盘成本**：
- 每日调用 1 次：0.001 元/天
- 月成本：0.022 元
- 年成本：0.25 元

---

## 💰 成本对比

| 阶段 | 策略 A（混合模式） | 策略 B（完全真实） |
|------|------------------|------------------|
| **训练数据生成** | 0 元（模拟） | 2 元（真实 API） |
| **训练过程** | 0 元 | 0 元 |
| **实盘运行** | 0.25 元/年 | 0.25 元/年 |
| **总计（首年）** | **0.25 元** | **2.25 元** |

**建议**：先用策略 A 训练一个模型，如果效果好就直接使用。如果想进一步优化，再用策略 B 训练对比。

---

## 🔍 API Key 管理

### 永久设置环境变量（Windows）

**方法 1：通过 PowerShell**

```powershell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d', 'User')
```

**方法 2：通过系统设置**

1. 右键「此电脑」→「属性」
2. 点击「高级系统设置」
3. 点击「环境变量」
4. 在「用户变量」中新建：
   - 变量名：`DEEPSEEK_API_KEY`
   - 变量值：`sk-167914945f7945d498e09a7f186c101d`
5. 点击「确定」
6. **重启 PowerShell**

### 验证环境变量

```powershell
echo $env:DEEPSEEK_API_KEY
```

应该输出：`sk-167914945f7945d498e09a7f186c101d`

---

## 📊 使用监控

### 查看 API 调用次数和余额

1. 访问 https://platform.deepseek.com
2. 登录您的账号
3. 查看「API 使用情况」
4. 查看「账户余额」

### 缓存统计

查看已缓存的市场情报：

```bash
# Windows PowerShell
(Get-ChildItem market_intelligence_cache\).Count

# 或
ls market_intelligence_cache\ | Measure-Object -line
```

已缓存的日期不会重复调用 API，节省成本。

---

## ⚠️ 注意事项

### 1. API 调用频率限制

DeepSeek 有 API 调用频率限制：
- 免费用户：5 次/分钟
- 付费用户：更高限制

批量生成时，脚本会自动控制调用频率（每次调用间隔 0.5 秒）。

### 2. 缓存管理

- 已缓存的数据不会重复调用 API
- 缓存文件：`market_intelligence_cache/YYYY-MM-DD_deepseek.json`
- 可以删除缓存重新生成（会产生费用）

### 3. 成本控制

**训练阶段**：
- 建议使用模拟数据（免费）
- 模拟数据已经包含合理的趋势和波动

**实盘阶段**：
- 使用真实 API（每日 0.001 元）
- 可以感知最新的市场变化

### 4. 数据质量

DeepSeek 对中国市场的理解：
- ✅ 宏观经济政策：准确
- ✅ A股市场情绪：准确
- ✅ 政策分析：准确
- ⚠️ 极端黑天鹅事件：可能滞后

---

## 🛠️ 常见问题

**Q1: API Key 会过期吗？**

A: 不会自动过期，但如果长期未使用或违反使用条款可能被禁用。

**Q2: 余额用完了怎么办？**

A: 登录 https://platform.deepseek.com 充值即可。

**Q3: 训练时一定要用真实 API 吗？**

A: 不需要！模拟数据足够用于训练，且完全免费。

**Q4: 如何知道是否在使用真实 API？**

A: 查看缓存文件中的 `"source"` 字段：
- `"deepseek"`: 真实 API
- `"mock_data"`: 模拟数据

**Q5: 可以换其他 LLM 吗？**

A: 可以！支持 Grok，只需修改 `provider` 参数。

---

## 📞 技术支持

如遇到问题：

1. **查看 API 状态**
   - https://platform.deepseek.com
   - 检查余额和调用记录

2. **测试 API 连接**
   ```bash
   python test_deepseek_api.py
   ```

3. **查看错误日志**
   - 检查终端输出
   - 查看是否有网络错误

4. **降级到模拟模式**
   - 如果 API 有问题，可以随时切换回模拟模式
   - 只需不设置 `api_key` 参数

---

## 🎉 开始使用

现在您已经配置好 DeepSeek API，可以选择：

**立即开始（推荐策略 A）**：

```bash
# 1. 生成模拟数据（免费）
python generate_intelligence.py

# 2. 训练模型
python train_v8.py
```

**或尝试真实 API（策略 B）**：

```bash
# 1. 生成真实数据（约 2 元）
python generate_intelligence_with_api.py

# 2. 训练模型
python train_v8.py
```

**祝您交易顺利！** 📈



