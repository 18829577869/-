# ZCF量化炒股

## 🚀 项目现状（2025年更新）

经过多年的迭代和优化，项目已发展成为一个功能完善的**实时股票交易预测系统**，支持：

- ✅ **多版本模型**：V7（纯技术分析）、V8（技术+LLM）、V9（投资组合管理）
- ✅ **实时数据源**：支持 Tushare、AkShare、baostock 多数据源
- ✅ **LLM 市场情报**：集成 DeepSeek/Grok，提供7维度市场分析
- ✅ **操作记录系统**：自动记录和汇总所有交易操作
- ✅ **持仓管理**：支持手动更新持仓状态，同步外部交易
- ✅ **实时预测**：支持实时数据获取和预测

### 📊 最新性能

- **V7 模型回测收益**: +10.57%
- **支持实时数据**: Tushare/AkShare 实时数据源
- **多维度分析**: 技术指标 + LLM 市场情报

## 🚀 项目现状（2025年更新）

经过多年的迭代和优化，项目已发展成为一个功能完善的**实时股票交易预测系统**，支持：

- ✅ **多版本模型**：V7（纯技术分析）、V8（技术+LLM）、V9（投资组合管理）
- ✅ **实时数据源**：支持 Tushare、AkShare、baostock 多数据源
- ✅ **LLM 市场情报**：集成 DeepSeek/Grok，提供7维度市场分析
- ✅ **操作记录系统**：自动记录和汇总所有交易操作
- ✅ **持仓管理**：支持手动更新持仓状态，同步外部交易
- ✅ **实时预测**：支持实时数据获取和预测

### 📊 最新性能

- **V7 模型回测收益**: +10.57%
- **支持实时数据**: Tushare/AkShare 实时数据源
- **多维度分析**: 技术指标 + LLM 市场情报

## 📖 监督学习与强化学习的区别

监督学习（如 LSTM）可以根据各种历史数据来预测未来的股票的价格，判断股票是涨还是跌，帮助人做决策。

<img src="img/2020-03-25-18-55-13.png" alt="drawing" width="50%"/>

而强化学习是机器学习的另一个分支，在决策的时候采取合适的行动 (Action) 使最后的奖励最大化。与监督学习预测未来的数值不同，强化学习根据输入的状态（如当日开盘价、收盘价等），输出系列动作（例如：买进、持有、卖出），使得最后的收益最大化，实现自动交易。

<img src="img/2020-03-25-18-19-03.png" alt="drawing" width="50%"/>

## 🤖 系统架构

### 模型版本演进

| 版本 | 观察空间 | 特点 | 适用场景 |
|------|---------|------|---------|
| **V7** | 126维价格序列 | 纯技术分析，稳定可靠 | 新手入门，稳健投资 |
| **V8** | 29维（21技术+8LLM） | 技术+市场情报，功能全面 | 综合决策，进阶使用 |
| **V9** | 历史窗口+组合+LLM | 投资组合管理，复杂策略 | 高级用户，组合管理 |

### 动作空间

所有版本统一使用 **7个离散动作**：

- 0: 卖出 100%
- 1: 卖出 50%
- 2: 卖出 25%
- 3: 持有
- 4: 买入 25%
- 5: 买入 50%
- 6: 买入 100%

### 奖励函数

奖励函数基于持仓收益，考虑交易成本和风险控制：

```python
# 当前利润
reward = current_net_worth - initial_balance
# 风险惩罚
if drawdown > threshold:
    reward -= penalty
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository_url>
cd ZCF-RLSTOCK

# 安装依赖
pip install -r ZCF_RL_STOCK/requirements.txt
```

### 2. 数据准备

```bash
cd ZCF_RL_STOCK

# 使用 baostock（免费，推荐新手）
python get_stock_data_v7_myportfolio.py

# 或使用 Tushare（需要注册，支持实时数据）
python get_stock_data_v7_tushare.py
```

### 3. 模型训练（可选）

```bash
# 训练 V7 模型
python train_v7.py
```

或直接使用预训练模型（推荐）：
- `ppo_stock_v7.zip`
- `models_v7/best/best_model.zip`

### 4. 实时预测

```bash
# 使用 V6 版本（推荐，功能最全）
python real_time_predict_v6.py
```

## 📊 核心功能

### 1. 实时预测系统

- ✅ 多数据源支持（Tushare/AkShare/baostock）
- ✅ 自动操作记录和汇总
- ✅ 实时持仓信息显示
- ✅ LLM 市场情报参考

### 2. LLM 市场情报

提供 7 个维度的市场分析：

1. ✅ **宏观经济数据** - GDP、CPI、利率政策分析
2. ✅ **新闻和舆情分析** - 市场热点、负面消息识别
3. ✅ **市场情绪指标** - 恐慌指数 VIX、投资者情绪
4. ✅ **资金流向数据** - 外资、融资融券、北向资金
5. ✅ **政策变化信息** - 货币/财政/监管政策影响
6. ✅ **国际市场联动** - 美股、港股相关性分析
7. ✅ **突发事件应对** - 地缘政治、疫情、自然灾害

### 3. 操作记录系统

- 自动记录所有仓位变动
- 操作汇总和待执行列表
- 历史操作查询
- CSV 格式日志文件

### 4. 持仓管理

- 支持手动更新持仓状态
- 自动加载和保存持仓信息
- 同步外部交易记录

## 📁 项目结构

```
ZCF-RLSTOCK/
├── ZCF_RL_STOCK/              # 主代码目录
│   ├── rlenv/                 # 强化学习环境
│   ├── train_v7.py            # V7 训练脚本
│   ├── real_time_predict_v6.py  # V6 实时预测（推荐）
│   ├── llm_market_intelligence.py  # LLM 市场情报
│   ├── update_portfolio.py    # 持仓更新工具
│   ├── models_v7/             # V7 模型文件
│   └── stockdata_v7/          # 股票数据
├── docs/                      # 文档目录
│   ├── 快速开始.md
│   ├── 训练指南.md
│   ├── 实时预测.md
│   └── ...
├── models_v7/                 # 训练好的模型
├── stockdata_v7/              # 股票数据
└── README.md                  # 本文件
```

## 📚 文档导航

详细文档请查看 [docs/README.md](../docs/README.md)：

- [🚀 快速开始](../docs/快速开始.md) - 5分钟上手
- [🎯 训练指南](../docs/训练指南.md) - 模型训练教程
- [📈 实时预测](../docs/实时预测.md) - 实时预测系统
- [💼 实盘交易](../docs/实盘交易.md) - 实盘交易指南
- [💡 LLM 市场情报](../docs/LLM市场情报.md) - LLM 集成说明
- [💼 持仓管理](../docs/持仓管理.md) - 持仓状态管理
- [📥 数据获取](../docs/数据获取.md) - 数据源配置
- [🐛 问题排查](../docs/问题排查.md) - 常见问题解决

## 🕵️‍♀️ 验证结果

### V7 模型回测

- **初始本金**: 100,000 元
- **股票代码**: `sh.600036` (招商银行)
- **回测收益**: +10.57%
- **模型稳定性**: 高

### 实时预测效果

系统支持实时数据获取和预测，能够：
- 实时获取最新股票数据
- 自动执行交易操作
- 记录所有操作历史
- 提供市场情报参考

## ⚙️ 配置说明

### 环境变量

```bash
# DeepSeek API 密钥
export DEEPSEEK_API_KEY='your_api_key'

# Tushare Token（可选）
export TUSHARE_TOKEN='your_token'
```

### 主要配置

```python
# real_time_predict_v6.py
MODEL_PATH = "ppo_stock_v7.zip"  # 模型路径
STOCK_CODE = 'sh.600036'  # 股票代码
ENABLE_LLM = True  # 是否启用 LLM
DEEPSEEK_API_KEY = "your_api_key"  # API 密钥
```

## 🔧 数据源对比

| 数据源 | 实时数据 | 费用 | 需要注册 | 推荐度 |
|--------|---------|------|---------|--------|
| Tushare | ✅ | 免费/付费 | ✅ | ⭐⭐⭐⭐⭐ |
| AkShare | ✅ | 免费 | ❌ | ⭐⭐⭐⭐ |
| baostock | ❌ | 免费 | ❌ | ⭐⭐⭐ |

## ⚠️ 重要提示

1. **仅供学习研究**: 本系统仅供学习和研究使用，不构成投资建议
2. **风险自负**: 实盘交易存在风险，请谨慎操作，风险自负
3. **数据准确性**: 请确保数据源可靠，定期验证数据准确性
4. **模型局限性**: 模型基于历史数据训练，无法预测突发事件

## 🛠️ 工具脚本

### 持仓更新工具

```bash
# 交互式更新
python update_portfolio.py

# 命令行更新
python update_portfolio.py --stock sh.600036 --shares 500 --balance 80000 --price 43.25

# 查看当前状态
python update_portfolio.py --show
```

## 📈 版本历史

- **V7** (2024): 126维价格序列，纯技术分析，稳定可靠
- **V8** (2024): 集成 LLM 市场情报，29维特征空间
- **V9** (2024): 投资组合管理，复杂策略支持
- **V6 实时预测** (2025): 多数据源支持，操作记录，持仓管理

## 👻 最后

- 股票 Gym 环境主要参考 [Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)，对观测状态、奖励函数和训练集做了修改。
- 俺完全是股票没入门的新手，难免存在错误，欢迎指正！
- 数据和方法皆来源于网络，无法保证有效性，**Just For Fun**！
- **⚠️ 重要**: 本系统仅供学习和研究使用，不构成投资建议。实盘交易需谨慎，风险自负。

## 📚 参考资料

### 学术论文

- Y. Deng, F. Bao, Y. Kong, Z. Ren and Q. Dai, "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 3, pp. 653-664, March 2017.

- [Yuqin Dai, Chris Wang, Iris Wang, Yilun Xu, "Reinforcement Learning for FX trading"](http://stanford.edu/class/msande448/2019/Final_reports/gr2.pdf)

- Chien Yi Huang. Financial trading as a game: A deep reinforcement learning approach. arXiv preprint arXiv:1807.02787, 2018.

### 开源项目

- [notadamking/Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [baostock](http://baostock.com/) - 免费证券数据平台
- [Tushare](https://tushare.pro/) - 金融数据接口
- [AkShare](https://akshare.akfamily.xyz/) - 金融数据接口库

### 教程和文档

- [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- [Welcome to Stable Baselines docs! - RL Baselines Made Easy](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

详见 [LICENSE](LICENSE) 文件

---

**最后更新**: 2025-11-28  
**当前版本**: V6 实时预测系统  
**推荐模型**: V7 (稳定可靠) 或 V8 (功能全面)
