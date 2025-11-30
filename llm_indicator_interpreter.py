"""
基于大语言模型的技术指标解释模块
使用 LLM 对技术指标进行智能解释和分析
"""

import os
import json
from typing import Dict, Optional, List
from datetime import datetime


class LLMIndicatorInterpreter:
    """基于LLM的技术指标解释器"""
    
    def __init__(self, 
                 llm_agent=None,
                 enable_cache: bool = True,
                 cache_dir: str = "indicator_interpretation_cache"):
        """
        初始化指标解释器
        
        参数:
            llm_agent: LLM代理对象（如 MarketIntelligenceAgent）
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
        """
        self.llm_agent = llm_agent
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def interpret_indicators(self, 
                            indicators: Dict,
                            stock_code: str,
                            current_price: float,
                            force_refresh: bool = False) -> Dict:
        """
        解释技术指标
        
        参数:
            indicators: 技术指标字典（来自 TechnicalIndicators.get_indicator_summary）
            stock_code: 股票代码
            current_price: 当前价格
            force_refresh: 是否强制刷新（不使用缓存）
        
        返回:
            包含解释信息的字典
        """
        # 检查缓存
        cache_key = self._generate_cache_key(indicators, stock_code)
        if not force_refresh and self.enable_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                cached_result['source'] = 'cache'
                return cached_result
        
        # 如果没有LLM代理，返回基础解释
        if self.llm_agent is None:
            return self._generate_basic_interpretation(indicators)
        
        # 使用LLM生成解释
        try:
            interpretation = self._generate_llm_interpretation(
                indicators, stock_code, current_price
            )
            
            # 保存到缓存
            if self.enable_cache:
                self._save_to_cache(cache_key, interpretation)
            
            interpretation['source'] = 'llm'
            return interpretation
        except Exception as e:
            print(f"⚠️  LLM解释生成失败: {e}")
            return self._generate_basic_interpretation(indicators)
    
    def _generate_cache_key(self, indicators: Dict, stock_code: str) -> str:
        """生成缓存键"""
        # 使用关键指标值生成缓存键
        key_data = {
            'stock_code': stock_code,
            'kdj_k': round(indicators.get('KDJ', {}).get('K', 0), 1),
            'kdj_d': round(indicators.get('KDJ', {}).get('D', 0), 1),
            'rsi': round(indicators.get('RSI', 50), 1),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存加载"""
        try:
            import hashlib
            cache_file = os.path.join(
                self.cache_dir, 
                hashlib.md5(cache_key.encode()).hexdigest() + '.json'
            )
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def _save_to_cache(self, cache_key: str, interpretation: Dict):
        """保存到缓存"""
        try:
            import hashlib
            cache_file = os.path.join(
                self.cache_dir,
                hashlib.md5(cache_key.encode()).hexdigest() + '.json'
            )
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(interpretation, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _generate_basic_interpretation(self, indicators: Dict) -> Dict:
        """生成基础解释（不使用LLM）"""
        # 安全获取指标数据，处理类型不匹配的情况
        kdj = indicators.get('KDJ', {})
        if not isinstance(kdj, dict):
            kdj = {}
        
        rsi = indicators.get('RSI', 50)
        if not isinstance(rsi, (int, float)):
            rsi = 50
        
        obv = indicators.get('OBV', {})
        if not isinstance(obv, dict):
            obv = {}
        
        macd = indicators.get('MACD', {})
        if not isinstance(macd, dict):
            macd = {}
        
        # 分析各个指标
        kdj_analysis = self._analyze_kdj_basic(kdj)
        rsi_analysis = self._analyze_rsi_basic(rsi)
        obv_analysis = self._analyze_obv_basic(obv)
        macd_analysis = self._analyze_macd_basic(macd)
        
        # 构建interpretation字典
        interpretation = {
            'summary': '技术指标基础分析',
            'kdj_analysis': kdj_analysis,
            'rsi_analysis': rsi_analysis,
            'obv_analysis': obv_analysis,
            'macd_analysis': macd_analysis,
            'source': 'basic'
        }
        
        # 生成综合信号和交易建议（使用interpretation字典）
        interpretation['overall_signal'] = self._generate_overall_signal(interpretation)
        interpretation['trading_suggestion'] = self._generate_trading_suggestion(interpretation)
        
        return interpretation
    
    def _analyze_kdj_basic(self, kdj: Dict) -> Dict:
        """基础KDJ分析"""
        # 确保kdj是字典类型
        if not isinstance(kdj, dict):
            kdj = {}
        
        k = float(kdj.get('K', 50)) if kdj.get('K') is not None else 50
        d = float(kdj.get('D', 50)) if kdj.get('D') is not None else 50
        j = float(kdj.get('J', 50)) if kdj.get('J') is not None else 50
        
        analysis = {
            'values': {'K': k, 'D': d, 'J': j},
            'signal': '中性',
            'description': ''
        }
        
        # KDJ超买超卖判断
        if k > 80 and d > 80:
            analysis['signal'] = '超买'
            analysis['description'] = 'KDJ处于超买区域，可能面临回调压力'
        elif k < 20 and d < 20:
            analysis['signal'] = '超卖'
            analysis['description'] = 'KDJ处于超卖区域，可能出现反弹机会'
        elif k > d:
            analysis['signal'] = '看涨'
            analysis['description'] = 'K线上穿D线，形成看涨信号'
        elif k < d:
            analysis['signal'] = '看跌'
            analysis['description'] = 'K线下穿D线，形成看跌信号'
        else:
            analysis['description'] = 'KDJ指标处于中性区域，等待明确信号'
        
        return analysis
    
    def _analyze_rsi_basic(self, rsi: float) -> Dict:
        """基础RSI分析"""
        analysis = {
            'value': rsi,
            'signal': '中性',
            'description': ''
        }
        
        if rsi > 70:
            analysis['signal'] = '超买'
            analysis['description'] = f'RSI={rsi:.1f}，处于超买区域，可能面临回调'
        elif rsi < 30:
            analysis['signal'] = '超卖'
            analysis['description'] = f'RSI={rsi:.1f}，处于超卖区域，可能出现反弹'
        elif rsi > 50:
            analysis['signal'] = '偏强'
            analysis['description'] = f'RSI={rsi:.1f}，处于强势区域'
        else:
            analysis['signal'] = '偏弱'
            analysis['description'] = f'RSI={rsi:.1f}，处于弱势区域'
        
        return analysis
    
    def _analyze_obv_basic(self, obv: Dict) -> Dict:
        """基础OBV分析"""
        # 确保obv是字典类型
        if not isinstance(obv, dict):
            obv = {}
        
        obv_ratio = float(obv.get('OBV_Ratio', 1.0)) if obv.get('OBV_Ratio') is not None else 1.0
        
        analysis = {
            'obv_ratio': obv_ratio,
            'signal': '中性',
            'description': ''
        }
        
        if obv_ratio > 1.2:
            analysis['signal'] = '放量'
            analysis['description'] = f'OBV比率={obv_ratio:.2f}，显示成交量放大，资金活跃'
        elif obv_ratio < 0.8:
            analysis['signal'] = '缩量'
            analysis['description'] = f'OBV比率={obv_ratio:.2f}，显示成交量萎缩，资金观望'
        else:
            analysis['description'] = f'OBV比率={obv_ratio:.2f}，成交量正常'
        
        return analysis
    
    def _analyze_macd_basic(self, macd: Dict) -> Dict:
        """基础MACD分析"""
        # 确保macd是字典类型
        if not isinstance(macd, dict):
            macd = {}
        
        dif = float(macd.get('DIF', 0)) if macd.get('DIF') is not None else 0
        dea = float(macd.get('DEA', 0)) if macd.get('DEA') is not None else 0
        macd_value = float(macd.get('MACD', 0)) if macd.get('MACD') is not None else 0
        
        analysis = {
            'values': {'DIF': dif, 'DEA': dea, 'MACD': macd_value},
            'signal': '中性',
            'description': ''
        }
        
        if dif > dea and macd_value > 0:
            analysis['signal'] = '看涨'
            analysis['description'] = 'MACD金叉，柱状图为正，显示上升动能'
        elif dif < dea and macd_value < 0:
            analysis['signal'] = '看跌'
            analysis['description'] = 'MACD死叉，柱状图为负，显示下降动能'
        elif dif > dea:
            analysis['signal'] = '偏强'
            analysis['description'] = 'MACD处于金叉状态，但动能较弱'
        else:
            analysis['signal'] = '偏弱'
            analysis['description'] = 'MACD处于死叉状态，动能较弱'
        
        return analysis
    
    def _generate_overall_signal(self, indicators: Dict) -> str:
        """生成综合信号"""
        # 如果indicators是interpretation字典（包含分析结果），直接使用
        if 'kdj_analysis' in indicators:
            kdj_signal = indicators.get('kdj_analysis', {}).get('signal', '中性') if isinstance(indicators.get('kdj_analysis'), dict) else '中性'
            rsi_signal = indicators.get('rsi_analysis', {}).get('signal', '中性') if isinstance(indicators.get('rsi_analysis'), dict) else '中性'
            obv_signal = indicators.get('obv_analysis', {}).get('signal', '中性') if isinstance(indicators.get('obv_analysis'), dict) else '中性'
            macd_signal = indicators.get('macd_analysis', {}).get('signal', '中性') if isinstance(indicators.get('macd_analysis'), dict) else '中性'
        else:
            # 如果是原始指标数据，需要先分析
            kdj_data = indicators.get('KDJ', {})
            rsi_data = indicators.get('RSI', 50)
            obv_data = indicators.get('OBV', {})
            macd_data = indicators.get('MACD', {})
            
            # 分析各个指标
            kdj_analysis = self._analyze_kdj_basic(kdj_data if isinstance(kdj_data, dict) else {})
            rsi_analysis = self._analyze_rsi_basic(rsi_data if isinstance(rsi_data, (int, float)) else 50)
            obv_analysis = self._analyze_obv_basic(obv_data if isinstance(obv_data, dict) else {})
            macd_analysis = self._analyze_macd_basic(macd_data if isinstance(macd_data, dict) else {})
            
            kdj_signal = kdj_analysis.get('signal', '中性')
            rsi_signal = rsi_analysis.get('signal', '中性')
            obv_signal = obv_analysis.get('signal', '中性')
            macd_signal = macd_analysis.get('signal', '中性')
        
        # 统计看涨和看跌信号
        bullish = sum([1 for s in [kdj_signal, rsi_signal, macd_signal] 
                      if s in ['看涨', '超卖', '偏强']])
        bearish = sum([1 for s in [kdj_signal, rsi_signal, macd_signal] 
                      if s in ['看跌', '超买', '偏弱']])
        
        if bullish >= 2:
            return '看涨'
        elif bearish >= 2:
            return '看跌'
        else:
            return '中性'
    
    def _generate_trading_suggestion(self, indicators: Dict) -> str:
        """生成交易建议"""
        overall_signal = self._generate_overall_signal(indicators)
        
        if overall_signal == '看涨':
            return '技术指标综合显示看涨信号，可考虑买入或加仓'
        elif overall_signal == '看跌':
            return '技术指标综合显示看跌信号，可考虑卖出或减仓'
        else:
            return '技术指标信号不明确，建议观望'
    
    def _generate_llm_interpretation(self, 
                                     indicators: Dict,
                                     stock_code: str,
                                     current_price: float) -> Dict:
        """使用LLM生成解释"""
        if not self.llm_agent:
            return self._generate_basic_interpretation(indicators)
        
        # 构建提示词
        prompt = self._build_interpretation_prompt(indicators, stock_code, current_price)
        
        try:
            # 调用LLM（这里需要根据实际的LLM代理接口调整）
            # 假设llm_agent有generate方法
            if hasattr(self.llm_agent, 'generate'):
                response = self.llm_agent.generate(prompt)
            elif hasattr(self.llm_agent, 'chat'):
                # 如果支持chat接口
                response = self.llm_agent.chat(prompt)
            else:
                # 回退到基础解释
                return self._generate_basic_interpretation(indicators)
            
            # 解析LLM响应
            interpretation = self._parse_llm_response(response, indicators)
            return interpretation
        except Exception as e:
            print(f"⚠️  LLM调用失败: {e}")
            return self._generate_basic_interpretation(indicators)
    
    def _build_interpretation_prompt(self, 
                                     indicators: Dict,
                                     stock_code: str,
                                     current_price: float) -> str:
        """构建LLM提示词"""
        kdj = indicators.get('KDJ', {})
        rsi = indicators.get('RSI', 50)
        obv = indicators.get('OBV', {})
        macd = indicators.get('MACD', {})
        
        prompt = f"""请分析以下股票的技术指标，并提供专业的交易建议：

股票代码：{stock_code}
当前价格：{current_price:.2f}

技术指标数据：
1. KDJ指标：
   - K值：{kdj.get('K', 0):.2f}
   - D值：{kdj.get('D', 0):.2f}
   - J值：{kdj.get('J', 0):.2f}

2. RSI指标：{rsi:.2f}

3. OBV指标：
   - OBV比率：{obv.get('OBV_Ratio', 1.0):.2f}

4. MACD指标：
   - DIF：{macd.get('DIF', 0):.4f}
   - DEA：{macd.get('DEA', 0):.4f}
   - MACD柱：{macd.get('MACD', 0):.4f}

请提供：
1. 各指标的简要分析
2. 综合技术信号（看涨/看跌/中性）
3. 交易建议（买入/卖出/持有）
4. 风险提示

请用中文回答，简洁明了。"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, indicators: Dict) -> Dict:
        """解析LLM响应"""
        # 这里需要根据实际的LLM响应格式进行解析
        # 简化处理：如果响应是字符串，直接使用；否则尝试解析JSON
        
        try:
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                # 尝试解析JSON
                try:
                    return json.loads(response)
                except:
                    # 如果不是JSON，构建基础结构
                    return {
                        'summary': 'LLM技术指标分析',
                        'llm_response': response,
                        'kdj_analysis': self._analyze_kdj_basic(indicators.get('KDJ', {})),
                        'rsi_analysis': self._analyze_rsi_basic(indicators.get('RSI', 50)),
                        'obv_analysis': self._analyze_obv_basic(indicators.get('OBV', {})),
                        'macd_analysis': self._analyze_macd_basic(indicators.get('MACD', {})),
                        'overall_signal': self._generate_overall_signal(indicators),
                        'trading_suggestion': self._generate_trading_suggestion(indicators)
                    }
        except Exception as e:
            print(f"⚠️  解析LLM响应失败: {e}")
            return self._generate_basic_interpretation(indicators)
    
    def format_interpretation(self, interpretation: Dict) -> str:
        """格式化解释结果用于显示"""
        lines = []
        lines.append("   " + "=" * 64)
        lines.append("   🤖 技术指标智能解释")
        lines.append("   " + "=" * 64)
        
        # KDJ分析
        if 'kdj_analysis' in interpretation:
            kdj = interpretation['kdj_analysis']
            lines.append(f"   📊 KDJ指标:")
            if isinstance(kdj, dict):
                values = kdj.get('values', {})
                signal = kdj.get('signal', '未知')
                desc = kdj.get('description', '')
                lines.append(f"      K={values.get('K', 0):.2f}, D={values.get('D', 0):.2f}, J={values.get('J', 0):.2f}")
                lines.append(f"      信号: {signal}")
                lines.append(f"      说明: {desc}")
        
        # RSI分析
        if 'rsi_analysis' in interpretation:
            rsi = interpretation['rsi_analysis']
            lines.append(f"   📈 RSI指标:")
            if isinstance(rsi, dict):
                value = rsi.get('value', 50)
                signal = rsi.get('signal', '未知')
                desc = rsi.get('description', '')
                lines.append(f"      值: {value:.2f}")
                lines.append(f"      信号: {signal}")
                lines.append(f"      说明: {desc}")
        
        # OBV分析
        if 'obv_analysis' in interpretation:
            obv = interpretation['obv_analysis']
            lines.append(f"   💰 OBV指标:")
            if isinstance(obv, dict):
                ratio = obv.get('obv_ratio', 1.0)
                signal = obv.get('signal', '未知')
                desc = obv.get('description', '')
                lines.append(f"      比率: {ratio:.2f}")
                lines.append(f"      信号: {signal}")
                lines.append(f"      说明: {desc}")
        
        # MACD分析
        if 'macd_analysis' in interpretation:
            macd = interpretation['macd_analysis']
            lines.append(f"   📉 MACD指标:")
            if isinstance(macd, dict):
                values = macd.get('values', {})
                signal = macd.get('signal', '未知')
                desc = macd.get('description', '')
                lines.append(f"      DIF={values.get('DIF', 0):.4f}, DEA={values.get('DEA', 0):.4f}, MACD={values.get('MACD', 0):.4f}")
                lines.append(f"      信号: {signal}")
                lines.append(f"      说明: {desc}")
        
        # 综合信号
        if 'overall_signal' in interpretation:
            signal = interpretation['overall_signal']
            icon = "🟢" if signal == '看涨' else "🔴" if signal == '看跌' else "⚪"
            lines.append(f"   {icon} 综合信号: {signal}")
        
        # 交易建议
        if 'trading_suggestion' in interpretation:
            lines.append(f"   💡 交易建议: {interpretation['trading_suggestion']}")
        
        # LLM响应（如果有）
        if 'llm_response' in interpretation:
            lines.append("   " + "-" * 64)
            lines.append(f"   🤖 LLM详细分析:")
            lines.append(f"      {interpretation['llm_response']}")
        
        lines.append("   " + "=" * 64)
        
        return "\n".join(lines)

