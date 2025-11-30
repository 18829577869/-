# company_financial_data.py - 公司财务数据获取模块
# -*- coding: utf-8 -*-
"""
获取公司财务数据（年报、财务指标等）用于辅助决策
支持 Tushare、AkShare、baostock 等数据源
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd

# 尝试导入数据源
TUSHARE_AVAILABLE = False
AKSHARE_AVAILABLE = False
BAOSTOCK_AVAILABLE = False

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    pass


class CompanyFinancialData:
    """公司财务数据获取类"""
    
    def __init__(self, stock_code: str, data_source: str = None):
        """
        初始化
        
        Args:
            stock_code: 股票代码，如 'sh.600730' 或 '600730'
            data_source: 数据源，'tushare', 'akshare', 'baostock' 或 None（自动选择）
        """
        self.stock_code = stock_code
        self.data_source = data_source
        
        # 转换股票代码格式
        self.ts_code = self._convert_to_tushare_code(stock_code)
        self.ak_code = self._convert_to_akshare_code(stock_code)
        self.bs_code = self._convert_to_baostock_code(stock_code)
        
        # 初始化数据源
        self._init_data_source()
    
    def _convert_to_tushare_code(self, code: str) -> str:
        """转换为Tushare格式：600730.SH"""
        if '.' in code:
            code = code.split('.')[-1]
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"
        return code
    
    def _convert_to_akshare_code(self, code: str) -> str:
        """转换为AkShare格式：600730"""
        if '.' in code:
            return code.split('.')[-1]
        return code
    
    def _convert_to_baostock_code(self, code: str) -> str:
        """转换为baostock格式：sh.600730"""
        if '.' in code:
            return code
        if code.startswith('6'):
            return f"sh.{code}"
        elif code.startswith('0') or code.startswith('3'):
            return f"sz.{code}"
        return code
    
    def _init_data_source(self):
        """初始化数据源"""
        if self.data_source:
            if self.data_source == "tushare" and TUSHARE_AVAILABLE:
                self.data_source = "tushare"
            elif self.data_source == "akshare" and AKSHARE_AVAILABLE:
                self.data_source = "akshare"
            elif self.data_source == "baostock" and BAOSTOCK_AVAILABLE:
                self.data_source = "baostock"
                bs.login()
            else:
                self.data_source = None
        
        # 自动选择数据源
        if not self.data_source:
            if TUSHARE_AVAILABLE:
                try:
                    ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
                    self.pro = ts.pro_api()
                    self.data_source = "tushare"
                except:
                    pass
            
            if not self.data_source and AKSHARE_AVAILABLE:
                self.data_source = "akshare"
            
            if not self.data_source and BAOSTOCK_AVAILABLE:
                bs.login()
                self.data_source = "baostock"
    
    def get_financial_summary(self) -> Dict:
        """
        获取公司财务摘要信息
        
        Returns:
            包含财务指标的字典
        """
        summary = {
            "stock_code": self.stock_code,
            "data_source": self.data_source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "financial_indicators": {},
            "recent_announcements": [],
            "error": None
        }
        
        try:
            if self.data_source == "tushare" and TUSHARE_AVAILABLE:
                summary = self._get_tushare_financial(summary)
            elif self.data_source == "akshare" and AKSHARE_AVAILABLE:
                summary = self._get_akshare_financial(summary)
            elif self.data_source == "baostock" and BAOSTOCK_AVAILABLE:
                summary = self._get_baostock_financial(summary)
            else:
                summary["error"] = "无可用数据源"
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def _get_tushare_financial(self, summary: Dict) -> Dict:
        """从Tushare获取财务数据"""
        try:
            # 获取最新财务指标
            df = self.pro.fina_indicator(ts_code=self.ts_code, period="20241231", fields="ts_code,end_date,roe,roa,eps,netprofit_margin,current_ratio,quick_ratio,debt_to_assets")
            if not df.empty:
                latest = df.iloc[-1]
                summary["financial_indicators"] = {
                    "roe": float(latest.get("roe", 0)) if pd.notna(latest.get("roe")) else None,  # 净资产收益率
                    "roa": float(latest.get("roa", 0)) if pd.notna(latest.get("roa")) else None,  # 总资产收益率
                    "eps": float(latest.get("eps", 0)) if pd.notna(latest.get("eps")) else None,  # 每股收益
                    "netprofit_margin": float(latest.get("netprofit_margin", 0)) if pd.notna(latest.get("netprofit_margin")) else None,  # 净利润率
                    "current_ratio": float(latest.get("current_ratio", 0)) if pd.notna(latest.get("current_ratio")) else None,  # 流动比率
                    "quick_ratio": float(latest.get("quick_ratio", 0)) if pd.notna(latest.get("quick_ratio")) else None,  # 速动比率
                    "debt_to_assets": float(latest.get("debt_to_assets", 0)) if pd.notna(latest.get("debt_to_assets")) else None,  # 资产负债率
                    "period": latest.get("end_date", "")
                }
            
            # 获取最近公告
            try:
                ann_df = self.pro.ann(ts_code=self.ts_code, start_date=(datetime.now() - timedelta(days=90)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"))
                if not ann_df.empty:
                    summary["recent_announcements"] = ann_df[["ann_date", "title"]].head(5).to_dict("records")
            except:
                pass
        except Exception as e:
            summary["error"] = f"Tushare获取失败: {str(e)}"
        
        return summary
    
    def _get_akshare_financial(self, summary: Dict) -> Dict:
        """从AkShare获取财务数据"""
        try:
            # 获取财务指标
            try:
                df = ak.stock_financial_analysis_indicator(symbol=self.ak_code)
                if not df.empty:
                    latest = df.iloc[0]
                    summary["financial_indicators"] = {
                        "roe": float(latest.get("净资产收益率", 0)) if pd.notna(latest.get("净资产收益率")) else None,
                        "roa": float(latest.get("总资产报酬率", 0)) if pd.notna(latest.get("总资产报酬率")) else None,
                        "eps": float(latest.get("每股收益", 0)) if pd.notna(latest.get("每股收益")) else None,
                        "netprofit_margin": float(latest.get("销售净利率", 0)) if pd.notna(latest.get("销售净利率")) else None,
                        "current_ratio": float(latest.get("流动比率", 0)) if pd.notna(latest.get("流动比率")) else None,
                        "period": latest.get("报告期", "")
                    }
            except:
                pass
            
            # 获取公司公告
            try:
                ann_df = ak.stock_notice_report(symbol=self.ak_code)
                if not ann_df.empty:
                    summary["recent_announcements"] = ann_df[["公告日期", "公告标题"]].head(5).to_dict("records")
            except:
                pass
        except Exception as e:
            summary["error"] = f"AkShare获取失败: {str(e)}"
        
        return summary
    
    def _get_baostock_financial(self, summary: Dict) -> Dict:
        """从baostock获取财务数据（功能有限）"""
        try:
            # baostock主要提供K线数据，财务数据有限
            # 这里可以获取一些基本指标
            summary["financial_indicators"] = {
                "note": "baostock主要提供K线数据，财务指标需使用Tushare或AkShare"
            }
        except Exception as e:
            summary["error"] = f"baostock获取失败: {str(e)}"
        
        return summary
    
    def format_for_llm(self) -> str:
        """
        将财务数据格式化为LLM可理解的文本
        
        Returns:
            格式化的财务信息文本
        """
        summary = self.get_financial_summary()
        
        if summary.get("error"):
            return f"⚠️ 财务数据获取失败: {summary['error']}"
        
        text = f"【公司财务信息 - {self.stock_code}】\n"
        text += f"数据源: {summary.get('data_source', '未知')}\n"
        text += f"更新时间: {summary.get('timestamp', '未知')}\n\n"
        
        indicators = summary.get("financial_indicators", {})
        if indicators:
            text += "📊 财务指标:\n"
            if indicators.get("roe") is not None:
                text += f"  净资产收益率(ROE): {indicators['roe']:.2f}%\n"
            if indicators.get("roa") is not None:
                text += f"  总资产收益率(ROA): {indicators['roa']:.2f}%\n"
            if indicators.get("eps") is not None:
                text += f"  每股收益(EPS): {indicators['eps']:.2f} 元\n"
            if indicators.get("netprofit_margin") is not None:
                text += f"  净利润率: {indicators['netprofit_margin']:.2f}%\n"
            if indicators.get("current_ratio") is not None:
                text += f"  流动比率: {indicators['current_ratio']:.2f}\n"
            if indicators.get("quick_ratio") is not None:
                text += f"  速动比率: {indicators['quick_ratio']:.2f}\n"
            if indicators.get("debt_to_assets") is not None:
                text += f"  资产负债率: {indicators['debt_to_assets']:.2f}%\n"
            if indicators.get("period"):
                text += f"  报告期: {indicators['period']}\n"
        
        announcements = summary.get("recent_announcements", [])
        if announcements:
            text += "\n📢 最近公告:\n"
            for ann in announcements[:5]:
                if isinstance(ann, dict):
                    date = ann.get("ann_date") or ann.get("公告日期", "")
                    title = ann.get("title") or ann.get("公告标题", "")
                    text += f"  {date}: {title}\n"
        
        return text


def get_company_financial_info(stock_code: str, data_source: str = None) -> str:
    """
    便捷函数：获取公司财务信息并格式化为文本
    
    Args:
        stock_code: 股票代码
        data_source: 数据源（可选）
    
    Returns:
        格式化的财务信息文本
    """
    fetcher = CompanyFinancialData(stock_code, data_source)
    return fetcher.format_for_llm()

