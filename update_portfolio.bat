@echo off
REM 持仓状态更新工具 - Windows 批处理脚本
REM 使用方法: update_portfolio.bat --stock sh.600036 --shares 500 --balance 78375 --price 43.25

cd /d %~dp0
python update_portfolio.py %*

