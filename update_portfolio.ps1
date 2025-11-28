# 持仓状态更新工具 - PowerShell 脚本
# 使用方法: .\update_portfolio.ps1 --stock sh.600036 --shares 500 --balance 78375 --price 43.25

# 切换到脚本所在目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# 执行 Python 脚本
python update_portfolio.py $args

