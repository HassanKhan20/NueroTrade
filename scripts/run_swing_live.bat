@echo off
REM NeuroTrade — multi-factor swing live runner. Invoked daily by Windows
REM Task Scheduler ("NeuroTrade Swing Rebalance"). See docs/paper_trading_setup.md.
cd /d "C:\Users\hassa\Downloads\NueroTrade"
".venv\Scripts\python.exe" -m scripts.run_swing_live >> "logs\swing_live_task.log" 2>&1
exit /b %ERRORLEVEL%
