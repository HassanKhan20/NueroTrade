@echo off
REM NeuroTrade CSM live runner — invoked daily by Windows Task Scheduler.
REM See docs/paper_trading_setup.md.

cd /d "C:\Users\hassa\Downloads\NueroTrade"
".venv\Scripts\python.exe" -m scripts.run_csm_live >> "logs\csm_live_task.log" 2>&1
exit /b %ERRORLEVEL%
