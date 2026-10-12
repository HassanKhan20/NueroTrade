@echo off
REM NeuroTrade dashboard — long-running, auto-restart on crash.
REM Registered with Windows Task Scheduler (trigger: At log on).
REM See docs/paper_trading_setup.md.

cd /d "C:\Users\hassa\Downloads\NueroTrade"
:loop
".venv\Scripts\python.exe" -m scripts.dashboard --port 8765 --host 127.0.0.1 >> "logs\dashboard_local.log" 2>&1
echo [%date% %time%] dashboard exited (code %ERRORLEVEL%), restarting in 10s >> "logs\dashboard_local.log"
timeout /t 10 /nobreak >nul
goto loop
