@echo off
REM NeuroTrade Larry Williams breakout watcher — long-running, auto-restart.
REM Registered with Windows Task Scheduler (trigger: At log on).
REM Volume-confirmed volatility breakouts + win-streak position sizing.

cd /d "C:\Users\hassa\Downloads\NueroTrade"
:loop
".venv\Scripts\python.exe" -m scripts.williams_watch --interval 300 --universe all --top-n 10 --equity 100000 --base-risk-pct 0.01 --always >> "logs\williams_watch.log" 2>&1
echo [%date% %time%] williams watcher exited (code %ERRORLEVEL%), restarting in 15s >> "logs\williams_watch.log"
timeout /t 15 /nobreak >nul
goto loop
