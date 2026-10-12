@echo off
REM NeuroTrade screener watcher — long-running, auto-restart on crash.
REM Registered with Windows Task Scheduler (trigger: At log on).
REM Emits graded callouts + live TP1/TP2/STOP events to the feed.
REM See docs/paper_trading_setup.md.

cd /d "C:\Users\hassa\Downloads\NueroTrade"
:loop
".venv\Scripts\python.exe" -m scripts.screener_watch --interval 300 --universe all --top-n 12 --equity 100000 --min-grade C --always >> "logs\screener_watch.log" 2>&1
echo [%date% %time%] watcher exited (code %ERRORLEVEL%), restarting in 15s >> "logs\screener_watch.log"
timeout /t 15 /nobreak >nul
goto loop
