@echo off
REM FaceSwap Studio - one-click launcher
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Installing dependencies (first run only - this can take a few minutes)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements-local.txt
    if errorlevel 1 (
        echo.
        echo Install failed. Make sure you have Python 3.10+ installed.
        pause
        exit /b 1
    )
)

echo Starting FaceSwap Studio at http://localhost:5000

REM Stop any old/stuck server still holding port 5000.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5000 .*LISTENING"') do taskkill /F /PID %%p >nul 2>&1

echo.
echo Tip: use start-bg.bat to run in the background (no console window),
echo      and stop.bat to stop it. Your uploaded photos and last result
echo      are remembered by the browser, so you never re-upload.
echo.

echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" backend\server.py
pause