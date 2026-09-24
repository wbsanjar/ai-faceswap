@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setup needed. Run run.bat once first to install dependencies.
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:":5000 .*LISTENING" >nul
if %errorlevel%==0 (
    echo Server is already running on port 5000.
) else (
    wscript //B "%~dp0background.vbs"
    echo FaceSwap Studio started in the background.
)

echo.
echo Open http://localhost:5000 in your browser.
pause