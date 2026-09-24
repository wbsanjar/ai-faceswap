@echo off
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":5000 .*LISTENING"') do taskkill /F /PID %%p >nul 2>&1
echo FaceSwap Studio stopped.
pause