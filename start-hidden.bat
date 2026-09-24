@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" backend\server.py > "%~dp0server.log" 2>&1