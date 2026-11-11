@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 dan Python Launcher diperlukan untuk setup pertama.
        goto :failed
    )
)
if not exist ".venv\.mtah-deps-v1" (
    ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
    if errorlevel 1 goto :failed
    type nul > ".venv\.mtah-deps-v1"
)
".venv\Scripts\python.exe" run.py %*
if errorlevel 1 goto :failed
if "%~1"=="" pause
exit /b 0

:failed
if "%~1"=="" pause
exit /b 1
