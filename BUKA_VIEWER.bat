@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.12 viewer\open_viewer.py %*
if errorlevel 1 pause
