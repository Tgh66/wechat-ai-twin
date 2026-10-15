@echo off
chcp 65001 >nul
title 微信 AI 数字分身工作台
cd /d %~dp0

:: 请求管理员权限（可访问受保护目录）
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 正在请求管理员权限...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

:: 使用项目内 venv（若有）否则系统 python
if exist venv\Scripts\python.exe (
    set PY=venv\Scripts\python.exe
) else (
    set PY=python
)

%PY% -c "import gradio, transformers, peft, wechatauto" >nul 2>&1
if %errorLevel% neq 0 (
    echo 首次运行，正在安装依赖...
    %PY% -m pip install -r requirements.txt
)

%PY% app.py
pause
