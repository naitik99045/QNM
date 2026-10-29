@echo off
setlocal enabledelayedexpansion
title QNM Autonomous Agent - Dual-Bottleneck Engine
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo ===============================================================================
echo            QNM (Quantized-Concept Network Model) Autonomous Agent
echo       Dual-Bottleneck Sandwich Architecture with First-Class LM Studio
echo ===============================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.10+ and add it to your PATH.
    echo.
    pause
    exit /b 1
)

:: 2. Check and start LM Studio if needed
if exist "C:\Users\hp\.lmstudio\bin\lms.exe" (
    "C:\Users\hp\.lmstudio\bin\lms.exe" status | findstr /i "ON" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [*] Starting local LM Studio server...
        "C:\Users\hp\.lmstudio\bin\lms.exe" server start
        timeout /t 3 /nobreak >nul
    ) else (
        echo [OK] LM Studio server is active on port 1234.
    )
    "C:\Users\hp\.lmstudio\bin\lms.exe" ps | findstr /i "qwen2.5-coder-3b-instruct" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [*] Loading Qwen2.5-Coder model into LM Studio...
        "C:\Users\hp\.lmstudio\bin\lms.exe" load "qwen2.5-coder-3b-instruct" -y
    )
)

:: 3. If command-line arguments are provided, forward them directly to QNM CLI
if not "%~1"=="" (
    python -m qnm.cli.main %*
    exit /b %ERRORLEVEL%
)

:: 4. Interactive Menu
:MENU
cls
echo ===============================================================================
echo            QNM (Quantized-Concept Network Model) Autonomous Agent
echo ===============================================================================
echo.
echo Select an operation:
echo.
echo   [1] Interactive Agent Chat REPL (multi-turn memory + telemetry)
echo   [2] Run Code Specialist Agent (codebase editing, tests, debugging)
echo   [3] Run Research Specialist Agent (large doc compression & web search)
echo   [4] Run Shell & DevOps Agent (system diagnostics & automation)
echo   [5] Ask a Single Prompt (with live token compression breakdown)
echo   [6] Start Persistent Daemon & OpenAI Proxy (http://127.0.0.1:8000)
echo   [7] View Lifetime Efficiency & Token Compression Scoreboard
echo   [8] Check LM Studio Server Status & Loaded Models
echo   [9] Run Full Automated Test Suite (24 tests)
echo   [A] Catalog of Available Agents (built-in + YAML presets)
echo   [M] Switch / Load Active LM Studio Model
echo   [0] Exit
echo.
set /p choice="Enter choice [0-9, A, M]: "

if /i "%choice%"=="1" goto CHAT
if /i "%choice%"=="2" goto CODE_AGENT
if /i "%choice%"=="3" goto RESEARCH_AGENT
if /i "%choice%"=="4" goto SHELL_AGENT
if /i "%choice%"=="5" goto ASK_PROMPT
if /i "%choice%"=="6" goto SERVE_DAEMON
if /i "%choice%"=="7" goto STATS
if /i "%choice%"=="8" goto LMS_STATUS
if /i "%choice%"=="9" goto RUN_TESTS
if /i "%choice%"=="A" goto LIST_AGENTS
if /i "%choice%"=="M" goto SWITCH_MODEL
if /i "%choice%"=="0" goto EXIT
goto MENU

:CHAT
cls
python -m qnm.cli.main chat
echo.
pause
goto MENU

:CODE_AGENT
cls
echo [Code Specialist Agent]
set /p code_task="Enter your coding task: "
if not "%code_task%"=="" (
    python -m qnm.cli.main agent code "%code_task%"
)
echo.
pause
goto MENU

:RESEARCH_AGENT
cls
echo [Research Specialist Agent]
set /p research_task="Enter your research query or topic: "
if not "%research_task%"=="" (
    python -m qnm.cli.main agent research "%research_task%"
)
echo.
pause
goto MENU

:SHELL_AGENT
cls
echo [Shell & DevOps Specialist Agent]
set /p shell_task="Enter system diagnostic or shell task: "
if not "%shell_task%"=="" (
    python -m qnm.cli.main agent shell "%shell_task%"
)
echo.
pause
goto MENU

:ASK_PROMPT
cls
set /p user_prompt="Enter prompt: "
if not "%user_prompt%"=="" (
    python -m qnm.cli.main ask "%user_prompt%"
)
echo.
pause
goto MENU

:SERVE_DAEMON
cls
echo Starting Persistent Background Daemon on http://127.0.0.1:8000...
echo External IDEs (Cursor, VS Code, Windsurf) can connect to:
echo Base URL: http://127.0.0.1:8000/v1
echo.
python -m qnm.cli.main serve
echo.
pause
goto MENU

:STATS
cls
python -m qnm.cli.main stats
echo.
pause
goto MENU

:LMS_STATUS
cls
python -m qnm.cli.main lms-status
echo.
pause
goto MENU

:RUN_TESTS
cls
echo Running Full Automated Verification Suite...
python -m pytest tests/ -v
echo.
pause
goto MENU

:LIST_AGENTS
cls
python -m qnm.cli.main agents
echo.
pause
goto MENU

:SWITCH_MODEL
cls
echo ===============================================================================
echo                LM Studio Model Inspector & Switcher
echo ===============================================================================
echo.
if exist "C:\Users\hp\.lmstudio\bin\lms.exe" (
    echo Currently Loaded Model:
    "C:\Users\hp\.lmstudio\bin\lms.exe" ps
    echo.
    echo Downloaded Models on this PC:
    "C:\Users\hp\.lmstudio\bin\lms.exe" ls
    echo.
    set /p new_model="Enter model key to load (or press Enter to cancel): "
    if not "!new_model!"=="" (
        echo.
        echo [*] Loading model "!new_model!"...
        "C:\Users\hp\.lmstudio\bin\lms.exe" load "!new_model!" -y
    )
) else (
    echo [ERROR] lms.exe CLI not found. Please verify LM Studio installation.
)
echo.
pause
goto MENU

:EXIT
echo.
echo Thank you for using QNM Agent. Exiting...
timeout /t 2 /nobreak >nul
exit /b 0
