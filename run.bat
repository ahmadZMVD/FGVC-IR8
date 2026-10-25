@echo off
setlocal EnableDelayedExpansion
title FGVC-IR8 Inference Server
cd /d "%~dp0"

:: 1. Fast-Path: If environment is already verified, launch server immediately
if exist ".env_verified.json" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" launcher.py --fast
        goto :eof
    ) else (
        python launcher.py --fast
        goto :eof
    )
)

:: 2. First-Time Setup Path: Locate Python interpreter on Windows
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "SYS_PY=python"
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "SYS_PY=py -3"
    ) else (
        echo [ERROR] Python 3.10+ was not found in system PATH.
        echo Please install Python from https://www.python.org/downloads/ and check "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo [INIT] Running first-time environment verification...
%SYS_PY% launcher.py --setup
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Environment setup encountered an error.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launcher.py --fast
) else (
    %SYS_PY% launcher.py --fast
)
endlocal