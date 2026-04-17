@echo off
REM start.bat - Universal AI Stack Launcher (Windows)
REM Automatically detects if your machine has an NVIDIA GPU and injects the hardware profiles.

echo Detecting host environment...

where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] NVIDIA GPU Detected! Launching with GPU acceleration...
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
) else (
    echo [INFO] No NVIDIA GPU detected.
    echo        Launching fully in CPU mode...
    docker compose up -d
)

REM ── LOGGING SYSTEM ─────────────────────────────────────────────────────────
if not exist logs mkdir logs

REM Generate a unique filename using the current timestamp (safe for Windows filenames)
set TIMESTAMP=%date:~-4,4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
REM Replace spaces with zeros for hours 0-9
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=logs\run_%TIMESTAMP%.log

echo [LOGS] Streaming all background processes into: %LOGFILE%
REM Execute docker compose logs in "follow" mode in the background
start /b cmd /c "docker compose logs -f > %LOGFILE% 2>&1"

pause
