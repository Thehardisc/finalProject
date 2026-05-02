@echo off
setlocal EnableDelayedExpansion

echo.
echo ==============================================================
echo      [Brain]  Emotion Analysis System - Launcher
echo ==============================================================
echo.

REM Load config values from .env for display
set MAX_SAMPLES_VAL=2500
set RETRAIN_INTERVAL_VAL=1800
set ACCURACY_GATE_VAL=0.40
set LLM_PROVIDER_VAL=RULE_BASED
set LOG_LEVEL_VAL=INFO
set RATE_LIMIT_MAX_VAL=60
set API_KEY_VAL=N/A

for /f "tokens=1,* delims==" %%a in (.env) do (
    if "%%a"=="MAX_SAMPLES"              set MAX_SAMPLES_VAL=%%b
    if "%%a"=="RETRAIN_INTERVAL_SECONDS" set RETRAIN_INTERVAL_VAL=%%b
    if "%%a"=="ACCURACY_GATE"            set ACCURACY_GATE_VAL=%%b
    if "%%a"=="LLM_PROVIDER"             set LLM_PROVIDER_VAL=%%b
    if "%%a"=="LOG_LEVEL"                set LOG_LEVEL_VAL=%%b
    if "%%a"=="RATE_LIMIT_MAX"           set RATE_LIMIT_MAX_VAL=%%b
    if "%%a"=="INTERNAL_API_KEY"         set API_KEY_VAL=%%b
)

echo [Config] Resolved environment from .env:
echo ----------------------------------------------------------
echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
echo    MAX_SAMPLES             = !MAX_SAMPLES_VAL!  ^(GoEmotions dataset size per run^)
echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!  ^(min test accuracy to deploy model^)
echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min per user
echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
echo ----------------------------------------------------------
echo.

REM Detect Hardware
echo [Search] Detecting host environment...


where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] NVIDIA GPU Detected! Launching with GPU acceleration...
    set COMPOSE_CMD=docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
) else (
    echo    [INFO] No NVIDIA GPU detected. Launching in CPU mode...
    set COMPOSE_CMD=docker compose up -d --build
)

echo.
%COMPOSE_CMD%
echo.

REM Setup Logging — create timestamped run directory
if not exist logs mkdir logs

set TIMESTAMP=%date:~-4,4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGDIR=logs\run_%TIMESTAMP%
if not exist %LOGDIR% mkdir %LOGDIR%

set INIT_LOG=%LOGDIR%\init.log
set LOGFILE=%LOGDIR%\all.log

REM Write init log header
(
    echo ==============================================================
    echo    InnerLink Launcher - Init Log
    echo    Timestamp : %TIMESTAMP%
    echo ==============================================================
    echo.
    echo [Config] Resolved environment from .env:
    echo ----------------------------------------------------------
    echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
    echo    MAX_SAMPLES             = !MAX_SAMPLES_VAL!
    echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
    echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!
    echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
    echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min
    echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
    echo ----------------------------------------------------------
    echo.
) > %INIT_LOG%

echo [Logs] Log directory: %LOGDIR%
echo [Logs] Init log: %INIT_LOG%
start /b cmd /c "docker compose logs -f > %LOGFILE% 2>&1"


REM Wait for containers
echo.
echo [Wait] Waiting for all containers to start...

REM Count expected services (naive count of lines from config)
for /f %%a in ('docker compose config --services ^| find /c /v ""') do set EXPECTED_SERVICES=%%a

set MAX_WAIT=120
set ELAPSED=0

:WAIT_LOOP
if !ELAPSED! geq !MAX_WAIT! goto :TIMEOUT

REM Count running services
for /f %%b in ('docker compose ps --services --status running 2^>nul ^| find /c /v ""') do set RUNNING=%%b

if !RUNNING! geq !EXPECTED_SERVICES! (
    echo    [OK] All !EXPECTED_SERVICES! containers are running!
    goto :HEALTH_CHECKS
)

echo    [Wait] !RUNNING! / !EXPECTED_SERVICES! containers running... (!ELAPSED!s)
ping 127.0.0.1 -n 6 >nul
set /a ELAPSED+=5
goto :WAIT_LOOP

:TIMEOUT
echo    [Warn] Timeout! Not all containers started within %MAX_WAIT%s.
echo    Check logs: %LOGFILE%
docker compose ps
exit /b 1

:HEALTH_CHECKS
REM Health Checks
echo.
echo [Health] Running service health checks...
set PASS=0
set FAIL=0

REM Check Redis
docker compose exec -T redis redis-cli ping 2>nul | find "PONG" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] Redis - PONG
    set /a PASS+=1
) else (
    echo    [Fail] Redis - not responding
    set /a FAIL+=1
)

REM Check PostgreSQL
docker compose exec -T db pg_isready -U user -d emotion_db 2>nul | find "accepting" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] PostgreSQL - accepting connections
    set /a PASS+=1
) else (
    echo    [Fail] PostgreSQL - not ready
    set /a FAIL+=1
)

REM Check HTTP services (using PowerShell for curl equivalent)
call :CheckHTTP "Ingestion Service (API :8000)" "http://localhost:8000/health"
call :CheckHTTP "API Service (WebSocket :8001)" "http://localhost:8001/health/status"
call :CheckHTTP "Frontend (UI :5173)" "http://localhost:5173"

REM Check Meta-Learner loaded
docker compose logs central_responder_service 2>nul | find "Running in META-LEARNER mode" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] Meta-Learner - loaded and active
    set /a PASS+=1
) else (
    echo    [Fail] Meta-Learner - not loaded ^(check central_responder_service logs^)
    set /a FAIL+=1
)

REM Test Pipeline
echo.
echo [Test] Running end-to-end pipeline test...

REM Extract API key from .env
for /f "tokens=1,2 delims==" %%A in (.env) do if "%%A"=="INTERNAL_API_KEY" set API_KEY=%%B

set API_TEST_URL=http://localhost:8000/messages
set CONTENT_TYPE=application/json
set PAYLOAD={\"conversation_id\": \"healthcheck\", \"user_id\": \"system\", \"text\": \"I am happy!\"}

for /f %%a in ('powershell -command "try { $response = Invoke-WebRequest -Uri '%API_TEST_URL%' -Method Post -ContentType '%CONTENT_TYPE%' -Headers @{ 'X-API-Key' = '%API_KEY%' } -Body '%PAYLOAD%' -UseBasicParsing; Write-Output $response.StatusCode } catch { Write-Output $_.Exception.Response.StatusCode.value__ }" 2^>nul') do set HTTP_CODE=%%a

if "%HTTP_CODE%"=="200" (
    echo    [OK] Pipeline test - message accepted ^(HTTP 200^)
    set /a PASS+=1
) else (
    echo    [Fail] Pipeline test - failed ^(HTTP %HTTP_CODE%^)
    set /a FAIL+=1
)

REM Print summary
echo.
echo ==============================================================
if !FAIL! equ 0 (
    echo    [Yay]  ALL CHECKS PASSED ^(%PASS%/%PASS%^)
) else (
    echo    [Warn]   %PASS% passed, %FAIL% failed
)
echo ==============================================================
echo.
echo    [Web]  Frontend:      http://localhost:5173
echo    [API]  API:           http://localhost:8001
echo    [In]   Ingestion:     http://localhost:8000
echo.
echo    [Logs] Directory:     %LOGDIR%
echo    [Log]  Init report:   %INIT_LOG%
echo    [Log]  Full dump:     %LOGFILE%
echo.
echo ==============================================================
echo.

REM Append health check results to init.log
(
    echo [Health Checks]
    echo    PASS : %PASS%
    echo    FAIL : %FAIL%
    if %FAIL% equ 0 (
        echo    Result : ALL CHECKS PASSED
    ) else (
        echo    Result : %FAIL% check^(s^) FAILED - see all.log for details
    )
    echo.
    echo [URLs]
    echo    Frontend  : http://localhost:5173
    echo    API       : http://localhost:8001
    echo    Ingestion : http://localhost:8000
    echo.
    echo [Log Files]
    echo    %LOGDIR%\all.log
    echo    %LOGDIR%\init.log
) >> %INIT_LOG%

echo    [Log] Init report saved: %INIT_LOG%
echo.

pause
exit /b %FAIL%

REM Helper Function
:CheckHTTP
set NAME=%~1
set URL=%~2
set MAX_RETRIES=10
set ATTEMPT=0

:CheckHTTPLoop
if !ATTEMPT! geq !MAX_RETRIES! (
    echo    [Fail] !NAME! - NOT responding at !URL!
    set /a FAIL+=1
    exit /b
)

powershell -command "try { $response = Invoke-WebRequest -Uri '!URL!' -UseBasicParsing; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo    [OK] !NAME! - responding
    set /a PASS+=1
    exit /b
)

ping 127.0.0.1 -n 3 >nul
set /a ATTEMPT+=1
goto :CheckHTTPLoop
