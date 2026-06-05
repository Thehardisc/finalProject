@echo off
setlocal EnableDelayedExpansion

echo.
echo ==============================================================
echo      [System]  Emotion Analysis System - Launcher
echo ==============================================================
echo.

REM ── Load config from .env ─────────────────────────────────────
set MAX_EMPATHETIC_SAMPLES_VAL=23000
set MIN_DB_SAMPLES_VAL=50
set RETRAIN_INTERVAL_VAL=1800
set ACCURACY_GATE_VAL=0.25
set LLM_PROVIDER_VAL=RULE_BASED
set LOG_LEVEL_VAL=INFO
set RATE_LIMIT_MAX_VAL=60
set API_KEY_VAL=N/A
set ANTHROPIC_KEY_VAL=

for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="MAX_EMPATHETIC_SAMPLES"  set MAX_EMPATHETIC_SAMPLES_VAL=%%b
    if "%%a"=="MIN_DB_SAMPLES"          set MIN_DB_SAMPLES_VAL=%%b
    if "%%a"=="RETRAIN_INTERVAL_SECONDS" set RETRAIN_INTERVAL_VAL=%%b
    if "%%a"=="ACCURACY_GATE"           set ACCURACY_GATE_VAL=%%b
    if "%%a"=="LLM_PROVIDER"            set LLM_PROVIDER_VAL=%%b
    if "%%a"=="LOG_LEVEL"               set LOG_LEVEL_VAL=%%b
    if "%%a"=="RATE_LIMIT_MAX"          set RATE_LIMIT_MAX_VAL=%%b
    if "%%a"=="INTERNAL_API_KEY"        set API_KEY_VAL=%%b
    if "%%a"=="ANTHROPIC_API_KEY"       set ANTHROPIC_KEY_VAL=%%b
)

REM Show whether ANTHROPIC_API_KEY is set (needed for synthetic data generation)
if "!ANTHROPIC_KEY_VAL!"=="" (
    set ANTHROPIC_STATUS=NOT SET ^(synthetic sentence generation will be skipped^)
) else (
    set ANTHROPIC_STATUS=SET ^(synthetic data generation enabled^)
)

echo [Config] Resolved environment from .env:
echo ----------------------------------------------------------
echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
echo    MAX_EMPATHETIC_SAMPLES  = !MAX_EMPATHETIC_SAMPLES_VAL!  ^(bootstrap dataset cap, runs once^)
echo    MIN_DB_SAMPLES          = !MIN_DB_SAMPLES_VAL!  ^(min DB rows to trigger continuous cycle^)
echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!  ^(min test accuracy to deploy model^)
echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min per user
echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
echo    ANTHROPIC_API_KEY       = !ANTHROPIC_STATUS!
echo ----------------------------------------------------------
echo.

REM ── Feature-vector parity check ───────────────────────────────
echo [Invariant] Checking feature-vector parity ^(inference vs trainer^)...
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -c "import pytest, numpy" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python -m pytest central_responder_service/training/test_feature_parity.py -q > "%TEMP%\parity_check.log" 2>&1
        if !ERRORLEVEL! equ 0 (
            echo    [OK] Feature-vector parity holds.
        ) else (
            echo    [FAIL] Feature-vector parity VIOLATION — inference and trainer disagree.
            echo           Launch aborted; fix build_feature_vector before starting.
            echo           Details:
            type "%TEMP%\parity_check.log"
            exit /b 1
        )
    ) else (
        echo    [SKIP] pytest/numpy not installed on host — parity not checked.
    )
) else (
    echo    [SKIP] python not on host — parity not checked.
)
echo.

REM ── Detect hardware ───────────────────────────────────────────
echo [Search] Detecting host environment...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo    [OK] NVIDIA GPU detected! Launching with GPU acceleration...
    set COMPOSE_CMD=docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
) else (
    echo    [INFO] No NVIDIA GPU detected. Launching in CPU mode...
    set COMPOSE_CMD=docker compose up -d --build
)
echo.

REM ── Setup log directory ───────────────────────────────────────
if not exist logs mkdir logs

set TIMESTAMP=%date:~-4,4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGDIR=logs\run_%TIMESTAMP%
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set STARTUP_LOG=%LOGDIR%\docker_startup.log
set INIT_LOG=%LOGDIR%\init.log
set LOGFILE=%LOGDIR%\all.log
set ERRORS_LOG=%LOGDIR%\errors.log
set IMPORTANT_LOG=%LOGDIR%\important.log

REM ── Launch stack, capture startup output ──────────────────────
echo [Launch] !COMPOSE_CMD!
echo [Launch] Capturing startup output to %STARTUP_LOG%
echo.
%COMPOSE_CMD% > "%STARTUP_LOG%" 2>&1
type "%STARTUP_LOG%"
if %ERRORLEVEL% neq 0 (
    echo    [Fail] docker compose exited with error. See %STARTUP_LOG%
    exit /b 1
)
echo.

REM ── Write init log header ─────────────────────────────────────
(
    echo ==============================================================
    echo    InnerLink Launcher - Init Log
    echo    Timestamp : %TIMESTAMP%
    echo ==============================================================
    echo.
    echo [Config] Resolved environment from .env:
    echo ----------------------------------------------------------
    echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
    echo    MAX_EMPATHETIC_SAMPLES  = !MAX_EMPATHETIC_SAMPLES_VAL!
    echo    MIN_DB_SAMPLES          = !MIN_DB_SAMPLES_VAL!
    echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
    echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!
    echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
    echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min
    echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
    echo    ANTHROPIC_API_KEY       = !ANTHROPIC_STATUS!
    echo ----------------------------------------------------------
    echo.
    echo [Hardware] Compose command: !COMPOSE_CMD!
    echo.
    echo ==============================================================
    echo [Docker Startup] build / create / start output
    echo ==============================================================
) > "%INIT_LOG%"
type "%STARTUP_LOG%" >> "%INIT_LOG%"

echo [Logs] Log directory  : %LOGDIR%
echo [Logs] Init log       : %INIT_LOG%
echo [Logs] All logs       : %LOGFILE%
echo [Logs] Errors only    : %ERRORS_LOG%
echo [Logs] Important      : %IMPORTANT_LOG%

REM ── Capture current time for --since (PowerShell UTC) ─────────
for /f %%a in ('powershell -NoProfile -Command "[DateTime]::UtcNow.ToString(\"yyyy-MM-ddTHH:mm:ssZ\")"') do set LAUNCH_TIME=%%a

REM ── Spawn per-service log files ───────────────────────────────
for /f %%s in ('docker compose config --services 2^>nul') do (
    start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% %%s > %LOGDIR%\%%s.log 2>&1"
)

REM ── Full combined log ─────────────────────────────────────────
start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% > %LOGFILE% 2>&1"

REM ── Errors-only log (findstr — no PowerShell required) ────────
start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% 2>&1 | findstr /i /c:"[ERROR" /c:"[CRITICAL" /c:"[WARNING" /c:"Traceback" /c:"Exception:" /c:"FATAL" /c:"CRASH" > %ERRORS_LOG%"

REM ── Important log (exclude noisy health pings) ────────────────
start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% 2>&1 | findstr /v /i /c:"GET /health" /c:"OPTIONS /health" /c:"GET / HTTP/1.1" > %IMPORTANT_LOG%"

REM ── Wait for containers ───────────────────────────────────────
echo.
echo [Wait] Waiting for all containers to start...

for /f %%a in ('docker compose config --services 2^>nul ^| find /c /v ""') do set EXPECTED_SERVICES=%%a

set MAX_WAIT=120
set ELAPSED=0

:WAIT_LOOP
if !ELAPSED! geq !MAX_WAIT! goto :TIMEOUT
for /f %%b in ('docker compose ps --services --status running 2^>nul ^| find /c /v ""') do set RUNNING=%%b
if !RUNNING! geq !EXPECTED_SERVICES! (
    echo    [OK] All !EXPECTED_SERVICES! containers are running!
    goto :HEALTH_CHECKS
)
echo    [Wait] !RUNNING! / !EXPECTED_SERVICES! containers running... ^(!ELAPSED!s^)
ping 127.0.0.1 -n 6 >nul
set /a ELAPSED+=5
goto :WAIT_LOOP

:TIMEOUT
echo    [Warn] Timeout! Not all containers started within %MAX_WAIT%s.
echo    Check logs: %LOGFILE%
docker compose ps
exit /b 1

REM ── Health checks ─────────────────────────────────────────────
:HEALTH_CHECKS
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

REM Check HTTP services
call :CheckHTTP "Ingestion Service (API :8000)" "http://localhost:8000/health"
call :CheckHTTP "API Service (WebSocket :8001)" "http://localhost:8001/conversation/conv-1/state"
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

REM ── Pipeline test ─────────────────────────────────────────────
echo.
echo [Test] Running end-to-end pipeline test...
for /f "tokens=1,2 delims==" %%A in (.env) do if "%%A"=="INTERNAL_API_KEY" set API_KEY=%%B

set API_TEST_URL=http://localhost:8000/messages
set PAYLOAD={\"conversation_id\": \"healthcheck\", \"user_id\": \"system\", \"text\": \"I am happy!\"}

for /f %%a in ('powershell -NoProfile -command "try { $r = Invoke-WebRequest -Uri \"%API_TEST_URL%\" -Method Post -ContentType \"application/json\" -Headers @{\"X-API-Key\"=\"%API_KEY%\"} -Body \"%PAYLOAD%\" -UseBasicParsing; Write-Output $r.StatusCode } catch { Write-Output $_.Exception.Response.StatusCode.value__ }" 2^>nul') do set HTTP_CODE=%%a

if "%HTTP_CODE%"=="200" (
    echo    [OK] Pipeline test - message accepted ^(HTTP 200^)
    set /a PASS+=1
) else (
    echo    [Fail] Pipeline test - failed ^(HTTP %HTTP_CODE%^)
    set /a FAIL+=1
)

REM ── Summary ───────────────────────────────────────────────────
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
echo    [Log]  Init report:   %INIT_LOG%          ^(config + health checks^)
echo    [Log]  Docker boot:   %STARTUP_LOG%       ^(build / create / start output^)
echo    [Log]  Errors only:   %ERRORS_LOG%        ^(ERROR/WARN/CRITICAL/Traceback^)
echo    [Log]  Important:     %IMPORTANT_LOG%     ^(no health pings^)
echo    [Log]  Full dump:     %LOGFILE%
echo    [Log]  Per-service:   %LOGDIR%\^<service^>.log
echo.
echo    Tip: type %ERRORS_LOG%
echo    Tip: type %IMPORTANT_LOG%
echo ==============================================================
echo.

REM Append health results to init.log
(
    echo.
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
    echo    %STARTUP_LOG%
    echo    %LOGDIR%\errors.log
    echo    %LOGDIR%\important.log
    echo    %LOGDIR%\all.log
) >> "%INIT_LOG%"

echo    [Log] Init report saved: %INIT_LOG%
echo.
pause
exit /b %FAIL%

REM ── Helper: HTTP check with retries ───────────────────────────
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
powershell -NoProfile -command "try { $r = Invoke-WebRequest -Uri '!URL!' -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo    [OK] !NAME! - responding
    set /a PASS+=1
    exit /b
)
ping 127.0.0.1 -n 3 >nul
set /a ATTEMPT+=1
goto :CheckHTTPLoop
