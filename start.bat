@echo off
setlocal EnableDelayedExpansion

echo.
echo ==============================================================
echo      [Brain]  Emotion Analysis System - Launcher
echo ==============================================================
echo.

REM ── STEP 1: Detect Hardware ──────────────────────────────────────────────────
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

REM ── STEP 2: Logging ──────────────────────────────────────────────────────────
if not exist logs mkdir logs

set TIMESTAMP=%date:~-4,4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGFILE=logs\run_%TIMESTAMP%.log

echo [Logs] Log file: %LOGFILE%
start /b cmd /c "docker compose logs -f > %LOGFILE% 2>&1"

REM ── STEP 3: Wait for all containers to be running ────────────────────────────
echo.
echo [Wait] Waiting for all containers to start...

REM Count expected services (naive count of lines from config)
for /f %%a in ('docker compose config --services ^| find /c /v ""') do set EXPECTED_SERVICES=%%a

set MAX_WAIT=120
set ELAPSED=0

:WAIT_LOOP
if !ELAPSED! geq !MAX_WAIT! goto :TIMEOUT

REM Count running services
for /f %%b in ('docker compose ps --status running --format json 2^>nul ^| find /c ""State"":""running""') do set RUNNING=%%b
if "!RUNNING!"=="" (
    for /f %%b in ('docker compose ps --status running 2^>nul ^| find /c /v ""') do set /a RUNNING=%%b - 1
)

if !RUNNING! geq !EXPECTED_SERVICES! (
    echo    [OK] All !EXPECTED_SERVICES! containers are running!
    goto :HEALTH_CHECKS
)

echo    [Wait] !RUNNING! / !EXPECTED_SERVICES! containers running... (!ELAPSED!s)
timeout /t 5 /nobreak >nul
set /a ELAPSED+=5
goto :WAIT_LOOP

:TIMEOUT
echo    [Warn] Timeout! Not all containers started within %MAX_WAIT%s.
echo    Check logs: %LOGFILE%
docker compose ps
exit /b 1

:HEALTH_CHECKS
REM ── STEP 4: Service Health Checks ────────────────────────────────────────────
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
call :CheckHTTP "API Service (WebSocket :8001)" "http://localhost:8001/conversation/conv-1/state"
call :CheckHTTP "Frontend (UI :5173)" "http://localhost:5173"

REM Check Meta-Learner loaded
docker compose logs central_responder_service 2>nul | find "Running in META-LEARNER mode" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] Meta-Learner - loaded and active
    set /a PASS+=1
) else (
    echo    [Fail] Meta-Learner - not loaded (check central_responder_service logs)
    set /a FAIL+=1
)

REM ── STEP 5: End-to-End Pipeline Test ─────────────────────────────────────────
echo.
echo [Test] Running end-to-end pipeline test...

set API_TEST_URL=http://localhost:8000/messages
set CONTENT_TYPE=application/json
set PAYLOAD={\"conversation_id\": \"healthcheck\", \"user_id\": \"system\", \"text\": \"I am happy!\"}

for /f %%a in ('powershell -command "try { $response = Invoke-WebRequest -Uri '%API_TEST_URL%' -Method Post -ContentType '%CONTENT_TYPE%' -Body '%PAYLOAD%' -UseBasicParsing; Write-Output $response.StatusCode } catch { Write-Output $_.Exception.Response.StatusCode.value__ }" 2^>nul') do set HTTP_CODE=%%a

if "%HTTP_CODE%"=="200" (
    echo    [OK] Pipeline test - message accepted (HTTP 200)
    set /a PASS+=1
) else (
    echo    [Fail] Pipeline test - failed (HTTP %HTTP_CODE%)
    set /a FAIL+=1
)

REM ── STEP 6: Summary ─────────────────────────────────────────────────────────
echo.
echo ==============================================================
if %FAIL% equ 0 (
    echo    [Yay]  ALL CHECKS PASSED ^(%PASS%/%PASS%^)
) else (
    echo    [Warn]   %PASS% passed, %FAIL% failed
)
echo ==============================================================
echo.
echo    [Web]  Frontend:  http://localhost:5173
echo    [API]  API:       http://localhost:8001
echo    [In]   Ingestion: http://localhost:8000
echo    [Log]  Logs:      %LOGFILE%
echo.
echo ==============================================================
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
if %ATTEMPT% geq %MAX_RETRIES% (
    echo    [Fail] %NAME% - NOT responding at %URL%
    set /a FAIL+=1
    exit /b
)

powershell -command "try { $response = Invoke-WebRequest -Uri '%URL%' -UseBasicParsing; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo    [OK] %NAME% - responding
    set /a PASS+=1
    exit /b
)

timeout /t 2 /nobreak >nul
set /a ATTEMPT+=1
goto :CheckHTTPLoop
