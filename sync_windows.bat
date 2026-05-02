@echo off
echo ==============================================================
echo    [Sync] Pulling latest changes from GitHub...
echo ==============================================================
echo.

:: Fetch the latest branches from the remote
git fetch origin

:: Switch to the addConvo branch and pull the latest changes
git checkout addConvo
git pull origin addConvo

echo.
echo ==============================================================
echo    [Docker] Rebuilding and starting the application...
echo ==============================================================
echo.

:: Build and start the docker containers in detached mode
docker compose up -d --build

echo.
echo ==============================================================
echo    [Done] The system is now syncing and starting up!
echo ==============================================================
pause
