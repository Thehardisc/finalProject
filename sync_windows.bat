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


