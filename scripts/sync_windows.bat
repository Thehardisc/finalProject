@echo off
echo ==============================================================
echo    [Sync] Pulling latest changes from GitHub...
echo ==============================================================
echo.

git fetch origin

git checkout addConvo
git pull origin addConvo


