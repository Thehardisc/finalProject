#!/bin/bash


echo "=============================================================="
echo "    [Sync] Pushing code to GitHub for Windows PC..."
echo "=============================================================="

MSG=${1:-"chore: Auto-sync update from Mac"}

echo ""
echo "-> Adding all changes..."
git add .

if git diff --cached --quiet; then
    echo "-> Nothing new to commit. Pushing existing commits..."
else
    echo "-> Committing changes with message: '$MSG'"
    git commit -m "$MSG"
fi

CURRENT_BRANCH=$(git branch --show-current)

echo "-> Pushing to origin/$CURRENT_BRANCH..."
git push origin "$CURRENT_BRANCH"

echo ""
echo "=============================================================="
if [ $? -eq 0 ]; then
    echo "    [Success] Code successfully pushed!"
    echo "    Now you can run your Windows sync script to download it."
else
    echo "    [Error] Failed to push code. Please check your internet connection."
fi
echo "=============================================================="
