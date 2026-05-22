#!/bin/bash

# A simple script to quickly commit and push all changes to GitHub.

echo "=============================================================="
echo "    [Sync] Pushing code to GitHub for Windows PC..."
echo "=============================================================="

# Check if a commit message was provided, otherwise use a default one
MSG=${1:-"chore: Auto-sync update from Mac"}

echo ""
echo "-> Adding all changes..."
git add .

# Only commit if there are staged changes
if git diff --cached --quiet; then
    echo "-> Nothing new to commit. Pushing existing commits..."
else
    echo "-> Committing changes with message: '$MSG'"
    git commit -m "$MSG"
fi

# Get the name of the current branch
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
