@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   NutriCheck - Push Code to GitHub
echo   Target: https://github.com/charankumarb-netizen/nutra-main
echo =======================================================
echo.

cd /d "%~dp0"

echo [1/5] Checking Git installation...
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git is not found in PATH! Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

echo [2/5] Initializing Git repository if needed...
if not exist ".git" (
    git init
    git branch -M main
)

echo [3/5] Setting remote repository origin...
git remote remove origin 2>nul
git remote add origin https://github.com/charankumarb-netizen/nutra-main.git

echo [4/5] Staging and committing files...
git add .
git commit -m "feat: complete NutriCheck dashboard, multilingual NLP pipeline and bug fixes"

echo [5/5] Pushing to GitHub (main branch)...
git push -u origin main

if %ERRORLEVEL% neq 0 (
    echo.
    echo [NOTE] If remote branch contains conflicting commits, attempting push with set-upstream...
    echo You can also try: git push -u origin main --force
)

echo.
echo =======================================================
echo   Done!
echo =======================================================
pause

