# NutriCheck - Push Code to GitHub
# Target: https://github.com/charankumarb-netizen/nutra-main

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  NutriCheck - Push Code to GitHub" -ForegroundColor Cyan
Write-Host "  Target: https://github.com/charankumarb-netizen/nutra-main" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot

# 1. Check Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git was not found in your PATH! Please install Git from https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# 2. Init Git if not present
if (-not (Test-Path ".git")) {
    Write-Host "[1/5] Initializing git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# 3. Configure Remote
Write-Host "[2/5] Configuring remote origin..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin https://github.com/charankumarb-netizen/nutra-main.git

# 4. Stage and commit
Write-Host "[3/5] Staging files..." -ForegroundColor Yellow
git add .

Write-Host "[4/5] Committing changes..." -ForegroundColor Yellow
git commit -m "feat: complete NutriCheck dashboard, multilingual NLP pipeline and bug fixes"

# 5. Push
Write-Host "[5/5] Pushing to GitHub (main branch)..." -ForegroundColor Green
git push -u origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[INFO] If remote repository already contains initial commits (e.g. README/LICENSE), run:" -ForegroundColor Yellow
    Write-Host "git push -u origin main --force" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Successfully pushed to https://github.com/charankumarb-netizen/nutra-main" -ForegroundColor Green
}

