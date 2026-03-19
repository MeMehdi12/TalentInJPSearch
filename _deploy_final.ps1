# Talentin JPSEARCH Deploy Script - Production Deploy
# Deploys backend changes to production server

Write-Host "🚀 Deploying Talentin JPSEARCH Fixes..." -ForegroundColor Green

# Step 1: Backend - Install dependencies & restart
cd backend
Write-Host "📦 Installing Python deps..." -ForegroundColor Yellow
pip install -r requirements.txt --upgrade

Write-Host "🔄 Restarting FastAPI server..." -ForegroundColor Yellow
# Kill existing uvicorn processes
Get-Process uvicorn* | Stop-Process -Force
Start-Sleep 2

# Start production server
Write-Host "⚡ Starting production server..." -ForegroundColor Green
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "search_api_v2:app --host 0.0.0.0 --port 8001 --workers 4 --reload=False"

cd ..

# Step 2: Run tests
Write-Host "🧪 Running integration tests..." -ForegroundColor Yellow
cd backend
python tests_integration.py
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed!"; exit 1 }
cd ..

# Step 3: Health check
Write-Host "✅ Health check..." -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "http://localhost:8001/api/v2/health" -TimeoutSec 10
Write-Host "Health: $($health.status)" -ForegroundColor Green

# Step 4: Verify skills fix
Write-Host "🔍 Testing skills filter fix..." -ForegroundColor Yellow
$testQuery = @{
    query = "react developer"
    selected_locations = @("Los Angeles")
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:8001/api/v2/smart-search" -Method Post -Body $testQuery -ContentType "application/json" -TimeoutSec 30
Write-Host "Top candidate: $($response.results[0].full_name)" -ForegroundColor Cyan
Write-Host "Score: $($response.results[0].score)" -ForegroundColor Cyan
if ($response.results[0].skills.Count -lt 2) {
    Write-Warning "⚠️ Top candidate has fewer than 2 skills - fix may not be working"
}

Write-Host "🎉 Deploy COMPLETE! Server running at http://localhost:8001" -ForegroundColor Green
Write-Host "Test endpoint: POST /api/v2/smart-search" -ForegroundColor Cyan
Write-Host '✅ Skills fix: No-skills profiles now penalized -0.35' -ForegroundColor Green
