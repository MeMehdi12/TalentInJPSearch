# Backend-Only Hotfix Deploy (No DB Upload)
# Updates backend code on VPS without touching DuckDB/database

$KEY_PATH = "$env:USERPROFILE\.ssh\talentin-v2"
$SERVER = "ubuntu@ec2-3-12-221-73.us-east-2.compute.amazonaws.com"

Write-Host "🔥 Backend Hotfix Deploy (No DB)..." -ForegroundColor Green

# Backend source files (skills fix)
$backendFiles = @(
    "backend\config.py",
    "backend\search_api_v2.py", 
    "backend\search_schema.py",
    "backend\search_service.py",
    "backend\ranking_service.py"
)

foreach ($f in $backendFiles) {
    $local  = ".\$f"
    $remote = "/var/www/talentin/$f"
    if (Test-Path $local) {
        scp -i $KEY_PATH -o StrictHostKeyChecking=no $local "$SERVER`:$remote"
        Write-Host "Uploaded $f to VPS"
    } else {
        Write-Host "⚠️  $f not found locally"
    }
}

# Restart service
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "sudo systemctl restart talentin-backend; systemctl status talentin-backend"
Write-Host '🔄 Backend restarted. Live: https://jp.talentin.ai' -ForegroundColor Green

Write-Host '✅ Skills fix deployed (no-skills penalty -0.35)' -ForegroundColor Cyan
