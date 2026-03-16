##############################################################################
# deploy_no_db.ps1 - Deployment to VPS WITHOUT DuckDB upload
#
# Run from repo root:
#   .\pipeline\deploy_no_db.ps1
#
# This script skips the DuckDB upload step.
##############################################################################

# -- CONFIGURE THESE ----------------------------------------------------------
$KEY_PATH     = "$env:USERPROFILE\.ssh\talentin-v2"
$SERVER       = "ubuntu@ec2-3-12-221-73.us-east-2.compute.amazonaws.com"
$VITE_API_KEY = ""   # paste your frontend API key here, or leave blank
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot   # repo root

function Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Ok([string]$msg)   { Write-Host "  OK: $msg"   -ForegroundColor Green }
function Fail([string]$msg) { Write-Host "  FAIL: $msg" -ForegroundColor Red ; exit 1 }

# -- 1. Verify key file -------------------------------------------------------
Step "Checking SSH key"
if (-not (Test-Path $KEY_PATH)) { Fail "PEM key not found at $KEY_PATH" }
Ok "Key found"

# -- 2. Write frontend .env for production build ------------------------------
Step "Writing frontend/.env"
$envLines = @(
    "VITE_API_URL=https://jp.talentin.ai",
    "VITE_API_KEY=$VITE_API_KEY",
    'VITE_CLIENT_MAP={"@talentin.ai":"00","@avistatech.net":"00","@loxo.co":"01"}',
    "VITE_DEFAULT_CLIENT_ID=00"
)
$envPath = Join-Path $ROOT "frontend\.env"
Set-Content -Path $envPath -Value $envLines -Encoding UTF8
Ok "frontend/.env written"

# -- 3. Build frontend --------------------------------------------------------
Step "Building frontend (npm run build)"
Push-Location (Join-Path $ROOT "frontend")
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { Fail "npm build failed (exit $LASTEXITCODE)" }
    Ok "Build complete - dist/"
} finally {
    Pop-Location
}

# -- 4. Upload backend source files -------------------------------------------
Step "Uploading backend source files"
$backendFiles = @(
    "backend\__init__.py",
    "backend\api_enhancements.py",
    "backend\client_auth.py",
    "backend\config.py",
    "backend\filter_service.py",
    "backend\hybrid_search.py",
    "backend\integration_api.py",
    "backend\normalizers.py",
    "backend\openai_config.py",
    "backend\openai_parser.py",
    "backend\query_preprocessor.py",
    "backend\ranking_service.py",
    "backend\requirements.txt",
    "backend\search_api_v2.py",
    "backend\search_schema.py",
    "backend\search_service.py",
    "backend\sparse_encoder.py",
    "backend\text_parser.py",
    "backend\mappings\company_aliases.json",
    "backend\mappings\school_aliases.json",
    "backend\prompts\jd_parser.txt",
    "backend\prompts\query_parser.txt"
)
# Ensure remote subdirectories exist
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "mkdir -p /var/www/talentin/backend/mappings /var/www/talentin/backend/prompts"
foreach ($f in $backendFiles) {
    $local  = Join-Path $ROOT $f
    $remote = "/var/www/talentin/$($f -replace '\\', '/')"
    if (-not (Test-Path $local)) {
        Write-Host "  SKIP (not found): $f" -ForegroundColor Yellow
        continue
    }
    scp -i $KEY_PATH -o StrictHostKeyChecking=no $local "${SERVER}:${remote}"
    if ($LASTEXITCODE -ne 0) { Fail "Upload failed for $f" }
    Ok $f
}

# -- 5. Ensure DEFAULT_CLIENT_ID=00 is in VPS .env ---------------------------
Step "Updating VPS .env (DEFAULT_CLIENT_ID)"
$envCheck = ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "grep -q '^DEFAULT_CLIENT_ID=' /var/www/talentin/backend/.env && echo found || echo missing"
if ($envCheck -match "found") {
    ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "sed -i 's/^DEFAULT_CLIENT_ID=.*/DEFAULT_CLIENT_ID=00/' /var/www/talentin/backend/.env"
    Ok "DEFAULT_CLIENT_ID updated"
} else {
    ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "echo 'DEFAULT_CLIENT_ID=00' >> /var/www/talentin/backend/.env"
    Ok "DEFAULT_CLIENT_ID added"
}

# -- 6. Push frontend dist/ ---------------------------------------------------
Step "Uploading frontend dist/"
$localDist  = Join-Path $ROOT "frontend\dist"
$remoteDist = "/var/www/talentin/frontend/dist/"
if (-not (Test-Path $localDist)) { Fail "dist/ not found - build must have failed" }
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "rm -rf $remoteDist"
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "mkdir -p $remoteDist"
scp -i $KEY_PATH -o StrictHostKeyChecking=no -r "${localDist}/." "${SERVER}:${remoteDist}"
if ($LASTEXITCODE -ne 0) { Fail "Frontend dist upload failed" }
Ok "Frontend dist uploaded"

# -- 7. Restart backend service -----------------------------------------------
Step "Restarting talentin-backend"
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "sudo systemctl restart talentin-backend"
if ($LASTEXITCODE -ne 0) { Fail "Service restart command failed" }
Start-Sleep -Seconds 3
$status = ssh -i $KEY_PATH -o StrictHostKeyChecking=no $SERVER "systemctl is-active talentin-backend"
if ($status -ne "active") { Fail "Service did not come up - status: $status" }
Ok "Service is active"

# -- Done ---------------------------------------------------------------------
Write-Host ""
Write-Host "Deployment complete!  https://jp.talentin.ai" -ForegroundColor Green
