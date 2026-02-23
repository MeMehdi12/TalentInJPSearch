#!/bin/bash

# ==============================================================================
# Talentin Production Deployment Test Script
# ==============================================================================
# Run this on your VPS to verify everything is configured correctly

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "  Talentin Deployment Tests"
echo "=================================="
echo ""

# Configuration
DOMAIN="${1:-jpsearch.mehdinamdar.me}"
BACKEND_PORT=8001
APP_DIR="/var/www/talentin"

pass_count=0
fail_count=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((pass_count++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((fail_count++))
}

test_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# ==============================================================================
# TEST 1: Check Directory Structure
# ==============================================================================
echo "[1/12] Checking directory structure..."

if [ -d "$APP_DIR/backend" ]; then
    test_pass "Backend directory exists"
else
    test_fail "Backend directory missing at $APP_DIR/backend"
fi

if [ -d "$APP_DIR/frontend/dist" ]; then
    test_pass "Frontend dist directory exists"
else
    test_fail "Frontend dist directory missing at $APP_DIR/frontend/dist"
fi

if [ -d "$APP_DIR/database" ]; then
    test_pass "Database directory exists"
else
    test_fail "Database directory missing at $APP_DIR/database"
fi

# ==============================================================================
# TEST 2: Check Files
# ==============================================================================
echo ""
echo "[2/12] Checking required files..."

if [ -f "$APP_DIR/backend/.env" ]; then
    test_pass "Backend .env file exists"
else
    test_fail "Backend .env file missing"
fi

if [ -f "$APP_DIR/backend/search_api_v2.py" ]; then
    test_pass "Backend main file exists"
else
    test_fail "Backend main file missing"
fi

if [ -f "$APP_DIR/frontend/dist/index.html" ]; then
    test_pass "Frontend build exists"
else
    test_fail "Frontend index.html missing (did you run npm build?)"
fi

if [ -f "$APP_DIR/database/talent_search.duckdb" ]; then
    test_pass "DuckDB database file exists"
else
    test_fail "DuckDB database file missing"
fi

# ==============================================================================
# TEST 3: Check Python Virtual Environment
# ==============================================================================
echo ""
echo "[3/12] Checking Python virtual environment..."

if [ -d "$APP_DIR/venv" ]; then
    test_pass "Virtual environment exists"
    
    if [ -f "$APP_DIR/venv/bin/python" ]; then
        test_pass "Python executable in venv"
    else
        test_fail "Python executable missing in venv"
    fi
else
    test_fail "Virtual environment missing at $APP_DIR/venv"
fi

# ==============================================================================
# TEST 4: Check Python Dependencies
# ==============================================================================
echo ""
echo "[4/12] Checking Python dependencies..."

if [ -f "$APP_DIR/venv/bin/pip" ]; then
    # Check key packages
    $APP_DIR/venv/bin/pip show fastapi > /dev/null 2>&1 && test_pass "FastAPI installed" || test_fail "FastAPI not installed"
    $APP_DIR/venv/bin/pip show qdrant-client > /dev/null 2>&1 && test_pass "Qdrant client installed" || test_fail "Qdrant client not installed"
    $APP_DIR/venv/bin/pip show sentence-transformers > /dev/null 2>&1 && test_pass "Sentence transformers installed" || test_fail "Sentence transformers not installed"
    $APP_DIR/venv/bin/pip show duckdb > /dev/null 2>&1 && test_pass "DuckDB installed" || test_fail "DuckDB not installed"
else
    test_fail "Pip not found in venv"
fi

# ==============================================================================
# TEST 5: Check Systemd Service
# ==============================================================================
echo ""
echo "[5/12] Checking systemd service..."

if [ -f "/etc/systemd/system/talentin-backend.service" ]; then
    test_pass "Systemd service file exists"
    
    if systemctl is-enabled talentin-backend > /dev/null 2>&1; then
        test_pass "Service is enabled"
    else
        test_fail "Service is not enabled"
    fi
    
    if systemctl is-active talentin-backend > /dev/null 2>&1; then
        test_pass "Service is running"
    else
        test_fail "Service is not running (run: systemctl start talentin-backend)"
    fi
else
    test_fail "Systemd service file missing"
fi

# ==============================================================================
# TEST 6: Check Backend Port
# ==============================================================================
echo ""
echo "[6/12] Checking backend port..."

if netstat -tuln 2>/dev/null | grep -q ":$BACKEND_PORT " || ss -tuln 2>/dev/null | grep -q ":$BACKEND_PORT "; then
    test_pass "Backend listening on port $BACKEND_PORT"
else
    test_fail "Backend NOT listening on port $BACKEND_PORT"
fi

# ==============================================================================
# TEST 7: Test Backend API (Local)
# ==============================================================================
echo ""
echo "[7/12] Testing backend API (localhost)..."

health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$BACKEND_PORT/api/v2/health 2>/dev/null || echo "000")

if [ "$health_response" = "200" ]; then
    test_pass "Backend health check responds (HTTP 200)"
else
    test_fail "Backend health check failed (HTTP $health_response)"
fi

# Test legacy endpoint
stats_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$BACKEND_PORT/api/stats 2>/dev/null || echo "000")

if [ "$stats_response" = "200" ]; then
    test_pass "Backend /api/stats endpoint works (legacy)"
else
    test_fail "Backend /api/stats endpoint failed (HTTP $stats_response)"
fi

# ==============================================================================
# TEST 8: Check Nginx
# ==============================================================================
echo ""
echo "[8/12] Checking nginx..."

if systemctl is-active nginx > /dev/null 2>&1; then
    test_pass "Nginx is running"
else
    test_fail "Nginx is not running"
fi

if [ -f "/etc/nginx/sites-available/talentin" ]; then
    test_pass "Nginx config file exists"
else
    test_fail "Nginx config file missing"
fi

if [ -L "/etc/nginx/sites-enabled/talentin" ]; then
    test_pass "Nginx config is enabled"
else
    test_fail "Nginx config not enabled (run: ln -s /etc/nginx/sites-available/talentin /etc/nginx/sites-enabled/)"
fi

# Test nginx config
if nginx -t > /dev/null 2>&1; then
    test_pass "Nginx configuration valid"
else
    test_fail "Nginx configuration has errors"
fi

# ==============================================================================
# TEST 9: Check SSL Certificate
# ==============================================================================
echo ""
echo "[9/12] Checking SSL certificate..."

if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    test_pass "SSL certificate exists"
    
    # Check expiry
    expiry_date=$(openssl x509 -enddate -noout -in "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null | cut -d= -f2)
    if [ -n "$expiry_date" ]; then
        test_pass "SSL expires: $expiry_date"
    fi
else
    test_warn "SSL certificate not found (run: certbot --nginx -d $DOMAIN)"
fi

# ==============================================================================
# TEST 10: Test Public HTTPS Access
# ==============================================================================
echo ""
echo "[10/12] Testing public HTTPS access..."

https_health=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN/api/v2/health 2>/dev/null || echo "000")

if [ "$https_health" = "200" ]; then
    test_pass "Public API accessible via HTTPS"
else
    test_fail "Public API NOT accessible (HTTP $https_health)"
fi

# ==============================================================================
# TEST 11: Test Frontend
# ==============================================================================
echo ""
echo "[11/12] Testing frontend..."

frontend_response=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN/ 2>/dev/null || echo "000")

if [ "$frontend_response" = "200" ]; then
    test_pass "Frontend accessible"
else
    test_fail "Frontend NOT accessible (HTTP $frontend_response)"
fi

# ==============================================================================
# TEST 12: Check Permissions
# ==============================================================================
echo ""
echo "[12/12] Checking file permissions..."

backend_owner=$(stat -c '%U' "$APP_DIR/backend" 2>/dev/null || echo "unknown")
if [ "$backend_owner" = "www-data" ]; then
    test_pass "Backend directory owned by www-data"
else
    test_warn "Backend directory owned by $backend_owner (should be www-data)"
fi

db_owner=$(stat -c '%U' "$APP_DIR/database/talent_search.duckdb" 2>/dev/null || echo "unknown")
if [ "$db_owner" = "www-data" ]; then
    test_pass "Database file owned by www-data"
else
    test_warn "Database file owned by $db_owner (should be www-data)"
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=================================="
echo "  Test Summary"
echo "=================================="
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Your deployment is ready.${NC}"
    echo ""
    echo "Access your app at: https://$DOMAIN"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please review the errors above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Restart backend: systemctl restart talentin-backend"
    echo "  - Check logs: journalctl -u talentin-backend -n 50"
    echo "  - Restart nginx: systemctl restart nginx"
    echo "  - Fix permissions: chown -R www-data:www-data $APP_DIR"
    exit 1
fi
