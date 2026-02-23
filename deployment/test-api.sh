#!/bin/bash

# ==============================================================================
# Quick Backend API Test Script
# ==============================================================================
# Test all API endpoints to ensure they're working

set -e

DOMAIN="${1:-https://jpsearch.mehdinamdar.me}"
API_BASE="$DOMAIN/api"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Testing Talentin API at: $API_BASE"
echo "============================================"
echo ""

test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    
    echo -n "Testing $description... "
    
    local response_code
    if [ "$method" = "GET" ]; then
        response_code=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE$endpoint" 2>/dev/null || echo "000")
    elif [ "$method" = "POST" ]; then
        response_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$API_BASE$endpoint" 2>/dev/null || echo "000")
    fi
    
    if [ "$response_code" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC} (HTTP $response_code)"
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $response_code)"
    fi
}

# V2 Endpoints
echo "=== V2 API Endpoints ==="
test_endpoint "GET" "/v2/health" "Health check"
test_endpoint "GET" "/v2/stats" "Stats endpoint"
# Note: /v2/search requires proper JSON body, skipping for now

echo ""
echo "=== Legacy API Endpoints ==="
test_endpoint "GET" "/stats" "Legacy stats"
test_endpoint "GET" "/filters/skills" "Skills filter"
test_endpoint "GET" "/filters/countries" "Countries filter"
test_endpoint "GET" "/filters/cities" "Cities filter"
test_endpoint "GET" "/filters/locations" "Locations filter"
test_endpoint "GET" "/filters/industries" "Industries filter"
test_endpoint "GET" "/filters/roles" "Roles filter"
test_endpoint "GET" "/filters/schools" "Schools filter"
test_endpoint "GET" "/filters/certifications" "Certifications filter"

echo ""
echo "=== Analytics Endpoints ==="
test_endpoint "GET" "/analytics/countries" "Countries analytics"
test_endpoint "GET" "/analytics/industries" "Industries analytics"

echo ""
echo "============================================"
echo "API Testing Complete!"
echo ""

# Full test with actual data
echo "Fetching sample data..."
curl -s "$API_BASE/stats" | python3 -m json.tool 2>/dev/null || echo "Stats data retrieved (not JSON formatted)"
