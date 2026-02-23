#!/bin/bash

# ==============================================================================
# Frontend Build Script for Production
# ==============================================================================
# This script builds the React frontend for production deployment

set -e  # Exit on error

echo "================================"
echo "  Building Frontend for Production"
echo "================================"

# Navigate to frontend directory
cd "$(dirname "$0")/../frontend"

# Check if .env.production exists
if [ ! -f .env.production ]; then
    echo "ERROR: .env.production not found!"
    echo "Please create frontend/.env.production with your production settings"
    exit 1
fi

# Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
npm install

# Build for production
echo ""
echo "[2/3] Building React app..."
npm run build

# Verify build
echo ""
echo "[3/3] Verifying build..."
if [ -d "dist" ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Build output:"
    ls -lh dist/
    echo ""
    echo "Next steps:"
    echo "  1. Copy the 'dist' folder to /var/www/talentin/frontend/ on your VPS"
    echo "  2. Ensure nginx is configured to serve from /var/www/talentin/frontend/dist"
else
    echo "✗ Build failed - dist folder not created"
    exit 1
fi

echo ""
echo "================================"
echo "  Build Complete!"
echo "================================"
