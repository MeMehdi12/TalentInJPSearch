#!/bin/bash
# ==============================================================================
# Server Setup Script - Run this on the AWS EC2 server
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================="
echo "  Talentin Server Setup"
echo "=================================="
echo ""

APP_DIR="/var/www/talentin"
UPLOAD_DIR="/home/ubuntu/talentin-upload"

# ==============================================================================
# 0. Copy Uploaded Files
# ==============================================================================
echo -e "${GREEN}[0/9]${NC} Copying uploaded files..."
if [ -d "$UPLOAD_DIR" ]; then
    # Copy backend files
    if [ -d "$UPLOAD_DIR/backend" ]; then
        mkdir -p $APP_DIR/backend
        cp -r $UPLOAD_DIR/backend/* $APP_DIR/backend/
        echo "✓ Backend files copied"
    fi
    
    # Copy frontend files
    if [ -d "$UPLOAD_DIR/frontend" ]; then
        mkdir -p $APP_DIR/frontend
        cp -r $UPLOAD_DIR/frontend/* $APP_DIR/frontend/
        echo "✓ Frontend files copied"
    fi
    
    # Copy deployment configs to home
    if [ -d "$UPLOAD_DIR/deployment" ]; then
        cp $UPLOAD_DIR/deployment/talentin-backend.service $HOME/ 2>/dev/null || true
        cp $UPLOAD_DIR/deployment/nginx-talentin.conf $HOME/ 2>/dev/null || true
        echo "✓ Deployment configs copied"
    fi
else
    echo -e "${YELLOW}⚠ Upload directory not found - assuming files already in place${NC}"
fi
echo ""

# ==============================================================================
# 1. Create Directory Structure
# ==============================================================================
echo -e "${GREEN}[1/9]${NC} Creating directory structure..."
sudo mkdir -p $APP_DIR/{backend,frontend,database,logs}
sudo chown -R $USER:$USER $APP_DIR
echo "✓ Directories created"
echo ""

# ==============================================================================
# 2. Install System Packages
# ==============================================================================
echo -e "${GREEN}[2/9]${NC} Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx
echo "✓ Packages installed"
echo ""

# ==============================================================================
# 3. Setup Python Virtual Environment
# ==============================================================================
echo -e "${GREEN}[3/9]${NC} Setting up Python virtual environment..."
cd $APP_DIR/backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
echo "✓ Virtual environment ready"
echo ""

# ==============================================================================
# 4. Install Python Dependencies
# ==============================================================================
echo -e "${GREEN}[4/9]${NC} Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo -e "${YELLOW}⚠ requirements.txt not found - skipping${NC}"
fi
echo ""

# ==============================================================================
# 5. Setup Environment File
# ==============================================================================
echo -e "${GREEN}[5/9]${NC} Checking environment file..."
if [ ! -f "$APP_DIR/backend/.env" ]; then
    echo -e "${YELLOW}⚠ Creating .env file - YOU MUST EDIT THIS!${NC}"
    if [ -f "$APP_DIR/backend/.env.template" ]; then
        cp $APP_DIR/backend/.env.template $APP_DIR/backend/.env
        echo "✓ .env file created from template - PLEASE EDIT IT!"
    else
        cat > $APP_DIR/backend/.env << 'EOF'
SEARCH_ENV=production
OPENAI_API_KEY=your-openai-key-here
QDRANT_API_KEY=your-qdrant-key-here
QDRANT_URL=your-qdrant-url-here
DATABASE_PATH=/var/www/talentin/database/talent_search.duckdb
LOG_LEVEL=INFO
EOF
        echo "✓ .env file created - PLEASE EDIT IT!"
    fi
else
    echo "✓ .env file exists"
fi
echo ""

# ==============================================================================
# 6. Install Backend Service
# ==============================================================================
echo -e "${GREEN}[6/8]${NC} Installing backend service..."
if [ -f "$HOME/talentin-backend.service" ]; then
    sudo cp $HOME/talentin-backend.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable talentin-backend
    echo "✓ Backend service installed"
else
    echo -e "${YELLOW}⚠ talentin-backend.service not found - skipping${NC}"
fi
echo ""

# ==============================================================================
# 7. Configure Nginx
# ==============================================================================
echo -e "${GREEN}[7/9]${NC} Configuring Nginx..."
if [ -f "$HOME/nginx-talentin.conf" ]; then
    sudo cp $HOME/nginx-talentin.conf /etc/nginx/sites-available/talentin
    sudo ln -sf /etc/nginx/sites-available/talentin /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && echo "✓ Nginx configuration valid"
else
    echo -e "${YELLOW}⚠ nginx-talentin.conf not found - skipping${NC}"
fi
echo ""

# ==============================================================================
# 8. Fix Permissions
# ==============================================================================
echo -e "${GREEN}[8/9]${NC} Setting permissions..."
sudo chown -R www-data:www-data $APP_DIR
sudo chmod -R 755 $APP_DIR
echo "✓ Permissions set"
echo ""

# ==============================================================================
# 9. Start Services
# ==============================================================================
echo -e "${GREEN}[9/9]${NC} Starting services..."
sudo systemctl start talentin-backend
sudo systemctl restart nginx
echo "✓ Services started"
echo ""

# ==============================================================================
# Status Check
# ==============================================================================
echo "=================================="
echo "  Status Check"
echo "=================================="
echo ""

echo "Backend Service:"
sudo systemctl status talentin-backend --no-pager | head -n 5
echo ""

echo "Nginx Service:"
sudo systemctl status nginx --no-pager | head -n 3
echo ""

# ==============================================================================
# Next Steps
# ==============================================================================
echo "=================================="
echo "  ✓ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Edit your .env file with real API keys:"
echo "     nano $APP_DIR/backend/.env"
echo ""
echo "  2. Restart backend after editing .env:"
echo "     sudo systemctl restart talentin-backend"
echo ""
echo "  3. Configure AWS Security Group to allow:"
echo "     - Port 80 (HTTP)"
echo "     - Port 443 (HTTPS)"
echo ""
echo "  4. Test your deployment:"
echo "     curl http://localhost:8001/api/v2/health"
echo ""
echo "  5. View logs:"
echo "     sudo journalctl -u talentin-backend -f"
echo ""
echo "Your app should be live at: http://$(curl -s ifconfig.me)"
echo ""
