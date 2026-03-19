@echo off
REM Deploy the backend and restart the server

REM Step 1: Navigate to backend directory
cd /d "%~dp0backend"

REM Step 2: (Optional) Pull latest changes from git
REM git pull

REM Step 3: (Optional) Install dependencies
REM pip install -r requirements.txt

REM Step 4: Restart the backend service (using Windows Service)
REM Replace 'talentin-backend' with your actual service name if different
sc stop talentin-backend
sc start talentin-backend

REM Step 5: (Optional) Print status
sc query talentin-backend

echo Backend deployment and restart complete.
pause
