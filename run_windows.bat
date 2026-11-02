@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.10 or 3.11 and enable Add to PATH.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.11 -m venv .venv 2>nul
  if errorlevel 1 py -3.10 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-prophet.txt
if errorlevel 1 (
  echo Prophet installation did not complete. The remaining application can still run; Prophet will be unavailable.
  echo Install Prophet later using: pip install -r requirements-prophet.txt
)
python backend\app.py
pause
