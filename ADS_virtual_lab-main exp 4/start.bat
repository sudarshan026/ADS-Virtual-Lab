@echo off
REM ADS Virtual Lab - Startup Script for Windows
REM Start backend (Flask) and Streamlit UI in separate windows

echo.
echo ========================================
echo  ADS Virtual Lab - Starting...
echo ========================================
echo.

REM Check if running from correct directory
if not exist "api" (
    echo ERROR: Please run this from d:\DL\ADS_virtual_lab\
    pause
    exit /b 1
)

echo Starting Backend (Flask API on port 5000)...
echo.
start cmd /k "cd /d %~dp0api && .venv\Scripts\python.exe app.py"

timeout /t 3 /nobreak

echo.
echo Starting Frontend (Streamlit on port 8501)...
echo.
start cmd /k "cd /d %~dp0 && api\.venv\Scripts\python.exe -m streamlit run virtual-lab-ui\app.py --server.port 8501"

echo.
echo ========================================
echo Startup Complete!
echo ========================================
echo.
echo Expected:
echo   Backend: http://localhost:5000
echo   Frontend: http://localhost:8501
echo.
echo Open your browser to:
echo   http://localhost:8501
echo.
echo If Streamlit is missing, install once with:
echo   api\.venv\Scripts\python.exe -m pip install -r api\requirements.txt -r virtual-lab-ui\requirements.txt
echo.
pause
