@echo off
cd /d "%~dp0"
echo Checking for virtual environment...

if exist ".venv\Scripts\activate.bat" (
    echo Activating .venv...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Attempting to use global python...
)

echo Starting Streamlit Dashboard...
streamlit run dashboard.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to start Streamlit.
    echo Please ensure Streamlit is installed: pip install streamlit
    pause
    exit /b %errorlevel%
)

pause
