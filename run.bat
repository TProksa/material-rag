@echo off
chcp 65001 >nul
echo.
echo  ==========================================
echo   Material RAG - Znalostni baze materialu
echo   s GPT-4o integraci
echo  ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [CHYBA] Python neni nainstalovan nebo neni v PATH.
    echo  Stahni Python z https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install dependencies if needed
echo  Kontroluji zavislosti...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo  Instaluji zavislosti (prvni spusteni - chvili potrva)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo  [CHYBA] Instalace selhala. Zkus spustit jako administrator.
        pause
        exit /b 1
    )
) else (
    :: Check if openai is installed
    pip show openai >nul 2>&1
    if errorlevel 1 (
        echo  Instaluji openai balicek...
        pip install openai==1.51.0
    )
)

echo.
echo  Spoustim server...
echo  Otevri prohlizec na: http://localhost:8000
echo.
echo  TIP: Pro AI odpovedi zadej OpenAI API klic
echo       v aplikaci (ikona ozubeného kola v panelu)
echo.
echo  Ukonceni: Ctrl+C
echo.

:: Open browser after short delay
start "" cmd /c "timeout /t 2 >nul && start http://localhost:8000"

:: Start server
python server.py

pause
