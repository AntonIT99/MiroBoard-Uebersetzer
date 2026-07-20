@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Die virtuelle Umgebung fehlt. Starte zuerst setup.bat.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" export_miro_board.py
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" echo Der Export wurde mit Fehlercode %CODE% beendet.
pause
exit /b %CODE%
