@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo FEHLER: Der Python-Launcher "py" wurde nicht gefunden.
  echo Installiere Python 3.11 oder neuer und aktiviere "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Erstelle virtuelle Python-Umgebung ...
  py -3 -m venv .venv
  if errorlevel 1 goto :error
)

echo Installiere Abhaengigkeiten ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Einrichtung abgeschlossen.
echo Als Naechstes: 1_start_miro_chrome.bat starten.
pause
exit /b 0

:error
echo.
echo Die Einrichtung ist fehlgeschlagen.
pause
exit /b 1
