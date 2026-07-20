@echo off
setlocal
cd /d "%~dp0"

set "CHROME="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"

if not defined CHROME (
  echo FEHLER: Google Chrome wurde nicht gefunden.
  echo Passe den Pfad in dieser BAT-Datei an oder installiere Chrome.
  pause
  exit /b 1
)

set "PROFILE=%~dp0.chrome-miro-profile"

echo Starte ein separates Chrome-Profil fuer den Miro-Export ...
echo Das ist absichtlich NICHT dein normales Chrome-Profil.
start "Miro Export Chrome" "%CHROME%" ^
  --remote-debugging-address=127.0.0.1 ^
  --remote-debugging-port=9222 ^
  --user-data-dir="%PROFILE%" ^
  --no-first-run ^
  --no-default-browser-check ^
  --disable-background-timer-throttling ^
  --disable-backgrounding-occluded-windows ^
  --disable-renderer-backgrounding ^
  "https://miro.com/app/dashboard/"

exit /b 0
