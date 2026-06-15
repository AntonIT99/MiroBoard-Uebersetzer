# Miro Board Übersetzer

Dieses Tool erstellt einen englischen Klon eines bestehenden Miro-Boards und übersetzt unterstützte Textelemente automatisch mit DeepL.

## Funktionsweise

Das Script:

1. kopiert ein bestehendes Miro-Board,
2. liest alle unterstützten Board-Items aus der Kopie,
3. übersetzt gefundene Texte nach Englisch,
4. schreibt die übersetzten Texte zurück in den geklonten Board.

Das Original-Board bleibt unverändert.

## Unterstützte Elemente

Aktuell werden folgende Miro-Elemente übersetzt:

* Sticky Notes
* Textboxen
* Shapes mit Text
* Cards: Titel und Beschreibung
* Frames: Titel

Nicht zuverlässig unterstützt werden unter anderem:

* Bilder mit eingebettetem Text
* Screenshots
* PDFs
* Kommentare
* Kanban-Boards
* Mindmaps
* Tabellen
* spezielle Miro-Widgets

## Voraussetzungen

Empfohlene Python-Version:

```powershell
py -3.13 --version
```

Virtuelle Umgebung erstellen:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
```

Abhängigkeiten installieren:

```powershell
python -m pip install --upgrade pip
pip install requests python-dotenv
```

## Konfiguration

Im Projektordner eine `.env` Datei anlegen:

```env
MIRO_ACCESS_TOKEN=dein_miro_oauth_access_token
DEEPL_AUTH_KEY=dein_deepl_api_key
```

Der Miro-Token muss zu einem Miro-Nutzer gehören, der den geklonten Board bearbeiten
kann. Außerdem muss die Miro-App/API-Autorisierung Schreibrechte für Boards haben.
Wenn Miro beim Zurückschreiben `insufficientPermissions` meldet, den Klon mit genau
diesem Miro-Nutzer öffnen und prüfen, ob Elemente manuell bearbeitet werden können.
Falls nicht, App neu autorisieren bzw. den Board in einem Team klonen, in dem die App
installiert ist und der Nutzer Bearbeitungsrechte hat.

Das Script prüft diese Schreibrechte direkt nach dem Klonen mit einem temporären
Test-Shape. Erst wenn Erstellen, Aktualisieren und Löschen dieses Test-Elements
funktionieren, werden Board-Items gelesen und DeepL-Übersetzungen gestartet.

Optional kann mit `--target-team-id` gesteuert werden, in welchem Miro-Team der
geklonte Board erstellt wird. Der Token-Nutzer und die Miro-App müssen auch in
diesem Ziel-Team die nötigen Rechte haben. Ohne `--target-team-id` entscheidet
Miro anhand des Token-/Account-Kontexts, in welchem Team der Klon landet.

Für DeepL Free wird normalerweise automatisch die Free-API-URL verwendet, wenn der API-Key auf `:fx` endet.

Optional kann die DeepL-API-URL manuell gesetzt werden:

```env
DEEPL_API_URL=https://api-free.deepl.com/v2/translate
```

oder für DeepL Pro:

```env
DEEPL_API_URL=https://api.deepl.com/v2/translate
```

## Verwendung

### Board mit Board-ID übersetzen

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop"
```

### Board mit vollständiger Miro-URL übersetzen

```powershell
python translate_miro_board.py `
  --source-board "https://miro.com/app/board/uXjVDEINBOARDID=/" `
  --clone-name "[EN] Mein Workshop"
```

### Britisches Englisch verwenden

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-lang "EN-GB"
```

### US-Englisch verwenden

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-lang "EN-US"
```

### Klon in einem bestimmten Miro-Team erstellen

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-team-id "DEIN_MIRO_TEAM_ID"
```

### Sprache automatisch erkennen lassen

Standardmäßig ist Deutsch als Ausgangssprache gesetzt.

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --source-lang ""
```

### Testlauf ohne Miro-Updates

Der Klon wird erstellt und die Übersetzung wird vorbereitet, aber die übersetzten Texte werden nicht in Miro zurückgeschrieben.

```powershell
python translate_miro_board.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN TEST] Mein Workshop" `
  --dry-run
```

## Typischer Ablauf

```powershell
cd C:\Pfad\zum\Projekt

.\.venv\Scripts\activate

python translate_miro_board.py `
  --source-board "https://miro.com/app/board/uXjVDEINBOARDID=/" `
  --clone-name "[EN] Mein Workshop"
```

Nach erfolgreichem Lauf gibt das Script die ID und, falls von Miro geliefert, den Link zum englischen Klon aus.

## Hinweise

Dieses Tool erstellt bei jedem Lauf einen neuen englischen Klon. Es synchronisiert nicht inkrementell mit einem bereits bestehenden englischen Board.

Vorteile:

* kein kompliziertes Mapping zwischen Original- und Klon-Items nötig
* Original bleibt unverändert
* einfacher manueller Workflow
* weniger fehleranfällig

Nachteile:

* der Link zum englischen Board kann sich bei jedem Lauf ändern
* nicht alle Miro-Komponenten sind per API vollständig übersetzbar
