# Miro Board Übersetzer

Dieses Tool erstellt einen englischen Klon eines bestehenden Miro-Boards und übersetzt unterstützte Textelemente lokal mit CTranslate2 und dem Hugging-Face-Modell `Helsinki-NLP/opus-mt-de-en`.

## Funktionsweise

Das Script:

1. kopiert ein bestehendes Miro-Board,
2. liest alle unterstützten Board-Items aus der Kopie,
3. prüft Schreibrechte auf der Kopie,
4. übersetzt gefundene Texte lokal nach Englisch,
5. schreibt die übersetzten Texte zurück in den geklonten Board.

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
pip install -r requirements.txt
```

`torch` wird für die einmalige Hugging-Face-nach-CTranslate2-Konvertierung
benötigt. Die spätere Übersetzung läuft über CTranslate2.

## Lokales Übersetzungsmodell vorbereiten

Das Script verwendet standardmäßig das Modell `Helsinki-NLP/opus-mt-de-en`, das einmalig heruntergeladen und nach CTranslate2 konvertiert werden muss:

```powershell
ct2-transformers-converter --model Helsinki-NLP/opus-mt-de-en --output_dir models/opus-mt-de-en-ct2 --quantization int8 --force
```

Unter Windows kann der explizite Aufruf aus der virtuellen Umgebung robuster sein:

```powershell
.\.venv\Scripts\ct2-transformers-converter.exe `
  --model Helsinki-NLP/opus-mt-de-en `
  --output_dir models/opus-mt-de-en-ct2 `
  --quantization int8 `
  --force
```

Danach liegt das lokale CTranslate2-Modell in:

```text
models/opus-mt-de-en-ct2
```

Nach dem Download und der Konvertierung läuft die Übersetzung lokal/offline und verursacht keine DeepL- oder Cloud-Übersetzungskosten.

## Konfiguration

Im Projektordner eine `.env` Datei anlegen:

```env
MIRO_ACCESS_TOKEN=dein_miro_oauth_access_token
```

Der Miro-Token muss zu einem Miro-Nutzer gehören, der den geklonten Board bearbeiten
kann. Außerdem muss die Miro-App/API-Autorisierung Schreibrechte für Boards haben.
Wenn Miro beim Zurückschreiben `insufficientPermissions` meldet, den Klon mit genau
diesem Miro-Nutzer öffnen und prüfen, ob Elemente manuell bearbeitet werden können.
Falls nicht, App neu autorisieren bzw. den Board in einem Team klonen, in dem die App
installiert ist und der Nutzer Bearbeitungsrechte hat.

Das Script prüft diese Schreibrechte nach dem Lesen der Board-Items und vor der
lokalen Übersetzung mit einem temporären Test-Shape. Erst wenn Erstellen,
Aktualisieren und Löschen dieses Test-Elements funktionieren, startet die
Übersetzung.

Optional kann mit `--target-team-id` gesteuert werden, in welchem Miro-Team der
geklonte Board erstellt wird. Der Token-Nutzer und die Miro-App müssen auch in
diesem Ziel-Team die nötigen Rechte haben. Ohne `--target-team-id` entscheidet
Miro anhand des Token-/Account-Kontexts, in welchem Team der Klon landet.

Übersetzungen werden in `translation_cache_ct2_de_en.json` zwischengespeichert.
Der Cache-Key enthält den Ausgangstext, das Tokenizer-Modell und den lokalen
CTranslate2-Modellpfad. Identische Texte werden dadurch nicht erneut übersetzt.

## Verwendung

### Board mit Board-ID übersetzen

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop"
```

### Board mit vollständiger Miro-URL übersetzen

```powershell
python main.py `
  --source-board "https://miro.com/app/board/uXjVDEINBOARDID=/" `
  --clone-name "[EN] Mein Workshop"
```

### Klon in einem bestimmten Miro-Team erstellen

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-team-id "DEIN_MIRO_TEAM_ID"
```

### Lokales Modell explizit angeben

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --translator "ct2" `
  --ct2-model-dir "models/opus-mt-de-en-ct2" `
  --hf-tokenizer-model "Helsinki-NLP/opus-mt-de-en" `
  --ct2-device "cpu" `
  --ct2-compute-type "int8" `
  --translation-batch-size 32
```

### NVIDIA-GPU verwenden

Für schnelle Übersetzung auf einer NVIDIA-GPU wie RTX 4080 Super oder RTX 5070 Ti
empfiehlt sich:

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-team-id "DEIN_MIRO_TEAM_ID" `
  --ct2-device "cuda" `
  --ct2-compute-type "int8_float16" `
  --translation-batch-size 64
```

Wenn CUDA/CTranslate2 auf dem System noch nicht korrekt eingerichtet ist oder
Speicherfehler auftreten, zuerst auf die CPU-Variante zurückgehen:

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Mein Workshop" `
  --target-team-id "DEIN_MIRO_TEAM_ID" `
  --ct2-device "cpu" `
  --ct2-compute-type "int8" `
  --translation-batch-size 32
```

### Testlauf ohne finale Miro-Updates

Der Klon wird erstellt, der Schreibrechte-Preflight läuft, und die Übersetzung
wird vorbereitet. Die übersetzten Texte werden aber nicht in Miro
zurückgeschrieben.

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN TEST] Mein Workshop" `
  --dry-run
```

## CLI-Argumente

| Argument | Default | Beschreibung |
| --- | --- | --- |
| `--source-board` | erforderlich | Miro-Board-ID oder vollständige Miro-Board-URL. |
| `--clone-name` | automatisch aus `--clone-prefix` | Name des geklonten Boards. |
| `--clone-prefix` | `[EN]` | Prefix für automatisch erzeugte Clone-Namen. |
| `--target-team-id` | leer | Optionales Miro-Ziel-Team für den geklonten Board. |
| `--translator` | `ct2` | Übersetzungsbackend. Aktuell ist nur `ct2` unterstützt. |
| `--ct2-model-dir` | `models/opus-mt-de-en-ct2` | Lokales konvertiertes CTranslate2-Modell. |
| `--hf-tokenizer-model` | `Helsinki-NLP/opus-mt-de-en` | Hugging-Face-Tokenizer-Modell oder lokaler Tokenizer-Pfad. |
| `--ct2-device` | `cpu` | CTranslate2-Gerät, z. B. `cpu` oder `cuda`. |
| `--ct2-compute-type` | `int8` | CTranslate2-Compute-Type, z. B. `int8`, `int8_float16`, `float32` oder `float16`. |
| `--translation-batch-size` | `32` | Anzahl Plain-Text-Einheiten pro lokaler Übersetzungsbatch. |
| `--sleep-after-copy` | `3.0` | Wartezeit nach dem Kopieren, bevor Items gelesen werden. |
| `--dry-run` | aus | Erstellt den Klon, führt den Schreibrechte-Preflight aus und übersetzt lokal, schreibt aber keine Texte zurück. |
| `--source-lang` | `DE` | Kompatibilitätsargument; das Standardmodell ist fest Deutsch nach Englisch. |
| `--target-lang` | `EN-US` | Kompatibilitätsargument; das Standardmodell ist fest Deutsch nach Englisch. |

## Typischer Ablauf

```powershell
cd C:\Pfad\zum\Projekt

.\.venv\Scripts\activate

python main.py `
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
