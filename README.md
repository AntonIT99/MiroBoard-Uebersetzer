# Miro Board Übersetzer

Dieses Tool erstellt einen englischen Klon eines bestehenden Miro-Boards oder aktualisiert einen bestehenden englischen Klon. Unterstützte Textelemente werden lokal mit CTranslate2 und dem Hugging-Face-Modell `facebook/nllb-200-distilled-1.3B` übersetzt.

## Funktionsweise

Das Script:

1. prüft das lokale CTranslate2-Backend inklusive CUDA, falls `--ct2-device cuda` gesetzt ist,
2. kopiert ein bestehendes Miro-Board,
3. liest alle unterstützten Board-Items aus der Kopie,
4. prüft Schreibrechte auf der Kopie,
5. übersetzt gefundene Texte lokal nach Englisch,
6. schreibt die übersetzten Texte zurück in den geklonten Board.

Das Original-Board bleibt unverändert.

Im Standardmodus wird ein neuer Klon erstellt. Mit `--update-existing-clone`
kann ein bestehender englischer Klon anhand einer Sync-State-Datei aktualisiert
werden.

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

Das Script verwendet standardmäßig `facebook/nllb-200-distilled-1.3B`. Dieses
NLLB-Modell ist größer und qualitativ stärker als das bisherige
`Helsinki-NLP/opus-mt-de-en`, läuft aber nach der Konvertierung weiterhin lokal
über CTranslate2.

```powershell
ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B --output_dir models/nllb-200-distilled-1.3B-ct2 --quantization int8 --force
```

Unter Windows kann der explizite Aufruf aus der virtuellen Umgebung robuster sein:

```powershell
.\.venv\Scripts\ct2-transformers-converter.exe `
  --model facebook/nllb-200-distilled-1.3B `
  --output_dir models/nllb-200-distilled-1.3B-ct2 `
  --quantization int8 `
  --force
```

Danach liegt das lokale CTranslate2-Modell in:

```text
models/nllb-200-distilled-1.3B-ct2
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

Das lokale Übersetzungsbackend wird noch früher geprüft: Vor dem Miro-Clone lädt
das Script das CTranslate2-Modell und führt eine Mini-Übersetzung aus. Bei
fehlendem Modell, fehlenden Python-Paketen oder fehlenden CUDA-DLLs bricht der
Lauf dadurch ab, ohne einen neuen Miro-Clone zu erzeugen.

Optional kann mit `--target-team-id` gesteuert werden, in welchem Miro-Team der
geklonte Board erstellt wird. Der Token-Nutzer und die Miro-App müssen auch in
diesem Ziel-Team die nötigen Rechte haben. Ohne `--target-team-id` entscheidet
Miro anhand des Token-/Account-Kontexts, in welchem Team der Klon landet.

Übersetzungen werden in `translation_cache_ct2_de_en.json` zwischengespeichert.
Der Cache-Key enthält den Ausgangstext, das Tokenizer-Modell, den lokalen
CTranslate2-Modellpfad, Sprachcodes, Beam-Size und Symbolschutz-Einstellung.
Identische Texte werden dadurch nicht erneut übersetzt, aber neue Modell- oder
Qualitätseinstellungen erzeugen sauber getrennte Cache-Einträge.

Emojis, Piktogramme, Dingbats und ähnliche Symbole werden standardmäßig vor der
Übersetzung geschützt und danach unverändert wieder eingesetzt. Das reduziert
verlorene oder falsch interpretierte Icons in Miro-Texten. Bei Bedarf kann dieses
Verhalten mit `--no-preserve-special-symbols` deaktiviert werden.

## Glossar

Domain-spezifische Begriffe können in `translation_glossary_de_en.json`
festgelegt werden. Das Glossar wird standardmäßig verwendet und kann mit
`--disable-glossary` deaktiviert oder mit `--glossary-file` auf eine andere Datei
umgestellt werden.

Listenformat:

```json
[
  {
    "source": "Nahkampf",
    "target": "Melee",
    "case_sensitive": false,
    "whole_word": true
  },
  {
    "source": "Fernkampf",
    "target": "Ranged",
    "case_sensitive": false,
    "whole_word": true
  }
]
```

Einfaches Map-Format:

```json
{
  "Nahkampf": "Melee",
  "Fernkampf": "Ranged",
  "Rüstung": "Armor"
}
```

Das Glossar wirkt in zwei Schritten:

* Exakter Override: Wenn eine komplette Plain-Text-Einheit nach dem Trimmen exakt
  einem Glossar-Quellbegriff entspricht, wird direkt der Glossar-Zielbegriff
  verwendet und CTranslate2 nicht aufgerufen.
* Post-Processing: Nach Cache-Hit oder CTranslate2-Übersetzung werden verbliebene
  Glossar-Quellbegriffe im übersetzten Text ersetzt. Bei HTML/Rich-Text passiert
  das nur in Textknoten, nie in HTML-Tags.

## Verwendung

### Rebuild-Modus: neuen englischen Klon erstellen

```powershell
python main.py `
  --source-board "uXjVDEINBOARDID=" `
  --clone-name "[EN] Strategy Game" `
  --target-lang EN-US
```

### Board mit vollständiger Miro-URL übersetzen

```powershell
python main.py `
  --source-board "https://miro.com/app/board/uXjVDEINBOARDID=/" `
  --clone-name "[EN] Mein Workshop"
```

### Update-Modus: bestehenden englischen Klon aktualisieren

```powershell
python main.py `
  --update-existing-clone `
  --source-board "uXjVDEINBOARDID=" `
  --clone-board "uXjVEXISTINGCLONEID=" `
  --target-lang EN-US
```

Der Update-Modus aktualisiert standardmäßig nur die übersetzten Textinhalte.
Das ist schneller und vermeidet unnötige Layout-Patches, wenn sich Positionen,
Größen oder Styles seit dem letzten Lauf nicht geändert haben.

Der Update-Modus benötigt eine Sync-State-Datei. Diese wird im Rebuild-Modus
automatisch erzeugt. Wenn `--sync-state-file` nicht gesetzt ist, verwendet das
Script einen deterministischen Dateinamen wie:

```text
miro_sync_state_uXjVDEINBOARDID__uXjVEXISTINGCLONEID.json
```

### Sync-State für vorhandenen übersetzten Klon initialisieren

Wenn der englische Klon schon existiert, aber noch keine Sync-State-Datei
vorliegt:

```powershell
python main.py `
  --update-existing-clone `
  --initialize-sync-state `
  --source-board "uXjVDEINBOARDID=" `
  --clone-board "uXjVEXISTINGCLONEID=" `
  --target-lang EN-US
```

Die Initialisierung nutzt Positions-/Geometrie-Heuristiken, speichert die
Sync-State-Datei und beendet den Lauf. Danach den Update-Modus erneut ausführen.
Bei niedriger Mapping-Qualität bricht das Script ab, außer
`--force-initialize-sync-state` wird gesetzt.

### Update-Modus mit Löschen entfernter Quell-Items

```powershell
python main.py `
  --update-existing-clone `
  --source-board "uXjVDEINBOARDID=" `
  --clone-board "uXjVEXISTINGCLONEID=" `
  --delete-missing-items `
  --target-lang EN-US
```

Ohne `--delete-missing-items` werden fehlende Quell-Items nur gemeldet, aber
nicht aus dem englischen Klon gelöscht.

### Update-Modus mit Layout-Synchronisierung

Wenn auch Positionen, Größen oder Styles vom Quell-Board übernommen werden
sollen:

```powershell
python main.py `
  --update-existing-clone `
  --source-board "uXjVDEINBOARDID=" `
  --clone-board "uXjVEXISTINGCLONEID=" `
  --update-layout `
  --target-lang EN-US
```

Für explizite Text-only-Läufe kann alternativ `--text-only-update` oder
`--no-update-layout` gesetzt werden.

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
  --ct2-model-dir "models/nllb-200-distilled-1.3B-ct2" `
  --ct2-model-family "nllb" `
  --hf-tokenizer-model "facebook/nllb-200-distilled-1.3B" `
  --source-lang-code "deu_Latn" `
  --target-lang-code "eng_Latn" `
  --ct2-device "cpu" `
  --ct2-compute-type "int8" `
  --translation-batch-size 32 `
  --beam-size 4
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
  --translation-batch-size 32 `
  --beam-size 4
```

Auf einer RTX 5070 Ti mit 16 GB VRAM ist `--ct2-device cuda` mit
`--ct2-compute-type int8_float16` der empfohlene Startpunkt. Wenn Speicherfehler
auftreten, zuerst `--translation-batch-size 16` testen.

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

Wenn der Fehler `cublas64_12.dll is not found or cannot be loaded` erscheint,
ist zwar der NVIDIA-Treiber vorhanden, aber CTranslate2 findet die CUDA-12
Runtime-Bibliotheken nicht. Installiere das NVIDIA CUDA Toolkit 12.x und stelle
sicher, dass der `bin`-Ordner im `PATH` liegt, z. B.:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

In einer neuen PowerShell sollte danach `where cublas64_12.dll` einen Pfad in
diesem CUDA-Ordner ausgeben. Ohne CUDA Toolkit kann die CPU-Variante genutzt
werden.

Wenn CUDA installiert ist, aber `where cublas64_12.dll` nichts findet, wurde der
CUDA-`bin`-Ordner noch nicht in die aktuelle Umgebung übernommen. Eine neue
PowerShell oder ein Neustart von PyCharm reicht oft aus. Das Script versucht
zusätzlich, typische CUDA-12-Installationen wie
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin` automatisch für
die DLL-Suche zu registrieren.

### Testlauf ohne finale Miro-Updates

Der Klon wird im Rebuild-Modus erstellt und die Übersetzung wird vorbereitet.
Item-Patches, Item-Erstellung, Item-Löschung und der temporäre
Schreibrechte-Preflight werden übersprungen.

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
| `--update-existing-clone` | aus | Aktualisiert einen bestehenden übersetzten Klon statt einen neuen Klon zu erstellen. |
| `--clone-board` | leer | Bestehender übersetzter Klon als Board-ID oder URL; erforderlich mit `--update-existing-clone`. |
| `--sync-state-file` | automatisch | Pfad zur Sync-State-Datei. |
| `--initialize-sync-state` | aus | Erstellt eine Sync-State-Datei für einen bereits bestehenden übersetzten Klon und beendet den Lauf. |
| `--force-initialize-sync-state` | aus | Erlaubt Initialisierung auch bei niedriger Mapping-Qualität. |
| `--delete-missing-items` | aus | Löscht im Update-Modus Clone-Items, deren Source-Items nicht mehr existieren. |
| `--update-layout` / `--no-update-layout` | aus | Aktualisiert Position, Geometrie und Style im Update-Modus, soweit die Miro API es akzeptiert. |
| `--text-only-update` | implizit | Aktualisiert im Update-Modus nur übersetzte Textfelder; Alias für `--no-update-layout`. |
| `--sync-supported-items-only` / `--no-sync-supported-items-only` | an | Synchronisiert nur unterstützte translatable Item-Typen. |
| `--translator` | `ct2` | Übersetzungsbackend. Aktuell ist nur `ct2` unterstützt. |
| `--ct2-model-dir` | `models/nllb-200-distilled-1.3B-ct2` | Lokales konvertiertes CTranslate2-Modell. |
| `--ct2-model-family` | `nllb` | Modellfamilie für Decoding-Regeln: `nllb`, `marian` oder `auto`. |
| `--hf-tokenizer-model` | `facebook/nllb-200-distilled-1.3B` | Hugging-Face-Tokenizer-Modell oder lokaler Tokenizer-Pfad. |
| `--source-lang-code` | `deu_Latn` | Source-Sprachcode für mehrsprachige Modelle wie NLLB. |
| `--target-lang-code` | `eng_Latn` | Target-Sprachcode für mehrsprachige Modelle wie NLLB. |
| `--ct2-device` | `cpu` | CTranslate2-Gerät, z. B. `cpu` oder `cuda`. |
| `--ct2-compute-type` | `int8` | CTranslate2-Compute-Type, z. B. `int8`, `int8_float16`, `float32` oder `float16`. |
| `--translation-batch-size` | `32` | Anzahl Plain-Text-Einheiten pro lokaler Übersetzungsbatch. |
| `--beam-size` | `4` | Beam-Search-Größe; höhere Werte sind oft besser, aber langsamer. |
| `--preserve-special-symbols` / `--no-preserve-special-symbols` | an | Schützt Emojis und ähnliche Symbole vor der Übersetzung und setzt sie unverändert wieder ein. |
| `--glossary-file` | `translation_glossary_de_en.json` | JSON-Glossar für exakte Overrides und Post-Processing. |
| `--disable-glossary` | aus | Deaktiviert das Glossar. |
| `--sleep-after-copy` | `3.0` | Wartezeit nach dem Kopieren, bevor Items gelesen werden. |
| `--dry-run` | aus | Plant Updates und übersetzt lokal, schreibt aber keine Miro-Items. Im Rebuild-Modus wird weiterhin ein Klon erstellt. |
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

## Sicherheit und Hinweise

Der Rebuild-Modus erstellt bei jedem Lauf einen neuen englischen Klon. Der
Update-Modus aktualisiert einen bestehenden Klon anhand der Sync-State-Datei.

Für zuverlässige Updates sollte die Sync-State-Datei aus einem Rebuild-Lauf
stammen. Bei bereits übersetzten Klonen kann `--initialize-sync-state` eine
Best-Effort-Zuordnung über Position und Geometrie erstellen; diese Heuristik ist
nicht perfekt.

Vor destruktiven Änderungen empfiehlt sich ein Testlauf mit `--dry-run`.
Gelöscht wird nur, wenn `--delete-missing-items` explizit gesetzt ist.

Unsupported Miro-Item-Typen werden nicht destruktiv synchronisiert.

Vorteile:

* Original bleibt unverändert
* Rebuild-Modus bleibt einfach und robust
* Update-Modus kann bestehende englische Boards wiederverwenden

Nachteile:

* der Link zum englischen Board kann sich bei jedem Lauf ändern
* nicht alle Miro-Komponenten sind per API vollständig übersetzbar
* Update-Modus hängt von der Qualität der Sync-State-Zuordnung ab
