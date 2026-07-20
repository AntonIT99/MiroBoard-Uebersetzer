# Miro Board Exporter – Python/CDP v10.2

## Änderung in 10.2: Lücken zwischen benachbarten Kacheln

Version 10.1 konnte den tatsächlichen Maßstab zwischen Miros Zoom-to-fit-Ansicht und dem Export-Zoom auf sehr breiten Boards unterschätzen. Dadurch wurde eine benachbarte Kachel weiter verschoben als das berechnete Raster vorsah. Sichtbarer Effekt: Zwischen zwei horizontal nebeneinanderliegenden Bildern fehlte ein Streifen des Boards.

Version 10.2 korrigiert dies auf zwei Ebenen:

1. Vor der Maßstabsmessung wird tatsächlicher Board-Inhalt in die Bildschirmmitte bewegt. Die Kalibrierung arbeitet damit nicht mehr überwiegend auf schwarzem Hintergrund oder wiederholten Miniaturmustern.
2. Das Raster erzwingt zusätzlich einen konservativen Mindestüberlappungsbereich von standardmäßig 65 Prozent horizontal und vertikal.

Die neuen Einstellungen stehen in `config.json`:

```json
"absolute_navigation": {
  "center_content_before_scale_calibration": true,
  "prefer_horizontal_scale": true,
  "calibration_drag_fraction": 0.3,
  "calibration_probe_width": 1800,
  "minimum_overlap_fraction_x": 0.65,
  "minimum_overlap_fraction_y": 0.65
}
```

`prefer_horizontal_scale` ist für sehr breite Boards sinnvoll: Miro zoomt in beide Richtungen gleich. Die horizontale Messung enthält auf solchen Boards aber deutlich mehr erkennbare Strukturen als die vertikale und ist deshalb zuverlässiger.

### Überlappung verändern

Mehr Sicherheit gegen ausgelassene Streifen:

```json
"minimum_overlap_fraction_x": 0.72,
"minimum_overlap_fraction_y": 0.70
```

Weniger Bilder und geringere Überlappung:

```json
"minimum_overlap_fraction_x": 0.55,
"minimum_overlap_fraction_y": 0.55
```

Werte zwischen `0.20` und `0.85` sind zulässig. Ein Wert von `0.65` bedeutet, dass der geplante Abstand zwischen zwei Kacheln höchstens 35 Prozent der Bildbreite beziehungsweise Bildhöhe beträgt.

## Auswahl der Detailstufe

Beim Start erscheint:

```text
Detailstufe festlegen:
  1. Miro-Zoom in Prozent direkt vorgeben (Standard)
  2. Maximale Anzahl Bilder vorgeben

Auswahl [1]:
```

Option 1 ist weiterhin die Standardoption. Miro verwendet feste Zoomstufen; das Programm wählt deshalb die erreichbare Stufe, die dem eingegebenen Wert am nächsten liegt.

## Bestehende Installation aktualisieren

Aus dem ZIP diese Dateien in den bisherigen Exporter-Ordner kopieren und ersetzen:

```text
export_miro_board.py
config.json
README.md
```

Neue Python-Pakete sind nicht erforderlich. `setup.bat` muss normalerweise nicht erneut ausgeführt werden.

Danach das separate Chrome-Fenster vollständig schließen und neu starten:

```text
1_start_miro_chrome.bat
2_export_board.bat
```

Für den ersten Test empfiehlt sich derselbe Zoom wie beim fehlerhaften Lauf. Die Konsolenausgabe zeigt jetzt eine größere Überlappung, beispielsweise:

```text
Kachelschritt: 717px horizontal / 236px vertikal
(Überlappung 1331px / 439px)
```

Die exakten Zahlen hängen von Fenstergröße und Crop-Einstellungen ab.

## Weitere Standardeinstellungen

Leere Bilder werden nicht in das PDF aufgenommen:

```json
"pdf_include_blank_tiles": false
```

Die Dateireihenfolge bleibt strikt zeilenweise: links nach rechts und danach von oben nach unten. Jede Kachel wird unabhängig aus der Zoom-to-fit-Ansicht positioniert.
