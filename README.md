# Digitale Audiosignalverarbeitung

Dieses Repository enthält die Übungen zur digitalen Audiosignalverarbeitung. Es ist in mehrere Laborwochen aufgeteilt und enthält Python-Skripte zur Erzeugung, Analyse und Visualisierung von Audiosignalen.

## Projektstruktur

- `assets/audio/`
  - Beispiel-Audiodateien wie `speech.wav` und `music.wav`
- `labs/src/`
  - Hilfsbibliotheken für die Aufgaben, z. B. `audio_utils.py` und `dsp_utils.py`
- `labs/week01/`
  - Grundlagen: Audio laden, anzeigen, herunter- und hochrechnen, quantisieren
- `labs/week02/`
  - FFT, DFT-Korrelation, Frequenzauflösung, On-bin/Off-bin, echte Audio-FFT

## Voraussetzungen

- Python 3.8+ oder neuer
- Empfohlene Bibliotheken:
  - `numpy`
  - `matplotlib`
  - `soundfile`

## Installation

Im Projektverzeichnis:

```bash
python -m pip install numpy matplotlib soundfile
```

## Verwendung

Jedes Laborskript kann direkt ausgeführt werden. Beispiel:

```bash
python dasp-labs-main/labs/week02/task04_fft_basics.py
```

Für andere Aufgaben die entsprechende Datei auswählen, z. B.:

- `task05_dft_correlation.py`
- `task06_frequency_bins.py`
- `task07_on_off_bin.py`
- `task08_real_audio_fft.py`

## Inhalte der Woche 02

- `task04_fft_basics.py`: Untersuchung von FFT-Ausgaben, Magnituden- und Phasenspektren
- `task05_dft_correlation.py`: DFT als Korrelation mit Cosinus und Sinus
- `task06_frequency_bins.py`: Frequenzauflösung und Bin-Abstand
- `task07_on_off_bin.py`: On-bin vs. Off-bin und spektrale Leakage
- `task08_real_audio_fft.py`: Analyse realer Audiodaten (Sprache und Musik)

## Hinweise

- `labs/src/__init__.py` macht das `src`-Verzeichnis importierbar.
- `dsp_utils.py` enthält Funktionen zur FFT-Berechnung und Plot-Erstellung.
- `audio_utils.py` enthält Audio-Ein-/Ausgabe- und Hilfsfunktionen.

## Weiteres

- Zum Einstieg eignen sich besonders die Skripte in `labs/week01/`.
- Für visuelle Ergebnisse sollten Sie in einer Umgebung mit Anzeigeunterstützung arbeiten (z. B. lokales Python, VS Code mit GUI).
