"""Task 3 — The mystery spectrum.

A file mystery_spectrum.npz contains the rfft output of an unknown signal.
Your job is to reconstruct the time-domain signal using the IDFT and listen
to what comes out.

After listening: look at the magnitude spectrum and try to connect what you
hear to what you see.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

SPECTRUM_FILE = Path(__file__).resolve().parents[1] / "assets" / "numpy_data" / "mystery_spectrum.npz"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"


def main() -> None:
    data = np.load(SPECTRUM_FILE)
    X = data["X"]           # complex rfft output
    N = int(data["N"])      # original signal length
    fs = int(data["fs"])    # sample rate

    print(f"Loaded spectrum: {len(X)} bins, fs={fs} Hz, original length N={N}")

    # TODO: Reconstruct the time-domain signal from the spectrum.
    #   Hint: np.fft.irfft(X, n=N) undoes the rfft.
    x = None

    # TODO: Save the reconstructed signal as a WAV file and listen to it.
    #   Hint: soundfile.write(path, signal, samplerate)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Plot the magnitude spectrum in dB and the reconstructed waveform.
    #   Hint: freqs = np.fft.rfftfreq(N, d=1/fs)
    #         magnitude_db = 20 * np.log10(np.abs(X) + 1e-8)

    # Reflection questions:
    # 1. Listen to the output WAV. What is the mystery signal?
    #
    # 2. Look at the magnitude spectrum. What frequency range carries most of
    #    the energy? Does the shape match what you heard?
    #
    # 3. The rfft returns only the positive-frequency half of the spectrum.
    #    How many complex values does X contain compared to N samples?
    #    Where did the other half go?
    #
    # 4. Recompute the rfft of x and compare it to the original X.
    #    Are they identical? (Hint: np.allclose)


if __name__ == "__main__":
    main()
