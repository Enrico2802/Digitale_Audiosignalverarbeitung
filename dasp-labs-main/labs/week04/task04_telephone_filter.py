"""Task 4 — Telephone filter: frequency-domain processing with STFT → ISTFT.

You will apply a telephone bandpass filter (300–3000 Hz) to a speech recording
by zeroing out all STFT bins outside the passband. This is filtering entirely
in the frequency domain: no filter coefficients, no convolution — just set
unwanted bins to zero before the inverse transform.

The same STFT → modify → ISTFT pipeline is the basis of noise gates, vocoders,
spectral equalisers, and many real-world audio effects.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_DIR = ROOT.parent / "assets" / "audio"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

FS = 48000
N_b = 2048
HOP = N_b // 4    # 75 % overlap
F_LOW = 300.0        # Hz — telephone passband lower edge
F_HIGH = 3000.0      # Hz — telephone passband upper edge


def main() -> None:
    x, fs = sf.read(AUDIO_DIR / "speech.wav")
    assert fs == FS, f"Expected {FS} Hz, got {fs} Hz"
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[: int(3.0 * fs)]

    # TODO: Compute the STFT of x.
    #   Hint: from scipy.signal import stft
    #         f_ax, t_ax, Zxx = stft(x, fs, nperseg=N_b,
    #                                noverlap=N_b - HOP, window='hann')
    f_ax = None
    t_ax = None
    Zxx = None

    # TODO: Build Zxx_filtered — a copy of Zxx with all bins outside
    #   [F_LOW, F_HIGH] set to zero.
    #   Hint: np.searchsorted(f_ax, F_LOW) gives the first bin index >= F_LOW.
    Zxx_filtered = None

    # TODO: Reconstruct the filtered signal with iSTFT (same parameters as above).
    #   Hint: from scipy.signal import istft
    y = None

    # TODO: Save both the original and the filtered signal as WAV files.
    #   Listen to both and compare.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Plot side-by-side spectrograms of original and filtered signal.
    #   Hint: compute S_db = 20*np.log10(np.abs(Zxx) + 1e-8) for both,
    #         then use pcolormesh in two subplots.
    #         Add horizontal lines at F_LOW and F_HIGH to mark the passband.

    # Reflection questions:
    # 1. Listen to both files. Does the filtered version sound like a telephone?
    #    What is missing compared to the original?
    #
    # 2. Look at the spectrograms. Is the cutoff sharp?
    #    Do you hear any ringing or artefacts in the filtered signal?
    #    (This kind of hard cutoff is called a "brick-wall" filter.)
    #
    # 3. Brick-wall filters cause ringing because a sudden step in the frequency
    #    domain corresponds to an infinitely long sinc function in the time domain.
    #    Week 5 will show how to design smoother filters that avoid this.
    #
    # 4. The round-trip is no longer perfect once you modify Zxx.
    #    Compute max|y[:len(x)] - x| and compare it to the unmodified round-trip
    #    error from Task 3. What does the error represent now?
    #
    # 5. Try changing F_HIGH to 1000 Hz. How does the speech intelligibility
    #    change? At what cutoff does it become hard to understand?


if __name__ == "__main__":
    main()
