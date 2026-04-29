"""Task 2 — Spectrograms of real audio with scipy.

In Task 1 you implemented the STFT loop yourself. Now you can use
scipy.signal.stft — one function call that does the same thing.
The goal here is not the mechanics but reading what the spectrogram tells you
about real signals: speech and a guitar recording.

Compare your plots with reference/task02a_reference.png (speech) and
reference/task02b_reference.png (guitar) to check your result.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_DIR = ROOT.parent / "assets" / "audio"

N_b = 2048
HOP = N_b // 4   # 75 % overlap — standard choice


def main() -> None:
    for filename, label, fs_expected in [
        ("speech.wav",          "Speech",         48000),
        ("guitar_e_string.wav", "Guitar E string", 48000),
    ]:
        x, fs = sf.read(AUDIO_DIR / filename)
        assert fs == fs_expected, f"Expected {fs_expected} Hz, got {fs} Hz"
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x[:int(3.0 * fs)]   # first 3 seconds

        f, t, Zxx = stft(x, fs, nperseg=N_b, noverlap=N_b - HOP, window='hann')
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-8)

        fig, ax = plt.subplots(figsize=(12, 5))
        img = ax.pcolormesh(t, f, S_db,
                            shading="auto", cmap="inferno", vmin=-80, vmax=0)
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{label} spectrogram  (N_b={N_b}, hop={HOP},  Δf={fs/N_b:.1f} Hz)")
        plt.colorbar(img, ax=ax, label="Magnitude [dB]")
        plt.tight_layout()
        plt.show()

    # TODO:
    # 1. Speech: can you spot pauses between words or breath sounds in the spectrogram?
    #    → Silent gaps appear as near-black columns. Sibilants (s, t) show broadband
    #      energy above 4 kHz. Vowels show concentrated horizontal formant bands.
    #
    # 2. Guitar: zoom in (ylim 0–1000 Hz). Fundamental at 82.4 Hz and harmonics
    #    at 164.8, 247.2, 329.6 Hz etc. are visible as bright horizontal lines.
    #
    # 3. Guitar attack vs. sustained: at t≈0 the spectrogram is bright across the
    #    full frequency range (broadband attack transient). From t≈0.1 s onward
    #    only the harmonic lines remain — the string settles into its resonant modes.
    #
    # 4. Speech vs. guitar: speech has more energy above 4 kHz (fricatives, sibilants).
    #    Guitar energy is concentrated in the harmonic series and decays above ~2 kHz.


if __name__ == "__main__":
    main()
