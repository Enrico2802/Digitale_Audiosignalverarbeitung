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

        # TODO: Compute the STFT of x and convert to dB magnitude.
        #   Hint: from scipy.signal import stft
        #         f, t, Zxx = stft(x, fs, nperseg=N_b, noverlap=N_b-HOP, window='hann')
        f = None
        t = None
        S_db = None

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
    #    What frequency range carries most of the energy?
    #    Can you identify any moment where a consonant (e.g. 's', 't') appears?
    #    How does a consonant look different from a vowel?
    #
    # 2. Guitar: zoom in by temporarily changing ax.set_ylim(0, 8000) to
    #    ax.set_ylim(0, 1000) and rerunning. You know from Week 3 that the
    #    fundamental is 82.4 Hz. Can you see horizontal lines at multiples of
    #    that frequency? How many harmonics are clearly visible?
    #
    # 3. Guitar (full y-axis view): look at the brightness at t=0 vs. t=1 s.
    #    Up to roughly what frequency does energy reach at the attack?
    #    How high does it reach during the sustained portion?
    #    What does this tell you about how a plucked string sounds over time?
    #
    # 4. Speech vs. guitar: which signal has more energy above 4 kHz?
    #    What does that tell you about the difference in their timbres?


if __name__ == "__main__":
    main()
