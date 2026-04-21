"""Task 1 — Spectral leakage on a real guitar recording.

You will analyse a short frame of a plucked guitar string using a rectangular
window. The goal is to observe spectral leakage and understand what causes it.

Compare your plot with reference/task01_reference.png to check your result.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_PATH = ROOT.parent / "assets" / "audio" / "guitar_e_string.wav"

FS_EXPECTED = 48000
N = 4096          # analysis frame: ~85 ms
ATTACK_SEC = 0.085


def main() -> None:
    x, fs = sf.read(AUDIO_PATH)
    assert fs == FS_EXPECTED, f"Expected {FS_EXPECTED} Hz, got {fs} Hz"

    if x.ndim == 2:
        x = x.mean(axis=1)

    # Extract one frame near the attack
    start = int(ATTACK_SEC * fs)
    frame = x[start:start + N]

    print(f"Sample rate          : {fs} Hz")
    print(f"Frame length         : {N} samples  ({N / fs * 1000:.1f} ms)")
    print(f"Frequency resolution : Δf = {fs / N:.2f} Hz")

    # -------------------------------------------------------------------------
    # TODO: Compute the magnitude spectrum of `frame` using a rectangular window
    #       (i.e. no windowing — just use the frame as-is).
    #
    # Step 1: compute the DFT using np.fft.rfft(frame)
    #         → this returns complex values X[k]
    #
    # Step 2: compute the frequency axis using np.fft.rfftfreq(N, d=1.0/fs)
    #         → this returns the frequency in Hz for each bin k
    #
    # Step 3: compute magnitude in dB:
    #         mag_db = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-9)
    #         (the + 1e-9 prevents log(0))
    #
    # Replace the three lines below with your solution:
    X = None
    freqs = None
    mag_db = None
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freqs, mag_db, linewidth=1.2, color="tab:blue")
    ax.set_xlim(0, 1500)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(f"Guitar E string — rectangular window  (N = {N}, Δf = {fs/N:.1f} Hz)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # TODO:
    # 1. How many prominent peaks can you count in the spectrum?
    #    Each peak corresponds to one harmonic of the string's fundamental.
    #    Estimate the fundamental frequency from the spacing between peaks.
    #
    # 2. Look at the base of each peak — energy spreads into neighbouring
    #    bins. This is spectral leakage. Why does it happen?
    #    Hint: think back to what the lecture showed about correlation products
    #    when a tone does not land exactly on a bin frequency.
    #
    # 3. Δf = fs / N = 11.7 Hz for this frame.
    #    You want to measure the fundamental frequency to within ±2 Hz accuracy.
    #    What is the minimum N you would need?  How long is that in milliseconds?


if __name__ == "__main__":
    main()
