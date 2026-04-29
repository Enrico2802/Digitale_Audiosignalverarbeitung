"""Task 1 — The STFT: building a spectrogram from scratch.

You will implement the Short-Time Fourier Transform step by step:
slide a Hann-windowed frame across a signal, compute the FFT of each frame,
and assemble the results into a spectrogram.

Compare your plot with reference/task01_reference.png to check your result.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

FS = 8000
DURATION = 1.0


def main() -> None:
    n_total = int(DURATION * FS)
    t = np.arange(n_total) / FS

    # Signal: 440 Hz for the first half, 880 Hz for the second half
    x = np.where(t < 0.5,
                 np.sin(2 * np.pi * 440.0 * t),
                 np.sin(2 * np.pi * 880.0 * t))

    N_b = 512
    hop = N_b // 4   # 75 % overlap

    # TODO: Compute the STFT of x.
    #   Slide a Hann-windowed frame across x in steps of `hop`, rfft each frame,
    #   and collect the results. Stack into a 2-D dB magnitude array when done.
    #   Hint: np.hanning(N_b) gives the window; np.fft.rfftfreq(N_b, d=1/FS) gives the frequency axis.
    frames = []
    times = []
    S_db = None
    freqs = None

    fig, ax = plt.subplots(figsize=(11, 5))
    img = ax.pcolormesh(times, freqs, S_db,
                        shading="auto", cmap="inferno", vmin=-80, vmax=0)
    ax.set_ylim(0, 1500)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"STFT spectrogram — 440 Hz → 880 Hz  (N_b={N_b}, hop={hop})")
    plt.colorbar(img, ax=ax, label="Magnitude [dB]")
    plt.tight_layout()
    plt.show()

    # TODO:
    # 1. At what time does the frequency change in the spectrogram?
    #    Does it match the expected 0.5 s?
    #
    # 2. Change N_b to 128 and rerun. How do the frequency bands look compared to N_b=512?
    #    Can you still clearly see that the frequency changed at t=0.5 s?
    #
    # 3. Change N_b to 2048 and rerun. What do you observe near t=0.5 s?
    #    How sharp is the transition in time compared to N_b=128?
    #
    # 4. What value of N_b gives the best balance between time and frequency resolution
    #    for this signal? Justify your answer with Δf = fs / N_b.


if __name__ == "__main__":
    main()
