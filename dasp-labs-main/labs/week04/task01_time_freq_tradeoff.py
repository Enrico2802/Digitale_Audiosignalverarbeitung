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

    window = np.hanning(N_b)
    frames = []
    times  = []

    for start in range(0, n_total - N_b + 1, hop):
        frame = x[start:start + N_b] * window
        frames.append(np.fft.rfft(frame))
        times.append((start + N_b // 2) / FS)

    S     = np.abs(np.array(frames)).T          # shape: (n_freqs, n_frames)
    S_db  = 20 * np.log10(S + 1e-9)
    S_db -= S_db.max()
    freqs = np.fft.rfftfreq(N_b, d=1 / FS)
    times = np.array(times)

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
    #    → Yes, the bright band clearly shifts from ~440 Hz to ~880 Hz at t≈0.5 s.
    #
    # 2. Change N_b to 128 and rerun. How do the frequency bands look compared to N_b=512?
    #    Can you still clearly see that the frequency changed at t=0.5 s?
    #    → N_b=128: Δf=62.5 Hz — wide blurry bands, poor frequency resolution.
    #      The transition at 0.5 s is still visible and very sharp in time.
    #
    # 3. Change N_b to 2048 and rerun. What do you observe near t=0.5 s?
    #    How sharp is the transition in time compared to N_b=128?
    #    → N_b=2048: Δf=3.9 Hz — very narrow sharp bands, excellent frequency resolution.
    #      But the transition smears over ~0.25 s because each frame covers 256 ms.
    #
    # 4. What value of N_b gives the best balance between time and frequency resolution
    #    for this signal? Justify your answer with Δf = fs / N_b.
    #    → N_b=512 is a good balance: Δf = 8000/512 = 15.6 Hz (resolves 440 vs 880 Hz),
    #      Δt = 512/8000 = 64 ms (short enough to see the 0.5 s transition cleanly).


if __name__ == "__main__":
    main()
