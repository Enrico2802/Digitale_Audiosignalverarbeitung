"""Task 2 — Window functions: can we see all the harmonics?

You will apply four different windows to the same short frame used in Task 1.
The goal is to see how window choice affects leakage and which harmonics become
visible — especially the weak ones.

Compare your plot with reference/task02_reference.png to check your result.
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
N = 4096
ATTACK_SEC = 0.085


def main() -> None:
    x, fs = sf.read(AUDIO_PATH)
    assert fs == FS_EXPECTED, f"Expected {FS_EXPECTED} Hz, got {fs} Hz"

    if x.ndim == 2:
        x = x.mean(axis=1)

    start = int(ATTACK_SEC * fs)
    frame = x[start:start + N]

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    windows = {
        "Rectangular": np.ones(N),
        "Hann":        np.hanning(N),
        "Hamming":     np.hamming(N),
        "Blackman":    np.blackman(N),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True, sharex=True)

    for ax, (name, w) in zip(axes.flat, windows.items()):

        # ---------------------------------------------------------------------
        # TODO: Compute the windowed magnitude spectrum for this window w.
        #
        # Step 1: apply the window by multiplying: frame * w
        #
        # Step 2: compute the DFT: np.fft.rfft(frame * w)
        #
        # Step 3: compute magnitude in dB and peak-normalise so the strongest bin is 0 dB:
        #         mag_db = 20 * np.log10(np.abs(X) + 1e-9)
        #         mag_db -= mag_db.max()
        #
        # Replace the two lines below with your solution:
        X = np.fft.rfft(frame * w)
        mag_db = 20 * np.log10(np.abs(X) + 1e-9)
        mag_db -= mag_db.max()
        # ---------------------------------------------------------------------

        ax.plot(freqs, mag_db, linewidth=1.2, color="tab:blue")
        ax.set_xlim(0, 1500)
        ax.set_ylim(-70, 5)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax in axes[:, 0]:
            ax.set_ylabel("Magnitude [dB]")
        if ax in axes[1, :]:
            ax.set_xlabel("Frequency [Hz]")

    fig.suptitle(
        f"Window comparison — guitar E string  (N = {N}, Δf = {fs/N:.1f} Hz)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    # TODO:
    # 1. Look at the region around 916 Hz in each plot.
    #    A harmonic is visible there with tapered windows but buried in leakage
    #    with the rectangular window. Why does it disappear with rectangular?
    #    Hint: think about the side-lobe level of the rectangular window and
    #    the level of the neighbouring stronger harmonics.
    #
    # 2. Count how many harmonics are visible above −50 dB in each window.
    #    Which window reveals the most? Which reveals the fewest?
    #
    # 3. Look at the harmonics above 400 Hz where some are much weaker than
    #    their neighbours. With rectangular, some weak harmonics are hidden;
    #    with Blackman they emerge. What determines whether a weak harmonic
    #    survives — is it the window's side-lobe level, the distance to the
    #    nearest strong harmonic, or both?
    #
    # 4. The rectangular window has noticeably narrower peaks than the others.
    #    This is the trade-off: lower side lobes come at the cost of a wider main lobe.
    #    When would the rectangular window's narrow peaks be an advantage?
    #    (Hint: think about two harmonics that are very close in frequency.)
    #
    # 5. Which window would you choose for analysing this guitar note, and why?


if __name__ == "__main__":
    main()
