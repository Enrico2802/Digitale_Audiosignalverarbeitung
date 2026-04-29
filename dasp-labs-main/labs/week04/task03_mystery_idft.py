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

    x = np.fft.irfft(X, n=N)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT_DIR / "mystery_reconstructed.wav"), x.astype(np.float32), fs)
    print(f"Saved: {OUTPUT_DIR / 'mystery_reconstructed.wav'}")

    freqs        = np.fft.rfftfreq(N, d=1 / fs)
    magnitude_db = 20 * np.log10(np.abs(X) + 1e-8)
    t_axis       = np.arange(N) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(freqs, magnitude_db, linewidth=0.8, color="steelblue")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title("Mystery Signal — Magnitude Spectrum")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_axis, x, linewidth=0.6, color="steelblue")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Reconstructed Time-Domain Signal")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    X_recomputed = np.fft.rfft(x)
    print(f"Round-trip matches original X: {np.allclose(X, X_recomputed)}")
    print(f"Max reconstruction error:      {np.max(np.abs(X - X_recomputed)):.2e}")

    # Reflection answers:
    # 1. Listen to the output WAV to identify the mystery signal.
    #
    # 2. The magnitude spectrum shows where energy is concentrated.
    #    Harmonic peaks = pitched sound; broadband = noise/transient.
    #
    # 3. rfft returns N//2 + 1 complex bins (for even N).
    #    The upper half is the complex conjugate mirror of the lower half —
    #    it carries no new information for real-valued signals.
    #
    # 4. np.allclose(X, np.fft.rfft(x)) should print True (within float64 precision).


if __name__ == "__main__":
    main()
