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
from scipy.signal import stft, istft

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_DIR = ROOT.parent / "assets" / "audio"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

FS     = 48000
N_b    = 2048
HOP    = N_b // 4
F_LOW  = 300.0
F_HIGH = 3000.0


def main() -> None:
    x, fs = sf.read(AUDIO_DIR / "speech.wav")
    assert fs == FS, f"Expected {FS} Hz, got {fs} Hz"
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[: int(3.0 * fs)]

    f_ax, t_ax, Zxx = stft(x, fs, nperseg=N_b, noverlap=N_b - HOP, window='hann')

    Zxx_filtered = Zxx.copy()
    low_idx  = np.searchsorted(f_ax, F_LOW)
    high_idx = np.searchsorted(f_ax, F_HIGH)
    Zxx_filtered[:low_idx, :]  = 0
    Zxx_filtered[high_idx:, :] = 0

    _, y = istft(Zxx_filtered, fs, nperseg=N_b, noverlap=N_b - HOP, window='hann')
    y = y[:len(x)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT_DIR / "speech_original.wav"),  x.astype(np.float32), fs)
    sf.write(str(OUTPUT_DIR / "speech_telephone.wav"), y.astype(np.float32), fs)
    print(f"Saved orignal and filtered WAVs to {OUTPUT_DIR}")

    S_db_orig     = 20 * np.log10(np.abs(Zxx)          + 1e-8)
    S_db_filtered = 20 * np.log10(np.abs(Zxx_filtered)  + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, S_db, title in [
        (axes[0], S_db_orig,     "Original speech"),
        (axes[1], S_db_filtered, f"Telephone filtered ({F_LOW:.0f}–{F_HIGH:.0f} Hz)"),
    ]:
        img = ax.pcolormesh(t_ax, f_ax, S_db,
                            shading="auto", cmap="inferno", vmin=-80, vmax=0)
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(title)
        ax.axhline(F_LOW,  color="cyan", linestyle="--", linewidth=1.5,
                   label=f"{F_LOW:.0f} Hz")
        ax.axhline(F_HIGH, color="lime", linestyle="--", linewidth=1.5,
                   label=f"{F_HIGH:.0f} Hz")
        ax.legend(loc="upper right", fontsize=9)
        plt.colorbar(img, ax=ax, label="Magnitude [dB]")

    plt.tight_layout()
    plt.show()

    _, x_rt = istft(Zxx, fs, nperseg=N_b, noverlap=N_b - HOP, window='hann')
    x_rt = x_rt[:len(x)]
    print(f"Unmodified round-trip error: {np.max(np.abs(x_rt - x)):.2e}")
    print(f"Filtered round-trip error:   {np.max(np.abs(y   - x)):.2e}")

    # Reflection answers:
    # 1. The filtered version sounds thin/tinny — bass and high frequencies removed.
    #    It clearly resembles a telephone call.
    #
    # 2. The cutoff is perfectly sharp (brick-wall). You may hear slight ringing
    #    at the cutoff frequencies due to the Gibbs phenomenon.
    #
    # 3. A hard step in frequency ↔ infinite sinc in time → ringing artefacts.
    #    Week 5 shows smooth (FIR/IIR) filters that avoid this.
    #
    # 4. The filtered round-trip error is much larger than the unmodified one:
    #    it now represents the intentionally removed frequency content,
    #    not numerical noise.
    #
    # 5. Lowering F_HIGH to 1000 Hz makes speech barely intelligible — consonants
    #    (which distinguish words) live largely above 1 kHz.


if __name__ == "__main__":
    main()
