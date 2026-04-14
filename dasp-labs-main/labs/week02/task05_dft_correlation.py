from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dsp_utils import compute_rfft  # noqa: E402

# In the lecture we derived the DFT as repeated correlation:
#
#   X[k] = sum_{n=0}^{N-1} x[n] * e^{-j2pi*k*n/N}
#
# Using Euler's formula  e^{-j*theta} = cos(theta) - j*sin(theta):
#
#   X[k] =   sum x[n] * cos(2pi*k*n/N)      <- C_cos[k]: cosine correlation
#          - j * sum x[n] * sin(2pi*k*n/N)  <- C_sin[k]: sine correlation
#
# This task makes that explicit: compute one bin by hand and compare with rfft.


def main():
    fs = 48000
    duration = 1.0
    n = np.arange(int(duration * fs))
    x = 0.8 * np.sin(2 * np.pi * 440 * n / fs)
    N = len(x)
    n = np.arange(N)

    # --- Compute bin k = 440 by hand ---
    k = 440
    C_cos = np.sum(x * np.cos(2 * np.pi * k * n / N))
    C_sin = np.sum(x * np.sin(2 * np.pi * k * n / N))
    X_manual = C_cos - 1j * C_sin

    # Compare with numpy's rfft
    freqs, X_rfft = compute_rfft(x, fs)
    X_rfft_k = X_rfft[k]

    print(f"Bin k = {k}  →  frequency: {k * fs / N:.1f} Hz")
    print()
    print(f"  C_cos[k]  (cosine correlation) = {C_cos:>12.4f}")
    print(f"  C_sin[k]  (sine correlation)   = {C_sin:>12.4f}")
    print()
    print(f"  X_manual = C_cos - j*C_sin     = {X_manual:.4f}")
    print(f"  np.fft.rfft[k]                 = {X_rfft_k:.4f}")
    print(f"  Match: {np.isclose(X_manual, X_rfft_k)}")
    print()
    print(f"  |X[k]|   = {np.abs(X_manual):.4f}")
    print(f"  Re(X[k]) = {X_manual.real:.4f}   (equals C_cos)")
    print(f"  Im(X[k]) = {X_manual.imag:.4f}  (equals -C_sin)")

    # --- What about a bin with no matching tone? ---
    k_empty = 300
    C_cos_empty = np.sum(x * np.cos(2 * np.pi * k_empty * n / N))
    C_sin_empty = np.sum(x * np.sin(2 * np.pi * k_empty * n / N))
    X_empty = C_cos_empty - 1j * C_sin_empty

    print()
    print(f"Bin k = {k_empty}  →  frequency: {k_empty * fs / N:.1f} Hz  (no tone here)")
    print(f"  C_cos[k]  = {C_cos_empty:>10.6f}")
    print(f"  C_sin[k]  = {C_sin_empty:>10.6f}")
    print(f"  |X[k]|    = {np.abs(X_empty):.6f}  (near zero — signal does not correlate)")

    # TODO:
    # 1. Try k = 880. The signal contains a tone at 880 Hz — what do C_cos and C_sin show?
    # 2. Try k = 441. Close to the tone but not exact — what happens to |X[k]|?
    # 3. The signal uses sine (not cosine). Which of C_cos or C_sin is larger at k = 440?
    #    Why? (Hint: sin(x) = cos(x - pi/2))


if __name__ == "__main__":
    main()
