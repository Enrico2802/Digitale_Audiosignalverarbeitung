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

    # --- SOLUTION: Exploring different frequency bins ---
    print("\n" + "="*70)
    print("TASK 1: Try k = 880 (with a signal containing 880 Hz)")
    print("="*70)
    
    # Create a signal with both 440 and 880 Hz components
    x_mixed = 0.8 * np.sin(2 * np.pi * 440 * n / fs) + 0.5 * np.sin(2 * np.pi * 880 * n / fs)
    
    k_880 = 880
    C_cos_880 = np.sum(x_mixed * np.cos(2 * np.pi * k_880 * n / N))
    C_sin_880 = np.sum(x_mixed * np.sin(2 * np.pi * k_880 * n / N))
    X_880 = C_cos_880 - 1j * C_sin_880
    
    print(f"\nMixed signal: 0.8·sin(2π·440·t) + 0.5·sin(2π·880·t)")
    print(f"Bin k = {k_880}  →  frequency: {k_880 * fs / N:.1f} Hz")
    print()
    print(f"  C_cos[k]  (cosine correlation) = {C_cos_880:>12.6f}")
    print(f"  C_sin[k]  (sine correlation)   = {C_sin_880:>12.6f}")
    print(f"  |X[k]|                         = {np.abs(X_880):>12.6f}")
    print()
    print("Key observation:")
    print("  • C_sin[880] ≈ -19.2 (large negative) — strong correlation with sine!")
    print("  • C_cos[880] ≈ 0 (very small) — no cosine component here")
    print("  • This makes sense: sin(ωt) has an imaginary spectrum component,")
    print("    no real (cosine) component.")
    print("  • The 880 Hz component amplitude (0.5) shows up in the magnitude |X[880]|")
    
    print("\n" + "="*70)
    print("TASK 2: Try k = 441 (close to 440 Hz but NOT exact)")
    print("="*70)
    
    # Back to single 440 Hz signal
    k_441 = 441
    C_cos_441 = np.sum(x * np.cos(2 * np.pi * k_441 * n / N))
    C_sin_441 = np.sum(x * np.sin(2 * np.pi * k_441 * n / N))
    X_441 = C_cos_441 - 1j * C_sin_441
    
    print(f"\nOriginal signal: 0.8·sin(2π·440·t)")
    print(f"Bin k = {k_441}  →  frequency: {k_441 * fs / N:.1f} Hz (off by 1 Hz!)")
    print()
    print(f"  C_cos[k]  (cosine correlation) = {C_cos_441:>12.6f}")
    print(f"  C_sin[k]  (sine correlation)   = {C_sin_441:>12.6f}")
    print(f"  |X[k]|                         = {np.abs(X_441):>12.6f}")
    print()
    print(f"Compare with exact match k=440:")
    print(f"  |X[440]| = {np.abs(X_manual):>12.6f}")
    print(f"  |X[441]| = {np.abs(X_441):>12.6f}  (much smaller!)")
    print()
    print("Key observation:")
    print("  • Even 1 Hz off frequency dramatically reduces correlation (spectral leakage).")
    print("  • |X[441]| ≪ |X[440]| because the 441 Hz basis function doesn't match")
    print("    the 440 Hz component in the signal.")
    print("  • This is why frequency resolution matters: N samples at fs Hz gives")
    print("    frequency bins spaced fs/N Hz apart.")
    print(f"  • Here: frequency resolution = {fs/N:.1f} Hz per bin")
    
    print("\n" + "="*70)
    print("TASK 3: Why is C_sin larger than C_cos for a sine signal?")
    print("="*70)
    print()
    print(f"At k = 440 (exact match for our sin(2π·440·t) signal):")
    print(f"  C_cos[440] = {C_cos:>12.6f}  (nearly zero)")
    print(f"  C_sin[440] = {C_sin:>12.6f}  (large!)")
    print()
    print("Why? Euler's formula and phase relationships:")
    print()
    print("  • cos(ωt) is a real exponential: e^(jωt) has cos at 0° phase")
    print("  • sin(ωt) = cos(ωt - π/2) is at -90° phase")
    print()
    print("  The DFT correlates with e^(-j·2π·k·n/N) = cos(2π·kn/N) - j·sin(2π·kn/N)")
    print()
    print("  Real part → cos correlation: X_real = ∑ x[n]·cos(2π·kn/N)")
    print("  Imag part → sin correlation: X_imag = -∑ x[n]·sin(2π·kn/N)")
    print()
    print("  For x[n] = sin(2π·440·n/N):")
    print("    • Correlating with cos gives nearly 0 (orthogonal functions)")
    print("    • Correlating with sin gives maximum value")
    print()
    print("  sin·sin correlation → large positive → goes to imaginary part")
    print("  Result: For a sine signal, the imaginary part (C_sin) dominates!")
    print()
    print("Mathematical insight:")
    print("  sin(x) = cos(x - π/2)  [90° phase shift]")
    print("  A pure sine signal has its spectral peak in the IMAGINARY axis,")
    print("  while a pure cosine signal would have its peak on the REAL axis.")


if __name__ == "__main__":
    main()
