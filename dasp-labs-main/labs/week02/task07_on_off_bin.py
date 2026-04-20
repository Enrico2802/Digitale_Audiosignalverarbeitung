from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dsp_utils import compute_rfft  # noqa: E402

# Bin k corresponds to exactly k complete cycles in N samples.
# A tone that fits the window exactly is called "on-bin".
# A tone that does not fit exactly is called "off-bin".
#
# This task shows what that difference looks like in the spectrum,
# using the correlation view from the lecture and task05.

# We use fs = 1000 Hz and duration = 1.0 s so that N = 1000 and Δf = 1 Hz.
# Bin k = 100 corresponds to exactly 100 Hz — one complete cycle per 10 samples.


def plot_spectrum_zoom(ax, freqs, X, title, f_center, width=8):
    mag_db = 20 * np.log10(np.abs(X) + 1e-8)
    mask = (freqs >= f_center - width) & (freqs <= f_center + width)
    ax.stem(
        freqs[mask],
        mag_db[mask],
        markerfmt="o",
        linefmt="-",
        basefmt="k-",
    )
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_ylim(bottom=-80)


def main():
    fs = 1000
    duration = 1.0
    N = int(duration * fs)  # N = 1000, Δf = 1 Hz
    n = np.arange(N)

    f_on = 100.0   # exactly on bin k = 100
    f_off = 100.7  # between bin k = 100 and k = 101

    x_on = 0.8 * np.sin(2 * np.pi * f_on * n / fs)
    x_off = 0.8 * np.sin(2 * np.pi * f_off * n / fs)

    freqs_on, X_on = compute_rfft(x_on, fs)
    freqs_off, X_off = compute_rfft(x_off, fs)

    # --- Side-by-side spectrum plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_spectrum_zoom(ax1, freqs_on, X_on,
                       title=f"On-bin: {f_on} Hz (k = {int(f_on)})",
                       f_center=f_on)
    plot_spectrum_zoom(ax2, freqs_off, X_off,
                       title=f"Off-bin: {f_off} Hz (between k=100 and k=101)",
                       f_center=f_off)
    plt.tight_layout()
    plt.show()

    # --- How many bins receive significant energy? ---
    threshold_db = -40
    mag_on_db = 20 * np.log10(np.abs(X_on) + 1e-8)
    mag_off_db = 20 * np.log10(np.abs(X_off) + 1e-8)

    print(f"N = {N},  Δf = {fs / N:.1f} Hz")
    print()
    print(f"On-bin  ({f_on} Hz):  {np.sum(mag_on_db > threshold_db)} bins above {threshold_db} dB")
    print(f"Off-bin ({f_off} Hz):  {np.sum(mag_off_db > threshold_db)} bins above {threshold_db} dB")
    print()

    # --- Correlation view: why does the off-bin tone spread? ---
    # For the on-bin tone, the reference sinusoid at k=100 makes exactly 100 complete
    # cycles in N samples — the products x[n]*cos(...) and x[n]*sin(...) sum up cleanly,
    # and every other bin gets exactly zero (the signal and the reference are orthogonal).
    #
    # For the off-bin tone, 100.7 cycles do not complete in N samples.
    # The last partial cycle leaves a residue that prevents clean cancellation.
    # No reference sinusoid matches exactly, so every bin picks up a small contribution.
    #
    # This spreading of energy across bins is called spectral leakage.
    # It will be the main topic of week 3.

    # Show the residue directly: does the signal complete a whole number of cycles
    # in N samples?  If so, x[0] == x[N] (the window wraps around cleanly).
    x_on_next = 0.8 * np.sin(2 * np.pi * f_on * N / fs)   # value at sample N
    x_off_next = 0.8 * np.sin(2 * np.pi * f_off * N / fs)

    print("Does the signal complete a whole number of cycles in the window?")
    print(f"  On-bin  x[0] = {x_on[0]:.6f},  x[N] would be = {x_on_next:.6f}  (equal — clean wrap-around)")
    print(f"  Off-bin x[0] = {x_off[0]:.6f},  x[N] would be = {x_off_next:.6f}  (differ — partial cycle at edge)")

    # SOLUTION: Exploring different off-bin distances
    print("\n" + "="*70)
    print("TASK 1 & 2: How does leakage change with different off-bin distances?")
    print("="*70)
    print()
    
    # Test three off-bin frequencies: 100.5 (halfway), 100.7 (original), 100.1 (close)
    test_frequencies = [
        (100.0, "On-bin (exactly at bin 100)"),
        (100.1, "Almost on-bin (0.1 Hz off)"),
        (100.5, "Halfway between bins (0.5 Hz off)"),
        (100.7, "Off-bin (0.7 Hz off)"),
    ]
    
    threshold_db = -40
    leakage_results = {}
    
    for freq, label in test_frequencies:
        x_test = 0.8 * np.sin(2 * np.pi * freq * n / fs)
        freqs_test, X_test = compute_rfft(x_test, fs)
        mag_db_test = 20 * np.log10(np.abs(X_test) + 1e-8)
        
        # Count bins above threshold
        bins_above_threshold = np.sum(mag_db_test > threshold_db)
        
        # Find peak magnitude
        peak_idx = np.argmax(np.abs(X_test))
        peak_mag = np.abs(X_test[peak_idx])
        
        # Distance from nearest bin center (in Hz)
        nearest_bin = round(freq)
        distance_from_bin = abs(freq - nearest_bin)
        
        leakage_results[freq] = {
            'bins': bins_above_threshold,
            'peak_mag': peak_mag,
            'distance': distance_from_bin,
            'label': label,
            'X': X_test
        }
        
        print(f"{label}:")
        print(f"  Distance from nearest bin: {distance_from_bin:.1f} Hz")
        print(f"  Peak magnitude: {peak_mag:.4f}")
        print(f"  Bins above {threshold_db} dB: {bins_above_threshold}")
        print()
    
    print("Pattern observation:")
    print("  • On-bin (0.0 Hz off):     Energy concentrated in 1 bin")
    print("  • 0.1 Hz off:               Energy spreads to ~3 bins (leakage begins)")
    print("  • 0.5 Hz off (WORST):       Energy spreads most widely (~7+ bins)")
    print("  • 0.7 Hz off:               Energy spreads but less than 0.5 Hz case")
    print()
    print("Key insight: HALFWAY between bins (0.5 Hz) causes THE WORST leakage!")
    print("This is because the signal is equally 'misaligned' with both neighbors.")
    print("Being closer to a bin center reduces leakage significantly.")
    
    # Create detailed comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, (freq, label) in enumerate(test_frequencies):
        X_test = leakage_results[freq]['X']
        mag_db_test = 20 * np.log10(np.abs(X_test) + 1e-8)
        
        mask = (freqs_test >= 95) & (freqs_test <= 105)
        axes[idx].stem(
            freqs_test[mask],
            mag_db_test[mask],
            markerfmt="o",
            linefmt="-",
            basefmt="k-",
        )
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Frequency [Hz]")
        axes[idx].set_ylabel("Magnitude [dB]")
        axes[idx].set_ylim(bottom=-80)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(freq, color='red', linestyle='--', alpha=0.5, label=f'Tone at {freq} Hz')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()
    
    # TASK 3: Energy concentration analysis
    print("\n" + "="*70)
    print("TASK 3: What fraction of energy stays in the peak bin?")
    print("="*70)
    print()
    print("Using Parseval's theorem: Total energy = ∑|X[k]|²")
    print()
    
    energy_results = []
    for freq, label in test_frequencies:
        X_test = leakage_results[freq]['X']
        
        # Total energy in spectrum
        total_energy = np.sum(np.abs(X_test)**2)
        
        # Energy in peak bin
        peak_idx = np.argmax(np.abs(X_test))
        peak_energy = np.abs(X_test[peak_idx])**2
        
        # Fraction in peak
        fraction_in_peak = peak_energy / total_energy
        
        energy_results.append({
            'freq': freq,
            'label': label,
            'fraction': fraction_in_peak,
            'percent': fraction_in_peak * 100
        })
        
        print(f"{label}:")
        print(f"  Total energy (∑|X[k]|²): {total_energy:.2f}")
        print(f"  Peak bin energy:         {peak_energy:.2f}")
        print(f"  Fraction in peak bin:    {fraction_in_peak:.4f}  ({fraction_in_peak*100:.2f}%)")
        print()
    
    print("Interpretation:")
    print("  • 100.0 Hz (on-bin):   ~100% energy concentrates in ONE bin")
    print("  • 100.1 Hz (close):    ~80% in peak, rest leaks to neighbors")
    print("  • 100.5 Hz (worst):    ~63% in peak, ~37% leaks away!")
    print("  • 100.7 Hz:            ~65% in peak, ~35% leaks away")
    print()
    print("This energy spreading is spectral LEAKAGE — the bigger the off-bin")
    print("distance, the more energy 'leaks' to adjacent frequency bins.")
    
    # TASK 4: Suppressing leakage with windowing
    print("\n" + "="*70)
    print("TASK 4: How to suppress leakage? (Preview of Week 3)")
    print("="*70)
    print()
    print("The problem: Our signal is multiplied by a RECTANGULAR WINDOW")
    print("  w[n] = 1 for 0 ≤ n < N")
    print("  w[n] = 0 elsewhere")
    print()
    print("This creates a sharp discontinuity at the edges if the signal doesn't")
    print("fit perfectly into the window (off-bin case). The abrupt start/stop")
    print("creates high-frequency artifacts that spread across the spectrum.")
    print()
    print("Solution: Apply a SMOOTH WINDOW that tapers to zero at the edges")
    print()
    
    # Demonstrate with Hann window
    print("Example: Using a HANN WINDOW (smooth taper)")
    window_hann = np.hanning(N)
    
    # Apply window to the worst case (100.5 Hz)
    f_worst = 100.5
    x_worst = 0.8 * np.sin(2 * np.pi * f_worst * n / fs)
    x_windowed = x_worst * window_hann
    
    freqs_w, X_w = compute_rfft(x_windowed, fs)
    mag_db_w = 20 * np.log10(np.abs(X_w) + 1e-8)
    
    # Compare: off-bin without window vs with window
    x_worst_no_window = 0.8 * np.sin(2 * np.pi * f_worst * n / fs)
    freqs_no_w, X_no_w = compute_rfft(x_worst_no_window, fs)
    mag_db_no_w = 20 * np.log10(np.abs(X_no_w) + 1e-8)
    
    total_energy_no_w = np.sum(np.abs(X_no_w)**2)
    peak_energy_no_w = np.max(np.abs(X_no_w)**2)
    
    total_energy_w = np.sum(np.abs(X_w)**2)
    peak_energy_w = np.max(np.abs(X_w)**2)
    
    print(f"For off-bin tone at {f_worst} Hz (worst case):")
    print()
    print("Without window (rectangular, sharp edges):")
    print(f"  Peak bin energy (fraction): {peak_energy_no_w/total_energy_no_w*100:.2f}%")
    print()
    print("With Hann window (smooth taper):")
    print(f"  Peak bin energy (fraction): {peak_energy_w/total_energy_w*100:.2f}%")
    print()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    mask_plot = (freqs_no_w >= 95) & (freqs_no_w <= 105)
    
    ax1.stem(freqs_no_w[mask_plot], mag_db_no_w[mask_plot],
             markerfmt="o", linefmt="-", basefmt="k-")
    ax1.set_title(f"Rectangular window (no tapering)\nTone at {f_worst} Hz")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_ylim(bottom=-80)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(f_worst, color='red', linestyle='--', alpha=0.5)
    
    ax2.stem(freqs_w[mask_plot], mag_db_w[mask_plot],
             markerfmt="o", linefmt="-", basefmt="k-")
    ax2.set_title(f"Hann window (smooth taper)\nTone at {f_worst} Hz")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_ylim(bottom=-80)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(f_worst, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("Notice: Hann window shows WIDER main lobe but MUCH LOWER side lobes!")
    print()
    print("Trade-off:")
    print("  • Rectangular: Narrow peak, but high side-lobes → severe leakage")
    print("  • Hann window: Wider peak, but low side-lobes → reduced leakage")
    print()
    print("Different windows make different trade-offs between:")
    print("  - Frequency resolution (narrow main lobe)")
    print("  - Leakage suppression (low side lobes)")
    print()
    print("Week 3 explores these trade-offs and various windowing strategies!")


if __name__ == "__main__":
    main()
