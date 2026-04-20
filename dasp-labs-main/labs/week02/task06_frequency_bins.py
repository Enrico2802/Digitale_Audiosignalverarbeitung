from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dsp_utils import compute_rfft  # noqa: E402

# compute_rfft(x, fs):
#     X = np.fft.rfft(x)
#     freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
#     return freqs, X


def inspect_case(frequency: float, duration: float, fs: int):
    n = np.arange(int(duration * fs))
    x = 0.8 * np.sin(2 * np.pi * frequency * n / fs)

    freqs, X = compute_rfft(x, fs)
    N = len(x)
    df = fs / N
    peak_idx = np.abs(X).argmax()
    peak_freq = freqs[peak_idx]

    print("=" * 60)
    print(f"Target tone frequency: {frequency:.2f} Hz")
    print(f"Signal length N: {N}")
    print(f"Sampling rate fs: {fs}")
    print(f"Frequency spacing Δf = fs/N = {df:.3f} Hz")
    print(f"Peak bin index: {peak_idx}")
    print(f"Peak bin frequency: {peak_freq:.3f} Hz")
    print(f"Frequency error: {abs(peak_freq - frequency):.3f} Hz")


def main():
    fs = 48000

    inspect_case(frequency=440.0, duration=1.0, fs=fs)
    inspect_case(frequency=440.0, duration=0.25, fs=fs)
    inspect_case(frequency=445.0, duration=1.0, fs=fs)

    # SOLUTION: Analyzing frequency resolution
    print("\n" + "="*70)
    print("TASK 1: Compare Δf for two durations. How are they related to N?")
    print("="*70)
    print()
    
    # Case 1: 1 second duration
    N1 = int(1.0 * fs)
    df1 = fs / N1
    print(f"Duration = 1.0 s:")
    print(f"  N = {N1}")
    print(f"  Δf = fs / N = {fs} / {N1} = {df1:.4f} Hz")
    
    # Case 2: 0.25 second duration
    N2 = int(0.25 * fs)
    df2 = fs / N2
    print(f"\nDuration = 0.25 s:")
    print(f"  N = {N2}")
    print(f"  Δf = fs / N = {fs} / {N2} = {df2:.4f} Hz")
    
    print(f"\nRelationship:")
    print(f"  Δf₁ / Δf₂ = (fs/N₁) / (fs/N₂) = N₂ / N₁ = {N2 / N1:.2f}")
    print(f"  When duration is reduced by 4× → frequency spacing increases by 4×")
    print(f"  Key insight: Δf = 1 / duration")
    print(f"    • 1.0 s duration → Δf = 1 Hz")
    print(f"    • 0.25 s duration → Δf = 4 Hz")
    print()
    print("  Shorter signals have WORSE frequency resolution!")
    
    print("\n" + "="*70)
    print("TASK 2: What happens when tone frequency is NOT at bin centre?")
    print("="*70)
    print()
    
    print("Comparing 440 Hz (exactly on a bin) vs 445 Hz (between bins):")
    print()
    
    # 440 Hz: should align with bins
    n_440 = np.arange(int(1.0 * fs))
    x_440 = 0.8 * np.sin(2 * np.pi * 440 * n_440 / fs)
    freqs_440, X_440 = compute_rfft(x_440, fs)
    magnitude_440 = np.abs(X_440)
    peak_idx_440 = np.argmax(magnitude_440)
    peak_mag_440 = magnitude_440[peak_idx_440]
    
    # 445 Hz: off-frequency
    x_445 = 0.8 * np.sin(2 * np.pi * 445 * n_440 / fs)
    freqs_445, X_445 = compute_rfft(x_445, fs)
    magnitude_445 = np.abs(X_445)
    peak_idx_445 = np.argmax(magnitude_445)
    peak_mag_445 = magnitude_445[peak_idx_445]
    
    print(f"  Target = 440 Hz (at bin centre):")
    print(f"    Peak magnitude: {peak_mag_440:.4f}")
    print(f"    Peak at bin {peak_idx_440} (frequency {freqs_440[peak_idx_440]:.2f} Hz)")
    print(f"    Energy is concentrated in ONE bin")
    print()
    print(f"  Target = 445 Hz (between bins 440 and 445):")
    print(f"    Peak magnitude: {peak_mag_445:.4f}")
    print(f"    Peak at bin {peak_idx_445} (frequency {freqs_445[peak_idx_445]:.2f} Hz)")
    print(f"    Energy is SPREAD across multiple bins (spectral leakage)")
    print()
    print("  Effect: Off-frequency tones appear WEAKER and spread across bins.")
    print("          This is called SPECTRAL LEAKAGE — energy leaks from the")
    print("          true frequency to adjacent bins, making peak appear shorter.")
    
    # Show nearby bins
    print(f"\n  Magnitude in nearby bins around the 445 Hz tone:")
    for i in range(max(0, peak_idx_445-3), min(len(X_445), peak_idx_445+4)):
        print(f"    Bin {i} ({freqs_445[i]:>7.1f} Hz): {magnitude_445[i]:>8.4f}")
    
    print("\n" + "="*70)
    print("TASK 3: Which signal length gives finer frequency resolution?")
    print("="*70)
    print()
    print("Frequency resolution: Δf = fs / N = 1 / duration")
    print()
    print("  For FINER (smaller) frequency resolution → LONGER signal needed")
    print()
    print(f"  • 0.25 s duration → Δf = {1/0.25:.1f} Hz     (coarse resolution)")
    print(f"  • 1.0 s duration  → Δf = {1/1.0:.1f} Hz     (finer resolution)")
    print(f"  • 10 s duration   → Δf = {1/10.0:.2f} Hz    (very fine resolution)")
    print()
    print("  In this task: 1.0 s gives finer resolution than 0.25 s")
    print("  Trade-off: Longer signals → better frequency resolution")
    print("             BUT more computation and must re-record signal")
    
    print("\n" + "="*70)
    print("TASK 4: What duration needed to resolve two tones 2 Hz apart?")
    print("="*70)
    print()
    print("To distinguish two frequencies f₁ and f₂:")
    print("  Need: Δf < |f₂ - f₁|")
    print()
    print("For tones 2 Hz apart:")
    print("  Need: Δf < 2 Hz")
    print("       1/duration < 2")
    print("       duration > 1/2 = 0.5 s")
    print()
    print("Minimum duration: 0.5 seconds")
    print()
    
    # Demonstrate: 0.4 s vs 0.5 s vs 1.0 s with two tones
    print("Demonstration: Two tones at 440 Hz and 442 Hz")
    print()
    
    test_durations = [0.4, 0.5, 1.0]
    for dur in test_durations:
        N = int(dur * fs)
        delta_f = fs / N
        n = np.arange(N)
        x_two_tones = (0.8 * np.sin(2 * np.pi * 440 * n / fs) + 
                       0.5 * np.sin(2 * np.pi * 442 * n / fs))
        freqs_2t, X_2t = compute_rfft(x_two_tones, fs)
        magnitude_2t = np.abs(X_2t)
        
        # Find peaks in the 430-450 Hz range
        freq_range = (freqs_2t >= 430) & (freqs_2t <= 450)
        peaks_in_range = magnitude_2t[freq_range]
        freqs_in_range = freqs_2t[freq_range]
        
        print(f"  Duration = {dur} s  →  Δf = {delta_f:.3f} Hz")
        if delta_f < 2.0:
            print(f"    ✓ Can resolve 2 Hz separation (Δf < 2 Hz)")
        else:
            print(f"    ✗ Cannot resolve 2 Hz separation (Δf ≥ 2 Hz)")
        print()


if __name__ == "__main__":
    main()
