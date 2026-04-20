from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dsp_utils import (  # noqa: E402
    compute_rfft,
    plot_waveform_and_spectrum,
    plot_phase_spectrum,
    dominant_bin_info,
)


def main():
    fs = 48000
    duration = 1.0
    n = np.arange(int(duration * fs))

    x_single = 0.8 * np.sin(2 * np.pi * 440 * n / fs)
    x_mix = 0.8 * np.sin(2 * np.pi * 440 * n / fs) + 0.5 * np.sin(2 * np.pi * 880 * n / fs)

    # --- Inspect the raw FFT output ---
    # compute_rfft returns N//2 + 1 complex coefficients, one per frequency bin.
    # Bin k corresponds to frequency k * fs / N Hz.
    # Each coefficient X[k] is a complex number encoding magnitude and phase.

    freqs, X = compute_rfft(x_single, fs)
    print(f"Signal length N       : {len(x_single)}")
    print(f"Number of frequency bins : {len(X)}")
    print(f"Frequency resolution  : {fs / len(x_single):.2f} Hz/bin")
    print(f"X[440] — bin at 440 Hz: {X[440]:.4f}  (complex number)")
    print(f"  magnitude |X[440]|  : {np.abs(X[440]):.2f}")
    print(f"  magnitude in dB     : {20 * np.log10(np.abs(X[440])):.1f} dB")
    print(f"  phase angle(X[440]) : {np.angle(X[440]):.4f} rad")
    print()

    # --- Euler's formula: a complex number in polar form ---
    # Any complex number can be written as:
    #   X[k] = |X[k]| * e^(j * angle(X[k]))
    # because Euler's formula gives us:
    #   e^(j*theta) = cos(theta) + j*sin(theta)
    # so magnitude sets the length, angle sets the direction in the complex plane.
    # Verify numerically that both representations give the same value:
    X_k = X[440]
    X_polar = np.abs(X_k) * np.exp(1j * np.angle(X_k))
    print(f"X[440] direct        : {X_k:.2f}")
    print(f"X[440] via Euler     : {X_polar:.2f}")
    print(f"Match: {np.isclose(X_k, X_polar)}")
    print()

    plot_waveform_and_spectrum(
        x_single,
        fs,
        wave_title="Waveform: single 440 Hz tone",
        spec_title="Magnitude spectrum: single 440 Hz tone",
        max_freq=2000,
    )

    plot_phase_spectrum(
        x_single,
        fs,
        title="Phase spectrum: single 440 Hz tone",
        max_freq=2000,
    )

    info_single = dominant_bin_info(x_single, fs)
    print("Single-tone dominant bin:")
    print(info_single)

    plot_waveform_and_spectrum(
        x_mix,
        fs,
        wave_title="Waveform: 440 Hz + 880 Hz",
        spec_title="Magnitude spectrum: 440 Hz + 880 Hz",
        max_freq=2000,
    )

    plot_phase_spectrum(
        x_mix,
        fs,
        title="Phase spectrum: 440 Hz + 880 Hz",
        max_freq=2000,
    )

    # TODO:
    # 1. Identify the main peaks in the magnitude spectrum.
    # 2. Why is the phase of bin k=440 (frequency 440 Hz) equal to -1.57 rad?
    # 3. Change one frequency and compare the result.
    # 4. Change the phase of one component. What changes in time domain?
    # 5. What changes in the magnitude spectrum? What changes in the phase spectrum?

    # --- SOLUTION: Exploring phase changes ---
    print("\n" + "="*70)
    print("TASK 4: What happens when we change the phase of one component?")
    print("="*70)
    print()

    # The original mixed signal has two sine components at default phase (0):
    # x_mix = 0.8*sin(2π*440*n/fs) + 0.5*sin(2π*880*n/fs)
    
    # Now let's add a phase shift to the 880 Hz component:
    phase_shift_880 = np.pi / 2  # 90 degrees
    x_mix_phase_shifted = (
        0.8 * np.sin(2 * np.pi * 440 * n / fs) +
        0.5 * np.sin(2 * np.pi * 880 * n / fs + phase_shift_880)
    )
    
    print(f"Original signal:     x = 0.8·sin(2π·440·t) + 0.5·sin(2π·880·t)")
    print(f"Phase-shifted signal: x = 0.8·sin(2π·440·t) + 0.5·sin(2π·880·t + π/2)")
    print()
    print("Observation: The magnitude spectrum should be IDENTICAL for both signals")
    print("because we only changed the PHASE, not the magnitude or frequency.")
    print("However, the TIME DOMAIN waveform will look DIFFERENT due to the")
    print("altered phase relationship between the two components.")
    print()
    
    plot_waveform_and_spectrum(
        x_mix_phase_shifted,
        fs,
        wave_title="Waveform: 440 Hz + 880 Hz (with 90° phase shift on 880 Hz)",
        spec_title="Magnitude spectrum: Same as original (phase doesn't affect magnitude)",
        max_freq=2000,
    )
    
    plot_phase_spectrum(
        x_mix_phase_shifted,
        fs,
        title="Phase spectrum: 440 Hz + 880 Hz (with phase shift on 880 Hz)",
        max_freq=2000,
    )
    
    # Demonstrate visually by showing both waveforms together
    print("\nComparison of the two time-domain waveforms:")
    print("(First 0.01 seconds - 480 samples at 48 kHz)")
    display_samples = int(0.01 * fs)
    time_display = np.arange(display_samples) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(time_display, x_mix[:display_samples], linewidth=1.5, label="Original")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Original: 440 Hz + 880 Hz (both at phase 0)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time_display, x_mix_phase_shifted[:display_samples], 
                 linewidth=1.5, color='orange', label="880 Hz shifted by +90°")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Phase-shifted: 440 Hz + 880 Hz (880 Hz shifted by +90°)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(time_display, x_mix[:display_samples], linewidth=1.5, 
                 label="Original", alpha=0.7)
    axes[2].plot(time_display, x_mix_phase_shifted[:display_samples], linewidth=1.5,
                 label="Phase-shifted", alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Overlay comparison")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey insights:")
    print("• Changing the phase of one component DOES NOT change the magnitude spectrum.")
    print("• It ONLY affects the PHASE spectrum and the TIME DOMAIN waveform shape.")
    print("• The magnitude spectrum remains the same because |a·sin(ωt +φ)| = |a·sin(ωt)|")
    print("• The time-domain waveform shows a different interference pattern due to")
    print("  the changed phase relationship between the 440 Hz and 880 Hz components.")
    print("• This is why phase information is important in signal processing!")


if __name__ == "__main__":
    main()