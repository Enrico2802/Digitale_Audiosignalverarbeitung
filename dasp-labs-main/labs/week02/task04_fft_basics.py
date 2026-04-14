from pathlib import Path
import sys
import numpy as np

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


if __name__ == "__main__":
    main()