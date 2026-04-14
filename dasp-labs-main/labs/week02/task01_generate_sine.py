from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from audio_utils import save_audio  # noqa: E402


def main():
    fs = 48000
    duration = 1.0
    frequency = 440.0
    amplitude = 0.8
    phase = 0.0

    n = np.arange(int(duration * fs))
    x = amplitude * np.sin(2 * np.pi * frequency * n / fs + phase)

    if np.max(np.abs(x)) > 1.0:
        print("Warning: signal clips — reduce amplitude or normalise before saving.")

    output_dir = ROOT / "week02" / "generated"
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    output_wav = output_dir / "audio" / f"week02_sine_{int(frequency)}hz.wav"
    save_audio(str(output_wav), x, fs)
    print(f"Saved: {output_wav}")

    # Plot the first 10 ms so the oscillation is visible
    n_plot = int(0.01 * fs)
    t = n[:n_plot] / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, x[:n_plot])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{int(frequency)} Hz sine wave")
    plt.tight_layout()

    output_png = output_dir / "plots" / f"week02_sine_{int(frequency)}hz_waveform.png"
    plt.savefig(output_png, dpi=150)
    plt.show()
    print(f"Saved: {output_png}")

    # TODO:
    # 1. Change frequency to 445 Hz, 220 Hz, and 880 Hz. Save each with a clear file name.
    # 2. Change amplitude to 0.3 and 1.0. What changes in the plot? What do you expect to hear?
    # 3. Change phase to np.pi / 2. What changes in the plot?
    # 4. Change duration to 0.5 s and 2.0 s.


if __name__ == "__main__":
    main()
