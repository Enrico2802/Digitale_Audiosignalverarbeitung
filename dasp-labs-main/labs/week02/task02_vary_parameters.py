from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


def plot_group(signals, labels, fs, title, output_path):
    t = np.arange(len(signals[0])) / fs

    plt.figure(figsize=(10, 4))
    for x, label in zip(signals, labels):
        plt.plot(t, x, label=label)
    plt.xlim(0, 0.01)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def main():
    fs = 48000
    duration = 0.03

    output_dir = ROOT / "week02" / "generated" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    n = np.arange(int(duration * fs))

    # --- Amplitude ---
    amp_signals = [
        0.3 * np.sin(2 * np.pi * 440 * n / fs),
        0.6 * np.sin(2 * np.pi * 440 * n / fs),
        0.9 * np.sin(2 * np.pi * 440 * n / fs),
    ]
    plot_group(amp_signals, ["A=0.3", "A=0.6", "A=0.9"], fs,
               "Effect of amplitude", output_dir / "week02_amplitude_comparison.png")

    # --- Frequency ---
    freq_signals = [
        0.8 * np.sin(2 * np.pi * 220 * n / fs),
        0.8 * np.sin(2 * np.pi * 440 * n / fs),
        0.8 * np.sin(2 * np.pi * 880 * n / fs),
    ]
    plot_group(freq_signals, ["220 Hz", "440 Hz", "880 Hz"], fs,
               "Effect of frequency", output_dir / "week02_frequency_comparison.png")

    # --- Phase ---
    phase_signals = [
        0.8 * np.sin(2 * np.pi * 440 * n / fs),
        0.8 * np.sin(2 * np.pi * 440 * n / fs + np.pi / 2),
        0.8 * np.sin(2 * np.pi * 440 * n / fs + np.pi),
    ]
    plot_group(phase_signals, ["phase=0", "phase=pi/2", "phase=pi"], fs,
               "Effect of phase", output_dir / "week02_phase_comparison.png")

    # TODO:
    # 1. Describe what changes in each comparison.
    # 2. Which parameter is most obvious in the waveform?
    # 3. Which parameter do you expect to be most obvious to a listener? Why?


if __name__ == "__main__":
    main()
