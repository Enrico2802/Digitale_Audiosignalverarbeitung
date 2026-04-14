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
    n = np.arange(int(duration * fs))

    x1 = 0.7 * np.sin(2 * np.pi * 440 * n / fs)
    x2 = 0.4 * np.sin(2 * np.pi * 880 * n / fs)
    x_sum = x1 + x2

    if np.max(np.abs(x_sum)) > 1.0:
        print("Warning: sum clips — reduce component amplitudes before saving.")

    # Plot first 10 ms
    n_plot = int(0.01 * fs)
    t = n[:n_plot] / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, x1[:n_plot], label="440 Hz")
    plt.plot(t, x2[:n_plot], label="880 Hz")
    plt.plot(t, x_sum[:n_plot], label="sum", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Superposition of two sinusoids")
    plt.legend()
    plt.tight_layout()

    output_dir = ROOT / "week02" / "generated"
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)

    output_png = output_dir / "plots" / "week02_superposition.png"
    plt.savefig(output_png, dpi=150)
    plt.show()
    print(f"Saved: {output_png}")

    output_wav = output_dir / "audio" / "week02_two_tone_mix.wav"
    save_audio(str(output_wav), x_sum, fs)
    print(f"Saved: {output_wav}")

    # TODO:
    # 1. Change the phase of one component. What changes in the waveform? What about the sound?
    # 2. Set both components to the same frequency but opposite phase (phi=0 and phi=pi).
    #    What do you expect? What do you observe?
    # 3. Add a third sinusoid with a different frequency and amplitude.
    #    Make sure the sum stays within [-1, 1] before normalising — check np.max(np.abs(x_sum)).


if __name__ == "__main__":
    main()
