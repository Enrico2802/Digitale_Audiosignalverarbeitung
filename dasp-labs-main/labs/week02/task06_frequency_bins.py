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

    # TODO:
    # 1. Compare Δf for the two durations. How are they related to N?
    # 2. What happens when the tone frequency is not exactly at a bin centre?
    # 3. Which signal length gives finer frequency resolution?
    # 4. What duration would you need to resolve two tones 2 Hz apart?


if __name__ == "__main__":
    main()
