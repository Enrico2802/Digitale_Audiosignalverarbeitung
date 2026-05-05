"""Task 1 — Probing a FIR filter with simple inputs.

A 5-tap box average is already set up and applied to a speech recording.
Run the scaffold first: it saves a filtered WAV file to generated/ — open it
and listen. Notice how the speech sounds different from the original.
Your goal in this task is to understand why.

Parts 1-3 probe the same filter using simple inputs:
a constant (DC) signal and single-frequency sinusoids.

The helper `apply_fir` below implements the FIR formula directly:
    y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ...
Use it for the analysis parts so you can see exactly what is happening.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_DIR = ROOT.parent / "assets" / "audio"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

FS = 48000
DURATION = 0.5   # seconds — short enough to work with step by step

# 5-tap box average: equal weight to the current sample and the 4 before it
B = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


def apply_fir(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply FIR filter coefficients b to signal x using the formula directly.

    For each output sample n:
        y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[P-1]*x[n-P+1]
    Samples before the start of x are treated as zero.
    """
    P = len(b)
    y = np.zeros(len(x))
    for n in range(len(x)):
        for p in range(P):
            if n - p >= 0:
                y[n] += b[p] * x[n - p]
    return y


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Scaffold — run this first, listen, then read on
    # -------------------------------------------------------------------------
    # np.convolve is Python's built-in shorthand for applying an FIR filter to
    # a long signal. We use it here because it is fast on audio-length signals.
    # The Parts below use apply_fir() instead so you can see the formula at work.
    x_speech, fs = sf.read(AUDIO_DIR / "speech.wav")
    assert fs == FS, f"Expected {FS} Hz, got {fs} Hz"
    if x_speech.ndim == 2:
        x_speech = x_speech.mean(axis=1)
    x_speech = x_speech[: int(3.0 * fs)]

    y_speech = np.convolve(x_speech, B, mode="same")
    sf.write(OUTPUT_DIR / "scaffold_filtered_speech.wav", y_speech, fs)
    print("Saved scaffold_filtered_speech.wav")
    print(f"  Filter coefficients: B = {B}")
    print(f"  Sum of coefficients: sum(B) = {B.sum():.4f}")
    print()
    print("Listen to both files. What do you notice about the filtered version?")

    # -------------------------------------------------------------------------
    # Part 1 — Test with a constant (DC) signal
    # -------------------------------------------------------------------------
    # A constant signal: x[n] = C for all n.
    # Applying the FIR formula:
    #   y[n] = b[0]*C + b[1]*C + ... + b[P-1]*C = C * (b[0]+b[1]+...+b[P-1])
    # The output is also constant — scaled by sum(b).
    # This is the filter's gain at DC (zero frequency, Ω = 0).

    # TODO:
    #   1a. Before running: what do you predict y[n] will be if x[n] = 1.0?
    #       Calculate sum(B) from the definition of B above.
    #
    #   1b. Generate a constant signal and apply the filter:
    t = np.arange(int(DURATION * FS)) / FS
    x_dc = np.ones(len(t))
    y_dc = apply_fir(B, x_dc)
    #
    #       Print a few output samples (avoid the first P=5, which are warming up):
    print("Part 1 — DC probe:")
    print(f"  y_dc[5]  = {y_dc[5]:.6f}")
    print(f"  y_dc[50] = {y_dc[50]:.6f}")
    print(f"  y_dc[500]= {y_dc[500]:.6f}")
    print()
    #
    #   1c. Does the output match your prediction?
    #
    #   1d. TODO: repeat with x_dc = np.full(len(t), 2.0).
    #       What is y[n] now? Does doubling the input double the output?
    #       (This is the linearity property of LTI systems from the lecture.)

    # -------------------------------------------------------------------------
    # Part 2 — Test with two sinusoids
    # -------------------------------------------------------------------------
    # From the lecture: for an LTI system, a sinusoid in gives a sinusoid out
    # at the same frequency — only the amplitude (and possibly phase) may change.

    F_LOW = 200.0    # Hz
    F_HIGH = 4000.0  # Hz

    x_low  = np.sin(2 * np.pi * F_LOW  * t)
    x_high = np.sin(2 * np.pi * F_HIGH * t)

    # TODO 2a — filter both sinusoids and print the gain at each frequency.
    #   After the first P=5 samples the filter has "warmed up"
    #   (because the input signal has arrived at all delays).
    #   Hint: apply_fir(B, x_low), np.max(np.abs(y[5:]))

    y_low  = ...   # TODO
    y_high = ...   # TODO

    gain_low  = ...   # TODO
    gain_high = ...   # TODO

    print("Part 2 — Sinusoid probe:")
    print(f"  Gain at {F_LOW:.0f} Hz  = {gain_low:.4f}")
    print(f"  Gain at {F_HIGH:.0f} Hz = {gain_high:.4f}")

    # TODO 2b — which frequency is more attenuated?
    #   Is this consistent with what you heard in the scaffold?

    # TODO 2c — plot input and output for both frequencies side by side.
    #   Use only the first 200 samples so the waveform is readable.
    #   Can you see the amplitude change? Can you spot a small time shift?
    #   Hint: plt.subplots(2, 2, sharex=True),   ax.plot(x_low[:200])

    # -------------------------------------------------------------------------
    # Part 3 — Verify: sinusoid in → sinusoid out
    # -------------------------------------------------------------------------
    # The lecture claims the output is always a sinusoid at the same frequency.
    # Verify this by finding the dominant frequency in the output with the DFT.

    # TODO 3a — filter x_low and find the dominant frequency in the output.
    #   Skip the first 5 samples (warm-up) before computing the DFT.
    #   Hint: np.fft.rfft(y_low[5:]), np.fft.rfftfreq(N, d=1/FS), np.argmax

    dominant_freq_low = ...   # TODO
    print(f"Part 3 — dominant frequency in filtered 200 Hz output: {dominant_freq_low:.1f} Hz")

    # TODO 3b — repeat for x_high (4 000 Hz). Is the dominant frequency still 4 000 Hz?

    # TODO 3c — write one sentence: why must this always be true for an LTI system?

    # -------------------------------------------------------------------------
    # Reflection questions (written answers — no code needed)
    # -------------------------------------------------------------------------
    # 1. The DC gain of the 5-tap average is sum(B) = 1.0. What does this mean
    #    physically? Does the filter change the loudness of a very slow signal?
    #
    # 2. The 4000 Hz sinusoid is more attenuated than the 200 Hz one.
    #    What does this tell you about the shape of the magnitude response?
    #    (Low-pass, high-pass, or band-pass?)
    #
    # 3. The output is slightly shifted in time relative to the input.
    #    Which sinusoid (200 Hz or 4000 Hz) shows a larger visible time shift?
    #    Why? (Hint: phase shift = Ω × delay in samples.)


if __name__ == "__main__":
    main()
