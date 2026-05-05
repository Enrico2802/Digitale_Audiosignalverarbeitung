"""Task 2 — Measuring and shaping the magnitude response.

From the lecture: probing a system with sinusoids at many frequencies gives
the frequency response H(f). Here you do exactly that — in a for loop.

Parts 4-5: measure gain at many frequencies and plot the result.
            This is the empirical magnitude response, built from experiment alone.

Part 6:     change the coefficients, re-listen, and explain what changed.

Extension:  apply a custom filter to a music file and describe the result.
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

# Same 5-tap box average as Task 1 — start here, then change it in Part 6
B = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


def apply_fir(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply FIR filter coefficients b to signal x using the formula directly."""
    P = len(b)
    y = np.zeros(len(x))
    for n in range(len(x)):
        for p in range(P):
            if n - p >= 0:
                y[n] += b[p] * x[n - p]
    return y


def analytical_magnitude(b: np.ndarray, freqs_hz: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute exact |H(f)| using the transfer function formula from the lecture.

    For each frequency f:
        Ω = 2π f / fs
        H = b[0]*exp(0) + b[1]*exp(-jΩ) + b[2]*exp(-j2Ω) + ...
        |H| = abs(H)
    """
    result = np.zeros(len(freqs_hz))
    for i, f in enumerate(freqs_hz):
        omega = 2 * np.pi * f / fs
        H = sum(b[p] * np.exp(-1j * omega * p) for p in range(len(b)))
        result[i] = abs(H)
    return result


def measure_gain(b: np.ndarray, freq_hz: float, fs: int = FS) -> float:
    """Probe the filter at one frequency and return the steady-state gain.

    Sends a unit sinusoid at freq_hz through the filter and returns
    peak(output) / peak(input) after the warm-up period.
    With a 100 ms window there are many complete cycles at every probe
    frequency, so the window reliably contains a peak.
    """
    duration = 0.1   # 100 ms — long enough to contain many cycles
    t = np.arange(int(duration * fs)) / fs
    x = np.sin(2 * np.pi * freq_hz * t)
    y = apply_fir(b, x)
    P = len(b)
    return np.max(np.abs(y[P:])) / np.max(np.abs(x[P:]))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Part 4 — Probe at many frequencies
    # -------------------------------------------------------------------------
    # We repeat the sinusoid-probing experiment from Part 2, but now at many
    # frequencies in a loop. Each iteration is one probe experiment.

    TEST_FREQS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 9600, 12000, 18000, 23900]

    print("Part 4 — probing the 5-tap box average:")
    gains = []
    for f in TEST_FREQS:
        g = measure_gain(B, f)
        gains.append(g)
        print(f"  {f:6.0f} Hz  →  gain = {g:.4f}")
    print()

    # TODO:
    #   4a. Before looking at the numbers: which frequencies do you expect to
    #       pass through with gain ≈ 1? Which do you expect to be attenuated?
    #       Write your prediction first, then compare with the printed output.
    #
    #   4b. At what frequency does the gain first drop below 0.5?
    #
    #   4c. What is the gain near the Nyquist frequency (23 900 Hz)?
    #       Is this consistent with the two-point average example from the lecture?
    #       (Recall: for any symmetric averaging filter, the gain at Nyquist = 0
    #       if the sum of alternating coefficients is zero.)

    # -------------------------------------------------------------------------
    # Part 5 — Plot the empirical magnitude response
    # -------------------------------------------------------------------------
    # The smooth line is the exact analytical formula from the lecture.
    # The dots are your measured probe values — they should fall on the curve.
    # Note: the response is NOT monotonically decreasing above ~9 600 Hz.
    # That bump is real (same sinc-sidelobe phenomenon as spectral leakage
    # in Week 3) — not a measurement error. The 5-tap box average is not a
    # clean low-pass filter above its first null.

    smooth_freqs = np.linspace(10, FS / 2 - 10, 2000)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smooth_freqs, analytical_magnitude(B, smooth_freqs),
            color="C0", linewidth=1.5, label="analytical formula")
    ax.plot(TEST_FREQS, gains, "o",
            color="C0", markersize=6, label="measured (probe)")
    ax.axhline(1.0, color="gray",   linestyle=":", linewidth=0.9, label="|H| = 1")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.9, label="|H| = 0.5  (−6 dB)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Gain |H|")
    ax.set_title("Empirical magnitude response — 5-tap box average")
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_magnitude_response.svg")
    plt.show()

    # TODO:
    #   5a. The dots should land on the analytical curve.
    #       What does this tell you about the relationship between
    #       sinusoidal probing and the transfer function formula?
    #
    #   5b. Describe the shape in one sentence. Is this a low-pass filter?
    #       What happens above 9 600 Hz? Why does the response not stay at zero
    #       after the first null?

    # -------------------------------------------------------------------------
    # Part 6 — Change the coefficients and re-listen
    # -------------------------------------------------------------------------
    # Now you have a tool: measure_gain tells you the shape of any FIR filter.
    # Use it to explore three different coefficient sets.
    # For each one: predict the shape first, then probe, then listen.

    FILTERS = {
        "2-tap average": np.array([0.5, 0.5]),
        "1-tap delay":   np.array([0.0, 1.0]),
        "11-tap average": np.full(11, 1.0 / 11),
    }

    x_speech, fs = sf.read(AUDIO_DIR / "speech.wav")
    if x_speech.ndim == 2:
        x_speech = x_speech.mean(axis=1)
    x_speech = x_speech[: int(3.0 * fs)]

    print("Part 6 — changing coefficients:")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Smooth analytical curves for all filters (probe markers on top)
    all_filters = {"5-tap average (original)": B, **FILTERS}
    for color, (name, b_new) in zip(["C0", "C1", "C2", "C3"], all_filters.items()):
        ax.plot(smooth_freqs, analytical_magnitude(b_new, smooth_freqs),
                color=color, linewidth=1.5, label=name)
        g_new = [measure_gain(b_new, f) for f in TEST_FREQS]
        ax.plot(TEST_FREQS, g_new, "o", color=color, markersize=5)

        # Apply to speech and save (skip original)
        if name != "5-tap average (original)":
            y = np.convolve(x_speech, b_new, mode="same")
            safe_name = name.replace(" ", "_").replace("-", "")
            sf.write(OUTPUT_DIR / f"task2_{safe_name}.wav", y, fs)
            print(f"  Saved task2_{safe_name}.wav  (b = {b_new})")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Gain |H|")
    ax.set_title("Part 6: magnitude responses — different coefficient sets")
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_coefficient_comparison.svg")
    plt.show()

    # TODO:
    #   6a. Listen to each saved WAV file and describe what you hear.
    #       Match each sound to its magnitude response shape in the plot.
    #
    #   6b. The "1-tap delay" has b = [0, 1], meaning y[n] = x[n-1].
    #       What is the gain at every frequency? Does the filtered speech
    #       sound different from the original? Why or why not?
    #
    #   6c. Compare the 2-tap and 11-tap averages. Which sounds more muffled?
    #       How does the magnitude response explain the difference?
    #
    #   6d. Design your own filter. Try b = [1, -1] (a difference filter).
    #       Predict the shape before probing: at DC, x[n]-x[n-1] = 0 for a
    #       constant signal, so the gain at Ω=0 should be 0. Is that what you see?
    #       Listen — does it sound like a high-pass filter?

    b_custom = ...   # TODO: try np.array([1.0, -1.0])
    # Hint: measure_gain / np.convolve / sf.write — same pattern as above

    # -------------------------------------------------------------------------
    # Extension — Apply your favourite filter to a music file
    # -------------------------------------------------------------------------
    # TODO:
    #   Choose one coefficient set from Part 6 (or design your own).
    #   Apply it to music.wav and listen. Describe in 2–3 sentences:
    #     - What frequency content changed?
    #     - Does the music sound better, worse, or just different?
    #     - Does your observation match the magnitude response plot?

    # x_music, fs_m = sf.read(AUDIO_DIR / "music.wav")
    # if x_music.ndim == 2:
    #     x_music = x_music.mean(axis=1)
    # x_music = x_music[: int(3.0 * fs_m)]
    # b_choice = ...
    # y_music = np.convolve(x_music, b_choice, mode="same")
    # sf.write(OUTPUT_DIR / "task2_music_filtered.wav", y_music, fs_m)

    # -------------------------------------------------------------------------
    # Reflection questions
    # -------------------------------------------------------------------------
    # 1. The 1-tap delay [0, 1] has |H| = 1 at every frequency.
    #    Yet y[n] = x[n-1], not x[n]. What does that tell you about
    #    the difference between gain and time delay?
    #
    # 2. Look at the Part 5 plot: the dots (sinusoidal probing) and the line
    #    (analytical formula H(e^{jΩ}) = Σ b[p] e^{-jΩp}) are drawn on the
    #    same axes. Do they agree? Why must they — what does each method
    #    actually compute?
    #
    # 3. In Part 6 you chose b and then heard the result. Filter design usually
    #    runs the other way: you know the sound you want, and you have to find b.
    #    Given only the magnitude response plots from Part 6, could you have
    #    predicted which filter would sound most muffled before listening?


if __name__ == "__main__":
    main()
