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

    # TODO:
    # 1. Change f_off to 100.5 Hz (halfway between bins). Does the leakage get worse or better?
    # 2. Change f_off to 100.1 Hz (almost on-bin). How does the spread change?
    # 3. What fraction of the total signal energy remains in the peak bin for each case?
    #    (Hint: sum all |X[k]|^2 and compare to |X[peak]|^2.)
    # 4. What would you need to do to the signal to suppress the leakage?
    #    (This is the question week 3 answers.)


if __name__ == "__main__":
    main()
