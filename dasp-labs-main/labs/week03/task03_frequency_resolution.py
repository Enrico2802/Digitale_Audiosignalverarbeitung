"""Task 3 — Frequency resolution and pitch estimation.

You will use frames of increasing length to see how resolution improves,
then use zero-padding to read the fundamental frequency precisely.
The final question asks you to explain the key distinction: zero-padding helps
you *read* a peak accurately, but it cannot substitute for a longer recording.

Compare your plots with reference/task03a_reference.png (Part 1) and
reference/task03b_reference.png (Part 2) to check your result.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

AUDIO_PATH = ROOT.parent / "assets" / "audio" / "guitar_e_string.wav"

FS_EXPECTED = 48000
ATTACK_SEC = 0.085

E2_HZ = 82.407  # reference pitch: open low E string


def main() -> None:
    x, fs = sf.read(AUDIO_PATH)
    assert fs == FS_EXPECTED, f"Expected {FS_EXPECTED} Hz, got {fs} Hz"

    if x.ndim == 2:
        x = x.mean(axis=1)

    start = int(ATTACK_SEC * fs)

    # --- Part 1: effect of frame length on resolution ---
    frame_lengths = [512, 1024, 2048, 4096, fs]  # last = 1 second
    xlim = (50, 500)

    fig, axes = plt.subplots(len(frame_lengths), 1,
                             figsize=(11, 2.6 * len(frame_lengths)),
                             sharey=True)

    print("Frame length vs. frequency resolution:")
    for ax, N in zip(axes, frame_lengths):
        segment = x[start:start + N]

        # ---------------------------------------------------------------------
        # TODO: Compute the windowed spectrum for this frame length N.
        #
        # Step 1: create a Hann window of length N using np.hanning(N)
        #
        # Step 2: apply the window: segment * w
        #
        # Step 3: compute the DFT: np.fft.rfft(segment * w)
        #
        # Step 4: compute the frequency axis: np.fft.rfftfreq(N, d=1.0/fs)
        #
        # Step 5: compute magnitude in dB and peak-normalise to 0 dB (same as Task 2):
        #         mag_db = 20 * np.log10(np.abs(X) + 1e-9)
        #         mag_db -= mag_db.max()
        #
        # Replace the lines below with your solution:
        w = None
        X = None
        freqs = None
        mag_db = None
        # ---------------------------------------------------------------------

        mask = (freqs >= xlim[0]) & (freqs <= xlim[1])
        ax.plot(freqs[mask], mag_db[mask], linewidth=1.4, color="tab:blue")
        ax.axvline(E2_HZ, color="tab:orange", linewidth=1.0,
                   linestyle="--", label=f"E2 = {E2_HZ} Hz")
        ax.set_xlim(*xlim)
        ax.set_ylim(-50, 5)
        ax.set_ylabel("dB")
        ax.set_title(f"N = {N}  →  Δf = {fs/N:.1f} Hz  ({N/fs*1000:.0f} ms)",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        if N == frame_lengths[0]:
            ax.legend(fontsize=9, loc="upper right")
        print(f"  N = {N:6d}  ({N/fs*1000:5.0f} ms)  Δf = {fs/N:.2f} Hz")

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Effect of frame length on resolution — guitar E string (Hann window)",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- Part 2: zero-padding to read the fundamental accurately ---
    N_long = fs  # 1 second
    segment = x[start:start + N_long]
    w = np.hanning(N_long)

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    colors = ["tab:blue", "tab:orange"]

    for color, pad_factor in zip(colors, [1, 8]):

        # ---------------------------------------------------------------------
        # TODO: Zero-pad the windowed segment and compute its spectrum.
        #
        # Step 1: compute N_pad = N_long * pad_factor
        #
        # Step 2: build the zero-padded signal:
        #         np.concatenate([segment * w, np.zeros(N_pad - N_long)])
        #
        # Step 3: compute the DFT of the padded signal: np.fft.rfft(x_pad)
        #
        # Step 4: compute the frequency axis: np.fft.rfftfreq(N_pad, d=1.0/fs)
        #
        # Step 5: compute magnitude in dB and peak-normalise to 0 dB:
        #         mag_pad = 20 * np.log10(np.abs(X_pad) + 1e-9)
        #         mag_pad -= mag_pad.max()
        #
        # Replace the lines below with your solution:
        N_pad = None
        x_pad = None
        X_pad = None
        freqs_pad = None
        mag_pad = None
        # ---------------------------------------------------------------------

        mask = (freqs_pad >= 60) & (freqs_pad <= 110)
        ax2.plot(freqs_pad[mask], mag_pad[mask], linewidth=1.5, color=color,
                 label=f"×{pad_factor} zero-padding  (Δf_display = {fs/N_pad:.3f} Hz)")

    ax2.axvline(E2_HZ, color="black", linewidth=1.0, linestyle="--",
                label=f"E2 reference = {E2_HZ} Hz")
    ax2.set_xlim(60, 110)
    ax2.set_ylim(-50, -20)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title(
        f"Zero-padding to locate the fundamental  (N = {N_long}, Hann window)",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print the estimated fundamental for reference
    N_pad = N_long * 8
    x_pad = np.concatenate([segment * w, np.zeros(N_pad - N_long)])
    X_pad = np.fft.rfft(x_pad)
    freqs_pad = np.fft.rfftfreq(N_pad, d=1.0 / fs)
    mag_pad = np.abs(X_pad)
    mask_f0 = (freqs_pad > 70) & (freqs_pad < 95)
    f0_est = freqs_pad[mask_f0][np.argmax(mag_pad[mask_f0])]
    cents_off = 1200 * np.log2(f0_est / E2_HZ)
    print(f"\nEstimated fundamental : {f0_est:.3f} Hz")
    print(f"Reference E2          : {E2_HZ:.3f} Hz")
    print(f"Deviation             : {cents_off:+.1f} cents  ({'sharp' if cents_off > 0 else 'flat'})")

    # TODO:
    # 1. In Part 1, at which frame length do you first clearly see the
    #    fundamental peak (~82 Hz) as a distinct bump above the noise floor?
    #    What is Δf at that point?
    #
    # 2. From the ×8 zero-padded plot in Part 2, read off the estimated
    #    fundamental frequency.  Compare it to the reference E2 = 82.407 Hz.
    #    Is the string in tune?  The terminal output gives the answer in cents
    #    (100 cents = 1 semitone).
    #
    # 3. Key question: in Part 2 you used N = 48000 samples (1 second).
    #    Could you have gotten the same result by taking the N = 4096 frame
    #    from Task 1 and zero-padding it to 384000 points?  Why or why not?
    #    What would the peak look like?


if __name__ == "__main__":
    main()
