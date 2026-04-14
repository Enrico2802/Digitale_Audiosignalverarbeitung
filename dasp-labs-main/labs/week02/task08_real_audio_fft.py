from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"
sys.path.append(str(LABS_ROOT / "src"))

from audio_utils import to_mono  # noqa: E402
from dsp_utils import plot_waveform_and_spectrum  # noqa: E402

import soundfile as sf


def main():
    files = [
        REPO_ROOT / "assets" / "audio" / "speech.wav",
        REPO_ROOT / "assets" / "audio" / "music.wav",
    ]

    for path in files:
        x, fs = sf.read(str(path))
        x = to_mono(x)

        segment = x[: int(1.0 * fs)]

        plot_waveform_and_spectrum(
            segment,
            fs,
            wave_title=f"Waveform: {path.name}",
            spec_title=f"Magnitude spectrum: {path.name}",
            max_freq=5000,
        )

    # TODO:
    # 1. Compare the spectra of speech and music.
    # 2. Which one looks more harmonic?
    # 3. Which one looks more broadband?
    # 4. Try a different segment.


if __name__ == "__main__":
    main()