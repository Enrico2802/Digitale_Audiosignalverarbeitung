from pathlib import Path
import sys
import soundfile as sf

LAB_WEEK_ROOT = Path(__file__).resolve().parent
LABS_ROOT = LAB_WEEK_ROOT.parent
REPO_ROOT = LABS_ROOT.parent
AUDIO_ROOT = REPO_ROOT / "assets" / "audio"
sys.path.append(str(LABS_ROOT / "src"))

from audio_utils import plot_waveform  # noqa: E402


def main():
    input_path = AUDIO_ROOT / "speech.wav"

    x, fs = sf.read(str(input_path))

    # Full waveform
    plot_waveform(
        x,
        fs,
        title="Speech waveform - full view",
        dont_show=True
    )

    # Zoomed waveform
    plot_waveform(
        x,
        fs,
        title="Speech waveform - zoomed view",
        start=0.20,
        end=0.23
    )

    # TODO:
    # 1. Try the same with music.wav.
    # 2. Choose a different zoom interval.
    # 3. Try to find a section in the speech signal where you can see clear periodicity (repeating patterns) and zoom in on that. Can you estimate the frequency of the sound from the plot?
    # 4. Explain what the full view shows and what the zoomed view shows.


if __name__ == "__main__":
    main()