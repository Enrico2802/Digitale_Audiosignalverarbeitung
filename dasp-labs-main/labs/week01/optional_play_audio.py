from pathlib import Path
import sys

import soundfile as sf
import sounddevice as sd

LAB_WEEK_ROOT = Path(__file__).resolve().parent
LABS_ROOT = LAB_WEEK_ROOT.parent
REPO_ROOT = LABS_ROOT.parent
AUDIO_ROOT = REPO_ROOT / "assets" / "audio"
GENERATED_ROOT = LAB_WEEK_ROOT / "generated"
sys.path.append(str(LABS_ROOT / "src"))


def play_file(path: Path):
    x, fs = sf.read(str(path))
    print(f"Playing: {path}")
    sd.play(x, fs)
    sd.wait()


def main():
    files = [
        AUDIO_ROOT / "speech.wav",
        GENERATED_ROOT / "speech_quantized_16_levels.wav",
    ]

    for path in files:
        if path.exists():
            play_file(path)
        else:
            print(f"File not found: {path}")

    # TODO:
    # 1. What will happen if you change the sample rate that's used for playback (in line 18)?
    # 2. Change it and find out whether you were right.
    # 3. Calculate and play back the quantization error signal.


if __name__ == "__main__":
    main()