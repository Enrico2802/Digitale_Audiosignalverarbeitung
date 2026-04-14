from pathlib import Path
import soundfile as sf

# some path handling to make it easier to load files
LAB_WEEK_ROOT = Path(__file__).resolve().parent
LABS_ROOT = LAB_WEEK_ROOT.parent
REPO_ROOT = LABS_ROOT.parent
AUDIO_ROOT = REPO_ROOT / "assets" / "audio"


def main():
    input_path = AUDIO_ROOT / "speech.wav"

    # read the audio file
    x, fs = sf.read(str(input_path))

    print(f"File: {input_path}")
    print(f"Shape: {x.shape}")
    print(f"Sample rate: {fs} Hz")
    print(f"Number of samples: {x.shape[0]}")
    print(f"Duration: {x.shape[0] / fs:.3f} s")
    print(f"Channels: {x.shape[1] if x.ndim > 1 else 1}")
    # print(f"Dtype: {x.dtype}")

    if x.ndim == 1 or x.shape[1] == 1:
        print("This file is mono.")
    else:
        print("This file is multi-channel.")

    # TODO:
    # 1. Change the input file to music.wav or sine_440hz.wav.
    # 2. Compare shape, duration, and channels.
    # 3. Add a short note below here explaining what changes.

    # NOTE:
    # - The soundfile.read function does not return information about the bit depth of the audio file in the same way it returns the sample rate. To determine the bit depth, you can check the dtype of the loaded audio signal (x.dtype). Common dtypes include:
    #   - int16: 16-bit audio (most common for CD-quality audio)
    #   - int32: 24-bit or 32-bit audio (used for higher quality recordings)
    #   - float32: 32-bit floating-point audio (used for high quality and processing)
    #   - float64: 64-bit floating-point audio (used for very high quality and processing)


if __name__ == "__main__":
    main()