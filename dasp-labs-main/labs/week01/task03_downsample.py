from pathlib import Path
import sys
import soundfile as sf

LAB_WEEK_ROOT = Path(__file__).resolve().parent
LABS_ROOT = LAB_WEEK_ROOT.parent
REPO_ROOT = LABS_ROOT.parent
AUDIO_ROOT = REPO_ROOT / "assets" / "audio"
GENERATED_ROOT = LAB_WEEK_ROOT / "generated"
sys.path.append(str(LABS_ROOT / "src"))

from audio_utils import save_audio, to_mono, plot_waveform, naive_downsample  # noqa: E402


def main():
    input_path = AUDIO_ROOT / "sine_sweep.wav"

    x, fs = sf.read(str(input_path))
    x = to_mono(x)

    print(f"Original sample rate: {fs} Hz")
    print(f"Original number of samples: {len(x)}")

    factors = [2, 4, 8]

    for factor in factors:
        x_ds = naive_downsample(x, factor)
        fs_ds = fs // factor

        output_wav = GENERATED_ROOT / f"sine_sweep_downsampled_by_{factor}.wav"
        output_plot = GENERATED_ROOT / f"sine_sweep_downsampled_by_{factor}.png"

        save_audio(str(output_wav), x_ds, fs_ds)

        plot_waveform(
            x_ds,
            fs_ds,
            title=f"Downsampled by factor {factor} (fs = {fs_ds} Hz)",
            start=0.0,
            # end=0.02,
            output_path=str(output_plot),
            dont_show=factor != factors[-1],  # only show the last plot
        )

        print(f"Saved: {output_wav}")


    # TODO:
    # 1. Repeat with speech.wav or music.wav (note that the output file names need to be adjusted accordingly).
    # 2. Listen to the generated files (focus on the highest downsampling factor).
    # 3. Describe what changes when the sample rate is reduced.
    # 4. Why is this function called naive_downsample?


if __name__ == "__main__":
    main()