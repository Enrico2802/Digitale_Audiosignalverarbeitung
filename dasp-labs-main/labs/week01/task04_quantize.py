from pathlib import Path
import sys
import soundfile as sf

LAB_WEEK_ROOT = Path(__file__).resolve().parent
LABS_ROOT = LAB_WEEK_ROOT.parent
REPO_ROOT = LABS_ROOT.parent
AUDIO_ROOT = REPO_ROOT / "assets" / "audio"
GENERATED_ROOT = LAB_WEEK_ROOT / "generated"
sys.path.append(str(LABS_ROOT / "src"))

from audio_utils import save_audio, to_mono, plot_waveform, quantize_signal  # noqa: E402


def main():
    input_path = AUDIO_ROOT / "speech.wav"

    x, fs = sf.read(str(input_path))
    x = to_mono(x)

    quantization_levels = [256, 16]

    for levels in quantization_levels:
        x_q = quantize_signal(x, levels)

        output_wav = GENERATED_ROOT / f"speech_quantized_{levels}_levels.wav"
        output_plot = GENERATED_ROOT / f"speech_quantized_{levels}_levels.png"

        save_audio(str(output_wav), x_q, fs)

        plot_waveform(
            x_q,
            fs,
            title=f"Quantized speech ({levels} levels)",
            start=0.20,
            end=0.23,
            output_path=str(output_plot),
        )

        print(f"Saved: {output_wav}")
        print(f"Saved: {output_plot}")

    # TODO:
    # 1. Add a comparison plot of original vs quantized.
    # 2. Calculate the quantization error and plot it.
    # 2. Try the same with sine_440hz.wav and/or music.wav.
    # 3. Which signal reveals quantization most clearly?
    # 4. What is the relation between bit depth and number of levels?


if __name__ == "__main__":
    main()