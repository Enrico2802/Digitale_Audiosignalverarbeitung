
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def load_audio(path: str):
    """
    Load an audio file and return signal and sample rate.

    Returns
    -------
    x : np.ndarray
        Audio signal, shape (N,) for mono or (N, C) for multi-channel.
    fs : int
        Sample rate in Hz.
    """
    x, fs = sf.read(path)
    return x, fs


def save_audio(path: str, x: np.ndarray, fs: int):
    """
    Save audio to disk. Creates parent directory if needed.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), x, fs)


def to_mono(x: np.ndarray) -> np.ndarray:
    """
    Convert stereo or multi-channel audio to mono by averaging channels.
    """
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def get_num_samples(x: np.ndarray) -> int:
    """
    Return number of time samples (frames).
    """
    return x.shape[0]


def get_duration(x: np.ndarray, fs: int) -> float:
    """
    Return duration in seconds.
    """
    return get_num_samples(x) / fs


def time_axis(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Return a time axis in seconds for plotting.
    """
    n = get_num_samples(x)
    return np.arange(n) / fs


def plot_waveform(
    x: np.ndarray,
    fs: int,
    title: str = "",
    start: float | None = None,
    end: float | None = None,
    output_path: str | None = None,
    dont_show: bool = False,
):
    """
    Plot a waveform. If start/end are given, plot only that time range.
    """
    x_mono = to_mono(x)
    t = time_axis(x_mono, fs)

    if start is not None or end is not None:
        start = 0.0 if start is None else start
        end = t[-1] if end is None else end
        mask = (t >= start) & (t <= end)
        t = t[mask]
        x_mono = x_mono[mask]

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_mono)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150)

    if not dont_show:
        plt.show()


def quantize_signal(x: np.ndarray, levels: int) -> np.ndarray:
    """
    Quantize a signal to a given number of amplitude levels.

    Assumes input approximately in range [-1, 1].
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")

    x_clipped = np.clip(x, -1.0, 1.0)

    # Map from [-1, 1] to [0, 1]
    x_norm = (x_clipped + 1.0) / 2.0

    # Quantize
    x_q = np.round(x_norm * (levels - 1)) / (levels - 1)

    # Map back to [-1, 1]
    return 2.0 * x_q - 1.0


def naive_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Very simple downsampling by sample decimation.

    This is intentionally naive for teaching purposes.
    It does NOT apply an anti-aliasing filter.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return x[::factor]


# def describe_audio(x: np.ndarray, fs: int) -> dict:
#     """
#     Return a small dictionary with useful audio metadata.
#     """
#     info = {
#         "shape": x.shape,
#         "sample_rate": fs,
#         "dtype": x.dtype,
#         "num_samples": get_num_samples(x),
#         "duration_seconds": get_duration(x, fs),
#         "channels": 1 if x.ndim == 1 else x.shape[1],
#     }
#     return info