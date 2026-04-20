"""
Digital Signal Processing utility functions for audio analysis.
Includes FFT computation, plotting, and signal analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_rfft(x, fs):
    """
    Compute the real FFT of a signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sample rate in Hz
        
    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz
    X : np.ndarray
        Complex FFT coefficients (rfft output)
    """
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, X


def plot_waveform_and_spectrum(x, fs, wave_title="Waveform", spec_title="Magnitude Spectrum", max_freq=None):
    """
    Plot the waveform and magnitude spectrum side by side.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sample rate in Hz
    wave_title : str
        Title for the waveform plot
    spec_title : str
        Title for the spectrum plot
    max_freq : float or None
        Maximum frequency to display. If None, shows all frequencies.
    """
    freqs, X = compute_rfft(x, fs)
    magnitude = np.abs(X)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform plot (show first 0.1 seconds or all if shorter)
    time = np.arange(len(x)) / fs
    display_samples = min(len(x), int(0.1 * fs))
    ax1.plot(time[:display_samples], x[:display_samples], linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(wave_title)
    ax1.grid(True, alpha=0.3)
    
    # Magnitude spectrum plot
    if max_freq is None:
        max_freq = fs / 2
    max_bin = int(max_freq * len(x) / fs)
    
    ax2.plot(freqs[:max_bin], magnitude[:max_bin], linewidth=1)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(spec_title)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max_freq])
    
    plt.tight_layout()
    plt.show()


def plot_phase_spectrum(x, fs, title="Phase Spectrum", max_freq=None):
    """
    Plot the phase spectrum.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sample rate in Hz
    title : str
        Title for the plot
    max_freq : float or None
        Maximum frequency to display. If None, shows all frequencies.
    """
    freqs, X = compute_rfft(x, fs)
    phase = np.angle(X)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if max_freq is None:
        max_freq = fs / 2
    max_bin = int(max_freq * len(x) / fs)
    
    # Only plot phase where magnitude is significant (to avoid noise)
    magnitude = np.abs(X)
    threshold = 0.01 * np.max(magnitude)
    
    for i in range(max_bin):
        if magnitude[i] > threshold:
            ax.plot(freqs[i], phase[i], 'b.', markersize=4)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_freq])
    ax.set_ylim([-np.pi, np.pi])
    
    plt.tight_layout()
    plt.show()


def dominant_bin_info(x, fs):
    """
    Find the dominant frequency bin and return its information.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sample rate in Hz
        
    Returns
    -------
    info : str
        Formatted string with dominant bin information
    """
    freqs, X = compute_rfft(x, fs)
    magnitude = np.abs(X)
    dominant_bin = np.argmax(magnitude)
    dominant_freq = freqs[dominant_bin]
    dominant_mag = magnitude[dominant_bin]
    dominant_phase = np.angle(X[dominant_bin])
    
    info = (
        f"Dominant bin index: {dominant_bin}\n"
        f"Dominant frequency: {dominant_freq:.2f} Hz\n"
        f"Magnitude: {dominant_mag:.4f}\n"
        f"Phase: {dominant_phase:.4f} rad ({np.degrees(dominant_phase):.2f}°)"
    )
    return info
