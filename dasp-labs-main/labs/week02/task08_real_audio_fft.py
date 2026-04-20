from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"
sys.path.append(str(LABS_ROOT / "src"))

from audio_utils import to_mono  # noqa: E402
from dsp_utils import plot_waveform_and_spectrum, compute_rfft  # noqa: E402

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

    # SOLUTION: Detailed spectral analysis and comparison
    print("="*70)
    print("TASK 1-3: Comparing speech and music spectra")
    print("="*70)
    print()
    
    # Load both audio files
    speech_path = REPO_ROOT / "assets" / "audio" / "speech.wav"
    music_path = REPO_ROOT / "assets" / "audio" / "music.wav"
    
    x_speech, fs_speech = sf.read(str(speech_path))
    x_music, fs_music = sf.read(str(music_path))
    
    x_speech = to_mono(x_speech)
    x_music = to_mono(x_music)
    
    # Extract 1-second segments for comparison
    seg_speech = x_speech[:int(1.0 * fs_speech)]
    seg_music = x_music[:int(1.0 * fs_music)]
    
    # Compute FFTs
    freqs_speech, X_speech = compute_rfft(seg_speech, fs_speech)
    freqs_music, X_music = compute_rfft(seg_music, fs_music)
    
    mag_speech = np.abs(X_speech)
    mag_music = np.abs(X_music)
    mag_speech_db = 20 * np.log10(mag_speech + 1e-8)
    mag_music_db = 20 * np.log10(mag_music + 1e-8)
    
    print("SPEECH vs MUSIC SPECTRAL CHARACTERISTICS")
    print()
    print("1. COMPARING THE SPECTRA:")
    print("-" * 70)
    print()
    print("SPEECH spectrum characteristics:")
    print("  • Shows SHARP PEAKS at specific frequencies (formants)")
    print("  • Peaks appear at vowel resonances (e.g., F1, F2, F3 formants)")
    print("  • Most energy concentrated below 4 kHz")
    print("  • Clear structure with well-defined frequency components")
    print()
    print("MUSIC spectrum characteristics:")
    print("  • Shows MANY HARMONICS (multiple peaks at regular intervals)")
    print("  • Harmonics spaced by fundamental frequency (e.g., 440 Hz → 880, 1320, ...)")
    print("  • Energy spreads more throughout the frequency range")
    print("  • May have both tonal (harmonic) and noise-like components")
    print()
    
    # Identify peaks in both spectra
    threshold_speech = 0.05 * np.max(mag_speech)
    threshold_music = 0.05 * np.max(mag_music)
    
    peaks_speech = np.where(mag_speech > threshold_speech)[0]
    peaks_music = np.where(mag_music > threshold_music)[0]
    
    print("2. HARMONIC vs BROADBAND ANALYSIS:")
    print("-" * 70)
    print()
    
    # Check for harmonic relationships in music
    if len(peaks_music) > 1:
        peak_freqs_music = freqs_music[peaks_music[:10]]  # First 10 peaks
        if len(peak_freqs_music) > 1:
            # Check if peaks have harmonic relationships
            ratio_diff = np.diff(peak_freqs_music) / peak_freqs_music[:-1]
            harmonic_ratio = np.mean(ratio_diff)
    
    print("SPEECH:")
    print(f"  • Number of significant peaks: {len(peaks_speech)}")
    print(f"  • Peak frequencies (formants): {freqs_speech[peaks_speech[:5]].astype(int)} Hz")
    print(f"  • Character: HARMONIC but with FORMANT structure")
    print(f"  • Reason: Speech is produced by exciting vocal folds (harmonic)")
    print(f"    then filtering through vocal tract (creates formants)")
    print()
    
    print("MUSIC:")
    print(f"  • Number of significant peaks: {len(peaks_music)}")
    print(f"  • Peak frequencies: {freqs_music[peaks_music[:5]].astype(int)} Hz")
    print(f"  • Character: HIGHLY HARMONIC (musical notes have clear harmonics)")
    print(f"  • Reason: Instruments produce periodic oscillations with harmonics")
    print()
    
    # Estimate spectral spread (flatness)
    energy_ratio_speech = np.sum(mag_speech_db[freqs_speech < 1000]) / np.sum(mag_speech_db)
    energy_ratio_music = np.sum(mag_music_db[freqs_music < 1000]) / np.sum(mag_music_db)
    
    print("3. HARMONIC vs BROADBAND:")
    print("-" * 70)
    print()
    print("SPEECH is MORE 'BROADBAND' in character:")
    print(f"  • Energy in low frequencies (0-1 kHz): {energy_ratio_speech*100:.1f}%")
    print(f"  • But also has frication noise (consonants add broadband energy)")
    print(f"  • Spectral shape changes rapidly (time-varying)")
    print()
    print("MUSIC is MORE 'HARMONIC' in character:")
    print(f"  • Energy in low frequencies (0-1 kHz): {energy_ratio_music*100:.1f}%")
    print(f"  • Clear harmonic series (fundamental + integer multiples)")
    print(f"  • More stable spectral content (sustained notes)")
    print()
    
    # Create detailed comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Speech waveform and spectrum
    time_speech = np.arange(len(seg_speech)) / fs_speech
    axes[0, 0].plot(time_speech[:int(0.2*fs_speech)], seg_speech[:int(0.2*fs_speech)], linewidth=0.5)
    axes[0, 0].set_title("Speech Waveform (first 0.2 s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    mask_speech = freqs_speech <= 5000
    axes[0, 1].plot(freqs_speech[mask_speech], mag_speech_db[mask_speech], linewidth=1)
    axes[0, 1].set_title("Speech Spectrum (Log Scale)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=-80)
    
    # Music waveform and spectrum
    time_music = np.arange(len(seg_music)) / fs_music
    axes[1, 0].plot(time_music[:int(0.2*fs_music)], seg_music[:int(0.2*fs_music)], linewidth=0.5)
    axes[1, 0].set_title("Music Waveform (first 0.2 s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)
    
    mask_music = freqs_music <= 5000
    axes[1, 1].plot(freqs_music[mask_music], mag_music_db[mask_music], linewidth=1)
    axes[1, 1].set_title("Music Spectrum (Log Scale)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=-80)
    
    plt.tight_layout()
    plt.show()
    
    # TASK 4: Try different segments
    print()
    print("="*70)
    print("TASK 4: How do the spectra change with different segments?")
    print("="*70)
    print()
    
    # Analyze multiple segments
    segment_times = [
        (0, 1),           # First second
        (1, 2),           # Second second
        (int(len(x_speech)/4/fs_speech), int(len(x_speech)/4/fs_speech) + 1),  # Middle
    ]
    
    print("Speech segments analysis:")
    for start, end in segment_times[:2]:  # First two for speech
        if start < len(seg_speech) / fs_speech and end <= len(seg_speech) / fs_speech:
            seg = seg_speech[int(start*fs_speech):int(end*fs_speech)]
            if len(seg) > 0:
                freqs, X = compute_rfft(seg, fs_speech)
                mag = np.abs(X)
                peak_idx = np.argmax(mag)
                peak_freq = freqs[peak_idx]
                
                print(f"  Segment {start}-{end}s:")
                print(f"    Dominant frequency: {peak_freq:.1f} Hz")
                print(f"    Peak magnitude: {np.max(mag):.2f}")
    
    print()
    print("Music segments analysis:")
    for start, end in segment_times[:2]:  # First two for music
        if start < len(seg_music) / fs_music and end <= len(seg_music) / fs_music:
            seg = seg_music[int(start*fs_music):int(end*fs_music)]
            if len(seg) > 0:
                freqs, X = compute_rfft(seg, fs_music)
                mag = np.abs(X)
                peak_idx = np.argmax(mag)
                peak_freq = freqs[peak_idx]
                
                print(f"  Segment {start}-{end}s:")
                print(f"    Dominant frequency: {peak_freq:.1f} Hz")
                print(f"    Peak magnitude: {np.max(mag):.2f}")
    
    print()
    print("Key observations about segment variation:")
    print("  • SPEECH: Dominant frequency shifts dramatically (different phones/vowels)")
    print("  • MUSIC: More stable harmonic structure across segments (sustained notes)")
    print("  • SPEECH: Spectral content highly time-dependent")
    print("  • MUSIC: Harmonic peaks remain relatively consistent per note")
    print()
    
    # Create segment comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Speech segments
    for i, (start, end) in enumerate([(0, 0.5), (0.5, 1.0)]):
        seg = seg_speech[int(start*fs_speech):int(end*fs_speech)]
        freqs, X = compute_rfft(seg, fs_speech)
        mag_db = 20 * np.log10(np.abs(X) + 1e-8)
        
        mask = freqs <= 5000
        axes[0, i].plot(freqs[mask], mag_db[mask], linewidth=1, label=f"{start}-{end}s")
        axes[0, i].set_title(f"Speech Segment {start}-{end}s")
        axes[0, i].set_xlabel("Frequency (Hz)")
        axes[0, i].set_ylabel("Magnitude (dB)")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(bottom=-80)
        axes[0, i].legend()
    
    # Music segments
    for i, (start, end) in enumerate([(0, 0.5), (0.5, 1.0)]):
        seg = seg_music[int(start*fs_music):int(end*fs_music)]
        freqs, X = compute_rfft(seg, fs_music)
        mag_db = 20 * np.log10(np.abs(X) + 1e-8)
        
        mask = freqs <= 5000
        axes[1, i].plot(freqs[mask], mag_db[mask], linewidth=1, label=f"{start}-{end}s")
        axes[1, i].set_title(f"Music Segment {start}-{end}s")
        axes[1, i].set_xlabel("Frequency (Hz)")
        axes[1, i].set_ylabel("Magnitude (dB)")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim(bottom=-80)
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()