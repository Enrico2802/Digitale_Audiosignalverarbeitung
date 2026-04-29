# Week 4 Lab: The STFT, Spectrograms, and Frequency-Domain Processing

## Setup

The lab uses audio files from the shared assets folder:

```
assets/audio/speech.wav
assets/audio/guitar_e_string.wav
```

48 kHz mono/stereo. Run all scripts from the repository root.

## Learning goals

By the end of this lab you should be able to:

- implement the STFT loop: windowed FFT applied to overlapping frames
- explain hop size and overlap and their effect on the spectrogram
- use `scipy.signal.stft` to compute a spectrogram and read what it reveals about real audio
- reconstruct a time-domain signal from a spectrum using `np.fft.irfft`
- apply a frequency-domain filter using STFT → modify bins → ISTFT

## Tasks

### Task 1 — Building the STFT from scratch

```bash
python labs/week04/task01_time_freq_tradeoff.py
```

Implement the STFT loop step by step and display the result as a spectrogram.
The signal is synthetic (440 Hz → 880 Hz) so you can directly observe the
time–frequency tradeoff by varying N.

### Task 2 — Spectrograms of real audio

```bash
python labs/week04/task02_stft_spectrogram.py
```

Now that you have built the STFT yourself, use `scipy.signal.stft` on speech
and guitar recordings. The focus is on reading the spectrogram: phonetic
structure, harmonic content, and the difference between attack and sustain.

### Task 3 — The mystery spectrum

```bash
python labs/week04/task03_mystery_idft.py
```

A file `assets/numpy_data/mystery_spectrum.npz` contains the rfft output of an unknown
signal. Reconstruct the time-domain signal with `np.fft.irfft` and listen to
what comes out.

### Task 4 — Telephone filter

```bash
python labs/week04/task04_telephone_filter.py
```

Apply a 300–3000 Hz bandpass filter to a speech recording entirely in the
frequency domain: STFT → zero bins outside the passband → iSTFT. Compare
the spectrograms and listen to the before/after.
