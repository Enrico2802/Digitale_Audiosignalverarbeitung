# Week 2 Lab: Sinusoids, Superposition, and the Frequency Domain

## Learning goals

By the end of this lab, you should be able to:

- generate sinusoids in NumPy from the formula $x[n] = A \sin(2\pi f n / f_s + \varphi)$
- vary amplitude, frequency, and phase and describe the effect on the waveform and sound
- add multiple sinusoids together and manage amplitude to avoid clipping
- compute an FFT and interpret the raw complex output
- connect Euler's formula to the real and imaginary parts of an FFT bin
- compute one DFT bin by hand using cosine and sine correlations ($C_{\cos}$, $C_{\sin}$)
- explain why $\Delta f = f_s / N$ and what controls frequency resolution
- observe the difference between on-bin and off-bin tones in the spectrum
- compare spectra of tones, speech, and music

## Tasks

Run all scripts from the repository root:

```bash
python labs/week02/taskNN_name.py
```

---

### Task 1 — Generate a sine wave (`task01_generate_sine.py`)

Generate a 440 Hz sine wave, save it as a WAV file, and plot the waveform.

### Task 2 — Vary amplitude, frequency, and phase (`task02_vary_parameters.py`)

Plot three side-by-side comparisons showing the effect of each parameter.
Describe what changes visually and what you expect to hear.

### Task 3 — Superposition (`task03_superposition.py`)

Add two sinusoids together, plot the sum alongside the components, and save the result as audio.
Try cancellation: same frequency, opposite phase.

### Task 4 — FFT basics and Euler's formula (`task04_fft_basics.py`)

Compute the FFT of a single tone and a two-tone mixture. Inspect the raw complex output:
magnitude, phase, and the polar form $X[k] = |X[k]| \cdot e^{j \angle X[k]}$.

### Task 5 — DFT correlation by hand (`task05_dft_correlation.py`)

Compute one DFT bin manually using the cosine and sine correlations:

$$C_{\cos}[k] = \sum x[n] \cos\!\left(\tfrac{2\pi kn}{N}\right), \quad C_{\sin}[k] = \sum x[n] \sin\!\left(\tfrac{2\pi kn}{N}\right)$$

Verify that $C_{\cos} - j C_{\sin}$ matches `numpy.fft.rfft` exactly.

### Task 6 — Frequency bins and resolution (`task06_frequency_bins.py`)

Compare $\Delta f = f_s / N$ for different signal lengths.
Find the peak bin for tones at 440 Hz and 445 Hz — observe the effect of bin spacing.

### Task 7 — On-bin vs off-bin (`task07_on_off_bin.py`)

Compare a 100 Hz tone (exactly on bin 100) with a 100.7 Hz tone (between bins).
Count how many bins receive significant energy in each case, and check whether the signal
completes a whole number of cycles in the window. (This is a preview of spectral leakage —
the main topic of week 3.)

### Task 8 — FFT of real audio (`task08_real_audio_fft.py`)

Load a speech recording and a music recording. Plot waveform and magnitude spectrum for each.
Compare: which is more harmonic? which is more broadband?

---

## Reflection questions

Write short answers to these questions after completing the tasks:

1. What are the three parameters of a sinusoid? What does each one control?
2. What happens to the waveform when you add two sinusoids at the same frequency with opposite phase?
3. Why is the FFT output complex-valued? What do the real and imaginary parts represent?
4. How does Euler's formula connect $e^{j\theta}$ to $\cos\theta$ and $\sin\theta$?
5. What is $C_{\cos}[k]$? What does a large value mean?
6. How is frequency spacing $\Delta f$ determined? How would you change it?
7. Why does `rfft` return $N/2 + 1$ bins and not $N$ or $N/2$?
8. Why does an off-bin tone produce energy in many bins rather than one?
9. How do the spectra of speech and music differ from a pure tone?
10. Why do we plot magnitude spectra in dB rather than on a linear scale?
