# Week 3 Lab: Frequency Resolution, Leakage, and Windowing

## Setup

The lab uses a real recording of a plucked guitar E string:

```
assets/audio/guitar_e_string.wav
```

48 kHz stereo, ~9 s total. Run all scripts from the
repository root.

## Learning goals

By the end of this lab you should be able to:

- compute and explain frequency resolution Δf = fs / N
- predict the minimum N needed for a given target Δf
- explain spectral leakage and why it occurs
- apply a window function and describe its effect on leakage and dynamic range
- explain why zero-padding does not improve frequency resolution but can help
  locate a peak accurately
- estimate the pitch of a real instrument note from its spectrum

## Tasks

### Task 1 — Spectral leakage

```bash
python labs/week03/task01_leakage.py
```

Analyse a short frame (N = 4096, ~85 ms) of the guitar note with a rectangular
window. Observe leakage spreading energy from the strong harmonics into
neighbouring bins.

### Task 2 — Window functions

```bash
python labs/week03/task02_windowing.py
```

Apply rectangular, Hann, Hamming, and Blackman windows to the same frame.
Compare how many harmonics are visible and how the choice of window affects the
leakage skirts.

### Task 3 — Frequency resolution and pitch estimation

```bash
python labs/week03/task03_frequency_resolution.py
```

Sweep frame length from 11 ms to 1 s and observe how Δf affects the spectrum.
Use zero-padding on the 1-second frame to estimate the fundamental frequency to
sub-Hz precision.  Is the string in tune?

## Results

Each task script contains `# TODO` blocks with the questions to answer.
Work through them in order — Task 1 → Task 2 → Task 3.
