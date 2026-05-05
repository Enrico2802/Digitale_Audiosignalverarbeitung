# Week 5 Lab: FIR Filters — Probing and Shaping the Frequency Response

## Setup

The lab uses audio files from the shared assets folder:

```
assets/audio/speech.wav
assets/audio/music.wav
```

48 kHz mono/stereo. Run all scripts from the repository root.

## Learning goals

By the end of this lab you should be able to:

- apply an FIR filter using the weighted-sum formula directly
- explain why a constant (DC) input gives a constant output scaled by `sum(b)`
- observe that a sinusoid in always produces a sinusoid out at the same frequency
- measure the gain of a filter at a single frequency using sinusoidal probing
- build the empirical magnitude response by probing at many frequencies in a loop
- compute the exact magnitude response from the transfer function formula `H(e^{jΩ})`
- change the coefficients and predict — and hear — what changes in the output

## Tasks

### Task 1 — Probing a FIR filter with simple inputs

```bash
python labs/week05/task01_sinusoid_probing.py
```

A 5-tap box average is applied to a speech recording in the scaffold — run it
first and listen. The three parts then probe the same filter with a DC signal
and with single-frequency sinusoids (200 Hz and 4 000 Hz), and verify that the
output is always a sinusoid at the input frequency.

### Task 2 — Measuring and shaping the magnitude response

```bash
python labs/week05/task02_transfer_function.py
```

Build the full magnitude response empirically by probing the filter at many
frequencies in a loop, then overlay the analytical curve from the transfer
function formula. Part 6 compares different coefficient sets — including a
pure delay and an 11-tap average — by measuring, plotting, and listening.
