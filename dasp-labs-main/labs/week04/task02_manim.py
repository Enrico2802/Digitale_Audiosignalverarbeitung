"""Manim animation for Task 2 — Reading Spectrograms.

Parallel render (recommended):
    python dasp-labs-main/labs/week04/render_parallel.py task02

Single scene:
    manim -qk dasp-labs-main/labs/week04/task02_manim.py SpeechSpectrogramScene
"""
from manim import *
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width  = 3840

FS   = 8000
N_b  = 512
HOP  = N_b // 4
DUR  = 3.0


def stft_image(x, N_b=N_b, max_hz=4000, vmin=-80):
    window = np.hanning(N_b)
    frames = []
    for start in range(0, len(x) - N_b + 1, HOP):
        frames.append(np.abs(np.fft.rfft(x[start:start + N_b] * window)))
    S     = np.array(frames).T
    S_db  = 20 * np.log10(S + 1e-9)
    S_db -= S_db.max()
    freqs = np.fft.rfftfreq(N_b, d=1 / FS)
    cap   = np.searchsorted(freqs, max_hz)
    norm  = Normalize(vmin=vmin, vmax=0)
    rgba  = (cm.inferno(norm(S_db[:cap, :])) * 255).astype(np.uint8)
    return rgba[::-1, :, :3]


def make_speech_signal():
    n = int(DUR * FS)
    t = np.arange(n) / FS
    # Simulate speech: alternating voiced (harmonic) and unvoiced (noise) segments
    rng  = np.random.default_rng(42)
    x    = np.zeros(n)
    f0   = 130.0
    for k in range(1, 8):
        x += (1 / k) * np.sin(2 * np.pi * f0 * k * t)
    # Vowel envelope: on during 0-0.3, 0.5-0.8, 1.0-1.3, 1.5-1.8, 2.0-2.3, 2.5-2.8 s
    env = np.zeros(n)
    for start, stop in [(0, 0.3), (0.5, 0.8), (1.0, 1.3), (1.5, 1.8), (2.0, 2.3), (2.5, 2.8)]:
        i0, i1 = int(start * FS), int(stop * FS)
        env[i0:i1] = 1.0
    voiced = x * env
    # Fricative (broadband noise) at short segments
    noise_env = np.zeros(n)
    for start, stop in [(0.35, 0.48), (0.85, 0.98), (1.35, 1.48)]:
        i0, i1 = int(start * FS), int(stop * FS)
        noise_env[i0:i1] = 0.6
    noise = rng.standard_normal(n) * noise_env
    return voiced + noise


def make_guitar_signal():
    n   = int(DUR * FS)
    t   = np.arange(n) / FS
    f0  = 82.4
    env = np.exp(-t * 1.5)
    x   = np.zeros(n)
    for k in range(1, 12):
        x += (1 / k) * np.sin(2 * np.pi * f0 * k * t) * env
    return x


class SpeechSpectrogramScene(Scene):
    def construct(self):
        title = Text("Spectrogram of Speech", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x   = make_speech_signal()
        img = ImageMobject(stft_image(x, max_hz=3500)).set_height(4.2).shift(UP * 0.1)
        border = Rectangle(width=img.width + 0.08, height=img.height + 0.08,
                           color=WHITE, stroke_width=2).move_to(img)

        x_lbl = Text("Time (0 → 3 s) →",          font_size=16, color=GREY_A).next_to(img, DOWN, buff=0.15)
        y_lbl = Text("↑ Frequency (0 → 3.5 kHz)", font_size=16, color=GREY_A).next_to(img, LEFT, buff=0.12)

        self.play(FadeIn(img), Create(border), Write(x_lbl), Write(y_lbl))
        self.wait(0.5)

        annotations = VGroup(
            Text("Dark columns = pauses / silence", font_size=19, color=GREY_B),
            Text("Bright horizontal bands = vowel formants", font_size=19, color=ORANGE),
            Text("Vertical streaks (broadband) = consonants (s, t, …)", font_size=19, color=BLUE_B),
            Text("Harmonic lines at multiples of f0 ≈ 130 Hz", font_size=19, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(DOWN, buff=0.3)

        for ann in annotations:
            self.play(Write(ann), run_time=0.6)
            self.wait(0.5)
        self.wait(2)


class GuitarSpectrogramScene(Scene):
    def construct(self):
        title = Text("Spectrogram of Guitar E2 String", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x   = make_guitar_signal()
        img = ImageMobject(stft_image(x, max_hz=2000)).set_height(4.2).shift(UP * 0.1)
        border = Rectangle(width=img.width + 0.08, height=img.height + 0.08,
                           color=WHITE, stroke_width=2).move_to(img)

        x_lbl = Text("Time (0 → 3 s) →",          font_size=16, color=GREY_A).next_to(img, DOWN, buff=0.15)
        y_lbl = Text("↑ Frequency (0 → 2 kHz)", font_size=16, color=GREY_A).next_to(img, LEFT, buff=0.12)

        self.play(FadeIn(img), Create(border), Write(x_lbl), Write(y_lbl))
        self.wait(0.5)

        annotations = VGroup(
            Text("Bright at t≈0: broadband attack transient", font_size=19, color=ORANGE),
            Text("Horizontal lines: harmonics at 82.4, 164.8, 247.2 Hz …", font_size=19, color=GREEN),
            Text("Lines fade over time: string energy decays exponentially", font_size=19, color=BLUE_B),
            Text("No energy above ~1.5 kHz during sustain phase", font_size=19, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(DOWN, buff=0.3)

        for ann in annotations:
            self.play(Write(ann), run_time=0.6)
            self.wait(0.5)
        self.wait(2)


class SpeechGuitarCompareScene(Scene):
    def construct(self):
        comparison = VGroup(
            Text("Speech vs. Guitar — Spectrogram Comparison", font_size=30, color=YELLOW),
            Text("Speech:", font_size=24, color=ORANGE),
            Text("  • Energy spread across wide frequency range", font_size=21),
            Text("  • Alternating voiced (harmonic) and unvoiced (noise) segments", font_size=21),
            Text("  • Formants vary over time as the vocal tract changes shape", font_size=21),
            Text("  • Strong energy above 4 kHz (sibilants)", font_size=21),
            Text("Guitar:", font_size=24, color=GREEN),
            Text("  • Energy concentrated in narrow harmonic lines", font_size=21),
            Text("  • All harmonics are exact multiples of f0 = 82.4 Hz", font_size=21),
            Text("  • Energy fades over time (exponential decay)", font_size=21),
            Text("  • Very little energy above 2–3 kHz during sustain", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).center()

        for mob in comparison:
            self.play(Write(mob), run_time=0.55)
            self.wait(0.2)
        self.wait(3)
