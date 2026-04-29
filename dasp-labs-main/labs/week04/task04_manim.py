"""Manim animation for Task 4 — Telephone Filter via STFT.

Parallel render (recommended):
    python dasp-labs-main/labs/week04/render_parallel.py task04

Single scene:
    manim -qk dasp-labs-main/labs/week04/task04_manim.py FilterPipelineScene
"""
from manim import *
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width  = 3840

FS     = 8000
N_b    = 512
HOP    = N_b // 4
F_LOW  = 300.0
F_HIGH = 3000.0
DUR    = 2.0


def make_speech_signal():
    n   = int(DUR * FS)
    t   = np.arange(n) / FS
    rng = np.random.default_rng(7)
    x   = np.zeros(n)
    for k in range(1, 7):
        x += (1 / k) * np.sin(2 * np.pi * 130 * k * t)
    env = np.zeros(n)
    for s, e in [(0, 0.28), (0.4, 0.68), (0.8, 1.08), (1.2, 1.48), (1.6, 1.88)]:
        env[int(s * FS):int(e * FS)] = 1.0
    x *= env
    noise_env = np.zeros(n)
    for s, e in [(0.3, 0.38), (0.7, 0.78), (1.1, 1.18)]:
        noise_env[int(s * FS):int(e * FS)] = 0.5
    x += rng.standard_normal(n) * noise_env
    return x


def stft_manual(x):
    window = np.hanning(N_b)
    frames = []
    for start in range(0, len(x) - N_b + 1, HOP):
        frames.append(np.fft.rfft(x[start:start + N_b] * window))
    return np.array(frames).T


def apply_telephone_mask(Zxx):
    freqs = np.fft.rfftfreq(N_b, d=1 / FS)
    Zxx_f = Zxx.copy()
    Zxx_f[:np.searchsorted(freqs, F_LOW),  :] = 0
    Zxx_f[np.searchsorted(freqs, F_HIGH):, :] = 0
    return Zxx_f


def to_image(Zxx, max_hz=4000, vmin=-80):
    freqs = np.fft.rfftfreq(N_b, d=1 / FS)
    cap   = np.searchsorted(freqs, max_hz)
    S_db  = 20 * np.log10(np.abs(Zxx[:cap, :]) + 1e-9)
    S_db -= S_db.max()
    norm  = Normalize(vmin=vmin, vmax=0)
    rgba  = (cm.inferno(norm(S_db)) * 255).astype(np.uint8)
    return rgba[::-1, :, :3]


class FilterPipelineScene(Scene):
    def construct(self):
        title = Text("Telephone Filter — STFT Pipeline", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        boxes = VGroup(
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.1,
                             color=BLUE,   fill_color=BLUE,   fill_opacity=0.2),
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.1,
                             color=YELLOW, fill_color=YELLOW, fill_opacity=0.2),
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.1,
                             color=GREEN,  fill_color=GREEN,  fill_opacity=0.2),
        ).arrange(RIGHT, buff=1.2).shift(UP * 0.3)

        labels = VGroup(
            Text("STFT",         font_size=22, color=BLUE),
            Text("Zero out\nbins < 300 Hz\nor > 3000 Hz", font_size=18, color=YELLOW),
            Text("iSTFT",        font_size=22, color=GREEN),
        )
        for lbl, box in zip(labels, boxes):
            lbl.move_to(box)

        arrows = VGroup(
            Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.1, color=WHITE),
            Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.1, color=WHITE),
        )

        in_lbl  = Text("x[n]\nspeech",   font_size=18, color=WHITE).next_to(boxes[0], LEFT,  buff=0.3)
        out_lbl = Text("y[n]\ntelephone", font_size=18, color=WHITE).next_to(boxes[2], RIGHT, buff=0.3)
        in_arr  = Arrow(in_lbl.get_right(),   boxes[0].get_left(), buff=0.1, color=WHITE)
        out_arr = Arrow(boxes[2].get_right(),  out_lbl.get_left(), buff=0.1, color=WHITE)

        for mob in [boxes[0], labels[0], in_lbl, in_arr]:
            self.play(FadeIn(mob), run_time=0.4)
        self.wait(0.3)
        self.play(FadeIn(arrows[0]))
        for mob in [boxes[1], labels[1]]:
            self.play(FadeIn(mob), run_time=0.4)
        self.wait(0.3)
        self.play(FadeIn(arrows[1]))
        for mob in [boxes[2], labels[2], out_lbl, out_arr]:
            self.play(FadeIn(mob), run_time=0.4)
        self.wait(0.8)

        note = VGroup(
            Text("No filter coefficients. No convolution.", font_size=21, color=WHITE),
            Text("Just set unwanted bins to zero before the inverse transform.", font_size=21, color=YELLOW),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.5)
        self.play(Write(note[0]))
        self.wait(0.4)
        self.play(Write(note[1]))
        self.wait(2.5)


class FrequencyMaskScene(Scene):
    def construct(self):
        title = Text("Frequency-Domain Masking: 300–3000 Hz", font_size=34, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x    = make_speech_signal()
        Zxx  = stft_manual(x)
        Zxx_f = apply_telephone_mask(Zxx)

        img_orig = ImageMobject(to_image(Zxx)).set_height(3.2)
        img_filt = ImageMobject(to_image(Zxx_f)).set_height(3.2)

        Group(img_orig, img_filt).arrange(RIGHT, buff=0.8).shift(UP * 0.3)

        for img, label, col in [
            (img_orig, "Original", BLUE),
            (img_filt, "Telephone filtered", GREEN),
        ]:
            border = Rectangle(width=img.width + 0.08, height=img.height + 0.08,
                               color=col, stroke_width=2.5).move_to(img)
            lbl    = Text(label, font_size=20, color=col).next_to(img, UP, buff=0.12)
            self.play(FadeIn(img), Create(border), Write(lbl), run_time=0.7)

        freqs = np.fft.rfftfreq(N_b, d=1 / FS)
        cap   = np.searchsorted(freqs, 4000)
        img_h = 3.2

        for img in [img_orig, img_filt]:
            y_low  = img.get_bottom()[1] + img_h * (F_LOW  / 4000)
            y_high = img.get_bottom()[1] + img_h * (F_HIGH / 4000)
            low_line  = DashedLine(
                [img.get_left()[0],  y_low,  0],
                [img.get_right()[0], y_low,  0],
                color=TEAL, stroke_width=2, dash_length=0.15)
            high_line = DashedLine(
                [img.get_left()[0],  y_high, 0],
                [img.get_right()[0], y_high, 0],
                color=GREEN_A, stroke_width=2, dash_length=0.15)
            self.play(Create(low_line), Create(high_line), run_time=0.5)

        legend = VGroup(
            Text("--- 300 Hz (low cut)", font_size=17, color=TEAL),
            Text("--- 3000 Hz (high cut)", font_size=17, color=GREEN_A),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.3)
        self.play(Write(legend))
        self.wait(3)


class BrickWallScene(Scene):
    def construct(self):
        explanation = VGroup(
            Text("The Brick-Wall Filter", font_size=34, color=YELLOW),
            Text("Setting bins to zero creates a perfectly sharp cutoff in frequency.", font_size=21),
            Text("In the time domain, this corresponds to convolution with a sinc function,", font_size=21),
            Text("which is infinitely long → causes ringing (Gibbs phenomenon).", font_size=21, color=RED),
            Text("", font_size=8),
            Text("Round-trip error:", font_size=24, color=ORANGE),
            Text("  Unmodified:  irfft(rfft(x)) ≈ x  — error is pure floating-point noise (~1e-15)",
                 font_size=20, color=GREEN),
            Text("  Filtered:    irfft(Zxx_filtered) ≠ x  — error = removed frequency content",
                 font_size=20, color=RED),
            Text("", font_size=8),
            Text("Week 5 — Smooth filters (FIR/IIR) avoid the ringing", font_size=22, color=BLUE_B),
            Text("by tapering the frequency response gradually instead of a hard step.", font_size=22, color=BLUE_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).center()

        for mob in explanation:
            self.play(Write(mob), run_time=0.6)
            self.wait(0.2)
        self.wait(3)
