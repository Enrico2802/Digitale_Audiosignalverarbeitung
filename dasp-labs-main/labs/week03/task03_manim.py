"""Manim animation for Task 3 — Frequency Resolution vs. Zero-Padding.

Answers the TODO reflection questions:
  Q1  At which frame length the fundamental (~82 Hz) first becomes visible
  Q2  Estimated fundamental from ×8 zero-padded spectrum
  Q3  Why N=4096 zero-padded to 384 000 pts does NOT give the same result
      as a genuine 1-second frame

Parallel render (recommended):
    python dasp-labs-main/labs/week03/render_parallel.py

Single scene:
    manim -qk dasp-labs-main/labs/week03/task03_manim.py Part1Scene
"""
from manim import *
import numpy as np

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width = 3840


FS = 1000
F1 = 82.4
F2 = 91.0


def compute_spectrum(N_real, pad_factor, fs=FS, f1=F1, f2=F2):
    n = np.arange(N_real)
    sig = np.sin(2 * np.pi * f1 * n / fs) + 0.9 * np.sin(2 * np.pi * f2 * n / fs)
    w = np.hanning(N_real)
    N_pad = N_real * pad_factor
    x_pad = np.concatenate([sig * w, np.zeros(N_pad - N_real)])
    X = np.fft.rfft(x_pad)
    freqs = np.fft.rfftfreq(N_pad, d=1.0 / fs)
    mag = 20 * np.log10(np.abs(X) + 1e-9)
    mag -= mag.max()
    return freqs, mag


def make_axes():
    return Axes(
        x_range=[70, 105, 5], y_range=[-55, 5, 10],
        x_length=10.5, y_length=4.6, tips=False,
    ).shift(DOWN * 0.2)


def make_ref_lines(ax):
    ref_items = VGroup()
    for f, tag, col in [(F1, f"f₁ = {F1} Hz", BLUE_B),
                        (F2, f"f₂ = {F2} Hz", GREEN_B)]:
        vl = DashedLine(ax.c2p(f, -55), ax.c2p(f, 5),
                        color=col, stroke_width=1.5, dash_length=0.12)
        lbl = Text(tag, font_size=14, color=col).next_to(ax.c2p(f, 5), UP, buff=0.05)
        ref_items.add(vl, lbl)
    return ref_items


def animate_cases(scene, ax, cases):
    prev_graph = None
    prev_info  = None
    for N_real, pad_factor, color, label in cases:
        freqs, mag = compute_spectrum(N_real, pad_factor)
        mask = (freqs >= 70) & (freqs <= 105)
        graph = ax.plot_line_graph(
            freqs[mask].tolist(), np.clip(mag[mask], -55, 5).tolist(),
            line_color=color, add_vertex_dots=False, stroke_width=2.3,
        )
        info = Text(label, font_size=17, color=color).to_edge(DOWN, buff=0.3)
        if prev_graph is None:
            scene.play(Create(graph), Write(info))
        else:
            scene.play(FadeOut(prev_graph), FadeIn(graph),
                       FadeOut(prev_info),  FadeIn(info),
                       run_time=0.85)
        scene.wait(2.5)
        prev_graph, prev_info = graph, info


class Part1Scene(Scene):
    def construct(self):
        title = Text("Frequency Resolution vs. Zero-Padding", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.6)
        self.play(title.animate.scale(0.6).to_edge(UP))

        ax = make_axes()
        x_lbl = Text("Frequency [Hz]", font_size=17).next_to(ax, DOWN, buff=0.1)
        y_lbl = Text("Magnitude [dB]", font_size=17).next_to(ax, LEFT, buff=0.05)
        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(Create(make_ref_lines(ax)))
        self.wait(0.4)

        animate_cases(self, ax, [
            (60,  1, RED,    "N = 60   (60 ms)    Δf = 16.7 Hz — unresolved blob"),
            (150, 1, ORANGE, "N = 150  (150 ms)   Δf = 6.7 Hz  — still merged"),
            (300, 1, GREEN,  "N = 300  (300 ms)   Δf = 3.3 Hz  — peaks just separated"),
            (FS,  1, BLUE,   "N = 1000 (1 s)      Δf = 1.0 Hz  — clearly resolved  (Q1)"),
        ])


class Part2Scene(Scene):
    def construct(self):
        ax = make_axes()
        x_lbl = Text("Frequency [Hz]", font_size=17).next_to(ax, DOWN, buff=0.1)
        y_lbl = Text("Magnitude [dB]", font_size=17).next_to(ax, LEFT, buff=0.05)
        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(Create(make_ref_lines(ax)))

      

        animate_cases(self, ax, [
            (60,  1, RED,    "N_real = 60, ×1 pad   Δf = 16.7 Hz  — merged"),
            (60,  8, ORANGE, "N_real = 60, ×8 pad   Δf_display = 2.1 Hz — smooth but still merged!"),
            (FS,  1, GREEN,  "N_real = 1000, ×1 pad  Δf = 1.0 Hz  — resolved"),
            (FS,  8, BLUE,   "N_real = 1000, ×8 pad  Δf_display = 0.125 Hz — resolved + precise  (Q2)"),
        ])


class ConclusionScene(Scene):
    def construct(self):
        conclusion = VGroup(
            Text("Q3 — Could N = 4096 zero-padded to 384 000 give the same result?",
                 font_size=22, color=YELLOW),
            Text("NO.", font_size=36, color=RED),
            Text("Real resolution:  Δf = fs / N_real  =  48 000 / 4 096  ≈  11.7 Hz",
                 font_size=21, color=RED),
            Text("That is wider than the ~8 Hz gap between harmonics.", font_size=21),
            Text("Zero-padding only interpolates between existing bins.", font_size=21),
            Text("It cannot invent frequency information that wasn't recorded.", font_size=21, color=RED),
            Text("→ The peak would still be one broad, merged blob.", font_size=21, color=RED),
            Text("A genuine 1-second frame has Δf = 1 Hz → peaks clearly separate.", font_size=21, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).center()

        for mob in conclusion:
            self.play(Write(mob), run_time=0.7)
            self.wait(0.3)
        self.wait(3)
