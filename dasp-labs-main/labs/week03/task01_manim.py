"""Manim animation for Task 1 — Spectral Leakage.

Answers the TODO reflection questions:
  Q2  Why leakage occurs (on-bin vs. off-bin DFT animation)
  Q3  Minimum N for ±2 Hz frequency accuracy

Parallel render (recommended):
    python dasp-labs-main/labs/week03/render_parallel.py task01

Single scene:
    manim -qk dasp-labs-main/labs/week03/task01_manim.py LeakageOnOffScene
"""
from manim import *
import numpy as np

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width = 3840


def make_stems(ax, xs, ys, color, sw=2.5):
    g = VGroup()
    for x, y in zip(xs, ys):
        g.add(Line(ax.c2p(x, 0), ax.c2p(x, float(y)), color=color, stroke_width=sw))
    return g


class LeakageOnOffScene(Scene):
    def construct(self):
        N    = 64
        n    = np.arange(N)
        bins = np.arange(N // 2 + 1)

        title = Text("Spectral Leakage", font_size=42, color=YELLOW)
        self.play(Write(title))
        self.wait(0.6)
        self.play(title.animate.scale(0.6).to_edge(UP))

        ax_t = Axes(
            x_range=[0, 63, 16], y_range=[-1.3, 1.3, 0.5],
            x_length=5.5, y_length=2.4, tips=False,
        ).shift(LEFT * 3.2 + UP * 0.5)

        ax_f = Axes(
            x_range=[0, 32, 8], y_range=[0, 1.1, 0.5],
            x_length=5.5, y_length=2.4, tips=False,
        ).shift(RIGHT * 3.2 + UP * 0.5)

        t_hdr = Text("Time domain",   font_size=18).next_to(ax_t, UP,   buff=0.05)
        f_hdr = Text("DFT magnitude", font_size=18).next_to(ax_f, UP,   buff=0.05)
        t_lbl = Text("n  (sample)",   font_size=16).next_to(ax_t, DOWN, buff=0.1)
        f_lbl = Text("k  (bin)",      font_size=16).next_to(ax_f, DOWN, buff=0.1)

        self.play(Create(ax_t), Create(ax_f),
                  Write(t_hdr), Write(f_hdr), Write(t_lbl), Write(f_lbl))

        sig_on  = np.sin(2 * np.pi * 8.0 * n / N)
        X_on    = np.abs(np.fft.rfft(sig_on)) * 2 / N
        t_graph  = ax_t.plot_line_graph(n.tolist(), sig_on.tolist(),
                       line_color=BLUE, add_vertex_dots=False, stroke_width=2.2)
        f_stems  = make_stems(ax_f, bins, X_on, BLUE)
        case_lbl = Text("On-bin  (k = 8.0 exactly) → single clean spike",
                        font_size=19, color=BLUE).to_edge(DOWN, buff=0.5)

        self.play(Create(t_graph), Create(f_stems), Write(case_lbl))
        self.wait(2.5)

        sig_off      = np.sin(2 * np.pi * 8.5 * n / N)
        X_off        = np.abs(np.fft.rfft(sig_off)) * 2 / N
        t_graph_off  = ax_t.plot_line_graph(n.tolist(), sig_off.tolist(),
                           line_color=RED, add_vertex_dots=False, stroke_width=2.2)
        f_stems_off  = make_stems(ax_f, bins, X_off, RED)
        case_lbl_off = Text("Off-bin  (k = 8.5) → energy spreads across ALL bins!",
                             font_size=19, color=RED).to_edge(DOWN, buff=0.5)

        self.play(
            Transform(t_graph,  t_graph_off),
            Transform(f_stems,  f_stems_off),
            Transform(case_lbl, case_lbl_off),
            run_time=1.4,
        )
        self.wait(2.5)


class LeakageWhyScene(Scene):
    def construct(self):
        why = VGroup(
            Text("Why spectral leakage occurs  (Q2)", font_size=28, color=YELLOW),
            Text("The DFT assumes the signal repeats every N samples.", font_size=21),
            Text("An off-bin frequency has a phase jump at the frame boundary.", font_size=21),
            Text("→ Discontinuity = energy at ALL frequencies.", font_size=21, color=RED),
            Text("Rectangular window = sinc in the frequency domain.", font_size=21),
            Text("The sinc's high side lobes leak into every neighbouring bin.", font_size=21, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).center()

        for mob in why:
            self.play(Write(mob), run_time=0.65)
            self.wait(0.2)
        self.wait(2.5)


class LeakageQ3Scene(Scene):
    def construct(self):
        q3 = VGroup(
            Text("Q3 — Minimum N for ±2 Hz accuracy", font_size=28, color=YELLOW),
            Text("Δf = fs / N  ≤  2 Hz", font_size=36),
            Text("N  ≥  fs / 2 Hz  =  48 000 / 2  =  24 000", font_size=32),
            Text("T  =  N / fs  =  24 000 / 48 000  =  0.5 s  =  500 ms", font_size=32),
            Text("→ Need a 500 ms frame  (vs. the 85 ms frame used above)",
                 font_size=22, color=GREEN),
        ).arrange(DOWN, buff=0.48)

        for mob in q3:
            self.play(Write(mob), run_time=0.85)
            self.wait(0.55)
        self.wait(3)
