"""Manim animation for Task 2 — Window Functions.

Answers the TODO reflection questions:
  Q1  Why the 916 Hz harmonic disappears with rectangular (side-lobe masking)
  Q2  Harmonics visible per window
  Q3  What determines weak-harmonic survival
  Q4  Trade-off: narrow main lobe vs. high side lobes
  Q5  Best window for guitar analysis

Parallel render (recommended):
    python dasp-labs-main/labs/week03/render_parallel.py task02

Single scene:
    manim -qk dasp-labs-main/labs/week03/task02_manim.py WindowShapeScene
"""
from manim import *
import numpy as np

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width = 3840

WINDOWS = [
    ("Rectangular", np.ones,     BLUE,   -13),
    ("Hann",        np.hanning,  GREEN,  -31),
    ("Hamming",     np.hamming,  ORANGE, -41),
    ("Blackman",    np.blackman, RED,    -57),
]

MAIN_LOBE = {
    "Rectangular": "2 bins (narrowest)",
    "Hann":        "4 bins",
    "Hamming":     "4 bins",
    "Blackman":    "6 bins (widest)",
}


class WindowShapeScene(Scene):
    def construct(self):
        N = 256
        n = np.arange(N)
        N_pad = 8192

        title = Text("Window Functions", font_size=40, color=YELLOW)
        self.play(Write(title))
        self.wait(0.6)
        self.play(title.animate.scale(0.6).to_edge(UP))

        ax_t = Axes(
            x_range=[0, N - 1, 64], y_range=[-0.1, 1.15, 0.5],
            x_length=5.5, y_length=2.5, tips=False,
        ).shift(LEFT * 3.2 + UP * 0.35)

        ax_f = Axes(
            x_range=[-20, 20, 10], y_range=[-70, 5, 20],
            x_length=5.5, y_length=2.5, tips=False,
        ).shift(RIGHT * 3.2 + UP * 0.35)

        t_hdr  = Text("Window shape (time domain)", font_size=17).next_to(ax_t, UP,   buff=0.05)
        f_hdr  = Text("Frequency response",         font_size=17).next_to(ax_f, UP,   buff=0.05)
        t_lbl  = Text("n  (sample)",                font_size=16).next_to(ax_t, DOWN, buff=0.1)
        f_lbl  = Text("bins from peak",             font_size=16).next_to(ax_f, DOWN, buff=0.1)
        f_ylbl = Text("dB",                         font_size=15).next_to(ax_f, LEFT, buff=0.05)

        self.play(Create(ax_t), Create(ax_f),
                  Write(t_hdr), Write(f_hdr),
                  Write(t_lbl), Write(f_lbl), Write(f_ylbl))

        prev_group = None

        for name, win_fn, color, sl_db in WINDOWS:
            w   = win_fn(N)
            W   = np.fft.fftshift(np.fft.fft(w, N_pad))
            W_db = 20 * np.log10(np.abs(W) / np.abs(W).max() + 1e-12)
            k_idx  = np.arange(-N_pad // 2, N_pad // 2)
            x_bins = k_idx * N / N_pad
            mask   = np.abs(x_bins) <= 20

            t_graph = ax_t.plot_line_graph(
                n.tolist(), w.tolist(),
                line_color=color, add_vertex_dots=False, stroke_width=2.5)
            f_graph = ax_f.plot_line_graph(
                x_bins[mask].tolist(), np.clip(W_db[mask], -70, 5).tolist(),
                line_color=color, add_vertex_dots=False, stroke_width=2.5)
            sl_line = DashedLine(ax_f.c2p(-20, sl_db), ax_f.c2p(20, sl_db),
                                 color=color, stroke_width=1.5, dash_length=0.12)
            sl_tag  = Text(f"{sl_db} dB", font_size=14, color=color).next_to(
                          ax_f.c2p(20, sl_db), RIGHT, buff=0.08)
            info = VGroup(
                Text(name, font_size=24, color=color),
                Text(f"Peak side lobe: {sl_db} dB   |   Main lobe: {MAIN_LOBE[name]}",
                     font_size=18, color=color),
            ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.3)

            current_group = VGroup(t_graph, f_graph, sl_line, sl_tag, info)

            if prev_group is None:
                self.play(Create(t_graph), Create(f_graph),
                          Create(sl_line), Write(sl_tag), Write(info))
            else:
                self.play(FadeOut(prev_group), FadeIn(current_group), run_time=0.9)

            self.wait(2.5)
            prev_group = current_group


class WindowTradeoffScene(Scene):
    def construct(self):
        header = VGroup(
            Text("Window",    font_size=20, color=GREY_A),
            Text("Side lobe", font_size=20, color=GREY_A),
            Text("Main lobe", font_size=20, color=GREY_A),
        ).arrange(RIGHT, buff=1.4)

        rows = [header]
        for name, _, color, sl_db in WINDOWS:
            row = VGroup(
                Text(name,              font_size=19, color=color),
                Text(f"{sl_db} dB",     font_size=19, color=color),
                Text(MAIN_LOBE[name],   font_size=19, color=color),
            ).arrange(RIGHT, buff=1.4)
            rows.append(row)

        table = VGroup(*rows).arrange(DOWN, buff=0.28).center().shift(UP * 0.4)

        tradeoff = VGroup(
            Text("The fundamental trade-off  (Q4)", font_size=26, color=YELLOW),
            table,
            Text("→ Hann or Hamming for guitar analysis: good leakage suppression",
                 font_size=20, color=GREEN),
            Text("   with enough resolution to see individual harmonics  (Q5)",
                 font_size=20, color=GREEN),
        ).arrange(DOWN, buff=0.4).center()

        for mob in tradeoff:
            self.play(Write(mob), run_time=0.7)
            self.wait(0.35)
        self.wait(3)


class WindowMaskingScene(Scene):
    def construct(self):
        masking = VGroup(
            Text("Q1  Why does 916 Hz disappear with rectangular?", font_size=24, color=YELLOW),
            Text("A strong harmonic's side lobes (−13 dB for rectangular)", font_size=20),
            Text("are high enough to drown the weaker 916 Hz harmonic.", font_size=20, color=RED),
            Text("With Blackman (−57 dB side lobes) those lobes are", font_size=20),
            Text("44 dB lower → the weak harmonic emerges above them.  (Q3)", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).center()

        for mob in masking:
            self.play(Write(mob), run_time=0.65)
            self.wait(0.3)
        self.wait(3)
