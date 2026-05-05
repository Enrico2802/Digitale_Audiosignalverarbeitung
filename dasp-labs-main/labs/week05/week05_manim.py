"""Manim animations for Week 05 — FIR filters, sinusoid probing, magnitude response.

Parallel render (recommended):
    python dasp-labs-main/labs/week05/render_parallel.py

Single scene:
    manim -qk dasp-labs-main/labs/week05/week05_manim.py FIRConceptScene
    manim -qk dasp-labs-main/labs/week05/week05_manim.py SinusoidProbeScene
    manim -qk dasp-labs-main/labs/week05/week05_manim.py MagnitudeResponseScene
"""
from manim import *
import numpy as np

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width  = 3840

FS    = 48000
B_BOX = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apply_fir(b, x):
    P = len(b)
    y = np.zeros(len(x))
    for n in range(len(x)):
        for p in range(P):
            if n - p >= 0:
                y[n] += b[p] * x[n - p]
    return y


def analytical_magnitude(b, freqs_hz, fs=FS):
    result = np.zeros(len(freqs_hz))
    for i, f in enumerate(freqs_hz):
        omega = 2 * np.pi * f / fs
        h = sum(b[p] * np.exp(-1j * omega * p) for p in range(len(b)))
        result[i] = abs(h)
    return result


def measure_gain(b, freq_hz, fs=FS):
    duration = 0.1
    t = np.arange(int(duration * fs)) / fs
    x = np.sin(2 * np.pi * freq_hz * t)
    y = apply_fir(b, x)
    p = len(b)
    return np.max(np.abs(y[p:])) / np.max(np.abs(x[p:]))


# ---------------------------------------------------------------------------
# Scene 1 — FIR formula and DC probing
# ---------------------------------------------------------------------------

class FIRConceptScene(Scene):
    def construct(self):
        title = Text("FIR Filter — The Formula", font_size=38, color=YELLOW)
        self.play(Write(title))
        self.wait(0.4)
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Formula
        formula = Text(
            "y[n] = b₀x[n] + b₁x[n-1] + b₂x[n-2] + ⋯ + b_{P-1}x[n-P+1]",
            font_size=34,
        ).shift(UP * 1.5)
        self.play(Write(formula), run_time=1.2)
        self.wait(0.5)

        # 5-tap box average coefficients visualized as blocks
        coeff_title = Text("5-tap box average:  b = [0.2, 0.2, 0.2, 0.2, 0.2]",
                           font_size=26, color=BLUE).shift(UP * 0.3)
        self.play(Write(coeff_title))
        self.wait(0.3)

        blocks = VGroup(*[
            VGroup(
                Square(side_length=0.9, color=BLUE, fill_color=BLUE, fill_opacity=0.25),
                Text("0.2", font_size=22, color=WHITE),
            ).arrange(ORIGIN)
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.15).shift(DOWN * 0.5)

        delay_labels = VGroup(*[
            Text(f"b[{i}]", font_size=18, color=BLUE_B).next_to(blocks[i], DOWN, buff=0.1)
            for i in range(5)
        ])

        self.play(LaggedStartMap(FadeIn, blocks, lag_ratio=0.15))
        self.play(LaggedStartMap(Write, delay_labels, lag_ratio=0.12))
        self.wait(0.5)

        sum_text = Text("∑ bᵢ = 1.0  ⟹  DC gain = 1",
                        font_size=30, color=GREEN).shift(DOWN * 1.8)
        self.play(Write(sum_text))
        self.wait(0.4)

        # DC input demo
        dc_demo = VGroup(
            Text("Input:  x[n] = 1.0  (constant)", font_size=24, color=WHITE),
            Text("Output: y[n] = 0.2·1 + 0.2·1 + 0.2·1 + 0.2·1 + 0.2·1 = 1.0", font_size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).shift(DOWN * 2.8)
        self.play(Write(dc_demo[0]))
        self.wait(0.3)
        self.play(Write(dc_demo[1]))
        self.wait(2.5)

        self.play(FadeOut(VGroup(formula, coeff_title, blocks, delay_labels,
                                 sum_text, dc_demo)))
        self.wait(0.2)

        # Linearity: x = 2 → y = 2
        lin_title = Text("Linearity — doubling the input doubles the output",
                         font_size=28, color=ORANGE).shift(UP * 1.2)
        lin_eq = VGroup(
            Text("x[n] = 2.0  ⟹  y[n] = 0.2 · 5 · 2.0 = 2.0",
                 font_size=30, color=WHITE),
            Text("Ratio:  y / x = 1.0  (gain unchanged)",
                 font_size=28, color=GREEN),
        ).arrange(DOWN, buff=0.25)
        self.play(Write(lin_title))
        self.play(Write(lin_eq))
        self.wait(3)


# ---------------------------------------------------------------------------
# Scene 2 — Sinusoid probing: low vs high frequency
# ---------------------------------------------------------------------------

class SinusoidProbeScene(Scene):
    def construct(self):
        title = Text("Sinusoid Probing — Low vs High Frequency", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.4)
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Compute signals
        n_samples = 200
        t = np.arange(n_samples) / FS
        f_low, f_high = 200.0, 4000.0
        x_low  = np.sin(2 * np.pi * f_low  * t)
        x_high = np.sin(2 * np.pi * f_high * t)
        y_low  = apply_fir(B_BOX, x_low)
        y_high = apply_fir(B_BOX, x_high)
        gain_low  = np.max(np.abs(y_low[5:]))  / np.max(np.abs(x_low[5:]))
        gain_high = np.max(np.abs(y_high[5:])) / np.max(np.abs(x_high[5:]))

        def make_axes_with_signal(signal, color, label, x_pos):
            ax = Axes(
                x_range=[0, n_samples, 50],
                y_range=[-1.3, 1.3, 0.5],
                x_length=4.5, y_length=2.5,
                axis_config={"color": GREY, "stroke_width": 1.5},
                tips=False,
            ).move_to(x_pos)
            pts = [ax.c2p(i, signal[i]) for i in range(n_samples)]
            curve = VMobject(color=color, stroke_width=2)
            curve.set_points_as_corners(pts)
            lbl = Text(label, font_size=18, color=color).next_to(ax, UP, buff=0.1)
            return VGroup(ax, curve, lbl)

        row_low = VGroup(
            make_axes_with_signal(x_low,  BLUE,  f"x_low (input {f_low:.0f} Hz)",      LEFT * 3.2 + UP * 1.0),
            make_axes_with_signal(y_low,  GREEN, f"y_low (output, gain={gain_low:.3f})", RIGHT * 1.8 + UP * 1.0),
        )
        row_high = VGroup(
            make_axes_with_signal(x_high, BLUE,   f"x_high (input {f_high:.0f} Hz)",       LEFT * 3.2 + DOWN * 1.8),
            make_axes_with_signal(y_high, ORANGE, f"y_high (output, gain={gain_high:.3f})", RIGHT * 1.8 + DOWN * 1.8),
        )

        arrow_low  = Arrow(row_low[0][0].get_right(),  row_low[1][0].get_left(),  buff=0.1, color=WHITE)
        arrow_high = Arrow(row_high[0][0].get_right(), row_high[1][0].get_left(), buff=0.1, color=WHITE)
        fir_lbl_low  = Text("FIR", font_size=18, color=WHITE).next_to(arrow_low,  UP, buff=0.05)
        fir_lbl_high = Text("FIR", font_size=18, color=WHITE).next_to(arrow_high, UP, buff=0.05)

        for mob in [row_low[0], arrow_low, fir_lbl_low, row_low[1]]:
            self.play(FadeIn(mob), run_time=0.45)
        self.wait(0.3)
        for mob in [row_high[0], arrow_high, fir_lbl_high, row_high[1]]:
            self.play(FadeIn(mob), run_time=0.45)
        self.wait(0.5)

        # Gain annotations
        gain_note = VGroup(
            Text(f"200 Hz gain  = {gain_low:.3f}  (nearly passes through)",
                 font_size=22, color=GREEN),
            Text(f"4000 Hz gain = {gain_high:.3f}  (strongly attenuated)",
                 font_size=22, color=ORANGE),
            Text("→ This is a LOW-PASS filter", font_size=24, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(DOWN, buff=0.4)

        for line in gain_note:
            self.play(Write(line), run_time=0.5)
            self.wait(0.2)
        self.wait(2.5)


# ---------------------------------------------------------------------------
# Scene 3 — Magnitude response: analytical curve + probing dots
# ---------------------------------------------------------------------------

class MagnitudeResponseScene(Scene):
    def construct(self):
        title = Text("Magnitude Response — Analytical vs Probing", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.4)
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Compute curves
        smooth_freqs = np.linspace(10, FS / 2 - 10, 400)
        test_freqs   = [50, 200, 400, 800, 1600, 3200, 6400, 12000, 23900]

        filters = {
            "5-tap average": (B_BOX,                      BLUE),
            "2-tap average": (np.array([0.5, 0.5]),        GREEN),
            "11-tap average":(np.full(11, 1.0/11),         ORANGE),
            "difference [1,-1]": (np.array([1.0, -1.0]),  RED),
        }

        ax = Axes(
            x_range=[0, FS / 2, 4000],
            y_range=[-0.05, 1.15, 0.25],
            x_length=9.5,
            y_length=5.0,
            axis_config={"color": GREY, "stroke_width": 1.5},
            x_axis_config={"numbers_to_include": np.arange(0, 25000, 4000)},
            y_axis_config={"numbers_to_include": np.arange(0, 1.25, 0.25)},
            tips=False,
        ).shift(DOWN * 0.3)

        x_label = Text("Frequency [Hz]", font_size=20, color=GREY).next_to(ax, DOWN, buff=0.2)
        y_label = Text("Gain |H|",       font_size=20, color=GREY).next_to(ax, LEFT, buff=0.2).rotate(PI/2)

        self.play(Create(ax), Write(x_label), Write(y_label))

        # Reference lines
        ref1 = ax.plot(lambda f: 1.0, color=GREY,   stroke_width=1, stroke_opacity=0.5)
        ref05 = ax.plot(lambda f: 0.5, color=YELLOW, stroke_width=1, stroke_opacity=0.5)
        lbl_ref1  = Text("|H|=1",    font_size=16, color=GREY  ).next_to(ax.c2p(500, 1.0),   RIGHT, buff=0.05)
        lbl_ref05 = Text("|H|=0.5",  font_size=16, color=YELLOW).next_to(ax.c2p(500, 0.5),   RIGHT, buff=0.05)
        self.play(Create(ref1), Create(ref05), Write(lbl_ref1), Write(lbl_ref05))
        self.wait(0.3)

        legend_items = []
        for name, (b, color) in filters.items():
            mag = analytical_magnitude(b, smooth_freqs)
            curve = ax.plot_line_graph(
                x_values=smooth_freqs, y_values=mag,
                line_color=color, stroke_width=2.5,
                add_vertex_dots=False,
            )
            gains_dots = [measure_gain(b, f) for f in test_freqs]
            dots = VGroup(*[
                Dot(ax.c2p(f, g), color=color, radius=0.07)
                for f, g in zip(test_freqs, gains_dots)
            ])
            self.play(Create(curve), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.08), run_time=0.5)
            legend_items.append(
                VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=3),
                       Text(name, font_size=18, color=color)).arrange(RIGHT, buff=0.12)
            )
            self.wait(0.15)

        legend = VGroup(*legend_items).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_edge(RIGHT, buff=0.3).shift(UP * 0.5)
        self.play(FadeIn(legend))
        self.wait(0.5)

        # Annotation: probing == analytical
        note = VGroup(
            Text("Dots (sinusoidal probing) land on the analytical curve:", font_size=22, color=WHITE),
            Text("ĝ(f) = |H(e^(jΩ))| = |∑ₚ bₚ · e^(-jΩp)|",
                 font_size=28, color=YELLOW),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.35)
        self.play(Write(note[0]), run_time=0.5)
        self.play(Write(note[1]), run_time=0.7)
        self.wait(3)
