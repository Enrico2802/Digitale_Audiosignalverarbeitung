"""Manim animation for Task 3 — IDFT and the rfft Round-Trip.

Parallel render (recommended):
    python dasp-labs-main/labs/week04/render_parallel.py task03

Single scene:
    manim -qk dasp-labs-main/labs/week04/task03_manim.py SpectrumBinsScene
"""
from manim import *
import numpy as np

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width  = 3840

FS = 8000
N  = 512


def make_demo_signal():
    t = np.arange(N) / FS
    return (np.sin(2 * np.pi * 440 * t) +
            0.6 * np.sin(2 * np.pi * 880 * t) +
            0.3 * np.sin(2 * np.pi * 1320 * t))


class SpectrumBinsScene(Scene):
    def construct(self):
        title = Text("From Spectrum Back to Signal — irfft", font_size=34, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x      = make_demo_signal()
        X      = np.fft.rfft(x)
        freqs  = np.fft.rfftfreq(N, d=1 / FS)
        mag_db = 20 * np.log10(np.abs(X) + 1e-9)
        mag_db -= mag_db.max()

        mask = freqs <= 2000
        f_show = freqs[mask]
        m_show = np.clip(mag_db[mask], -60, 0)

        ax = Axes(
            x_range=[0, 2000, 400], y_range=[-65, 5, 20],
            x_length=10, y_length=4, tips=False,
        ).shift(DOWN * 0.3)
        x_lbl = Text("Frequency [Hz]",  font_size=16).next_to(ax, DOWN, buff=0.1)
        y_lbl = Text("Magnitude [dB]",  font_size=16).next_to(ax, LEFT, buff=0.05)
        hdr   = Text("rfft magnitude spectrum of demo signal",
                     font_size=18, color=WHITE).next_to(ax, UP, buff=0.05)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl), Write(hdr))

        stems = VGroup()
        for f, m in zip(f_show, m_show):
            stems.add(Line(ax.c2p(f, -65), ax.c2p(f, m),
                           color=BLUE_B, stroke_width=1.8))

        self.play(Create(stems), run_time=1.5)

        for f_peak, label, col in [(440, "440 Hz", GREEN),
                                    (880, "880 Hz", ORANGE),
                                    (1320, "1320 Hz", RED)]:
            peak_m = float(mag_db[np.argmin(np.abs(freqs - f_peak))])
            dot = Dot(ax.c2p(f_peak, np.clip(peak_m, -60, 0)), color=col, radius=0.08)
            lbl = Text(label, font_size=15, color=col).next_to(dot, UP, buff=0.1)
            self.play(FadeIn(dot), Write(lbl), run_time=0.5)

        note = Text("rfft returns N/2 + 1 complex bins  (N=512 → 257 bins)",
                    font_size=19, color=YELLOW).to_edge(DOWN, buff=0.3)
        self.play(Write(note))
        self.wait(3)


class IrfftReconScene(Scene):
    def construct(self):
        title = Text("irfft — Reconstructing the Time-Domain Signal", font_size=32, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x     = make_demo_signal()
        X     = np.fft.rfft(x)
        x_rec = np.fft.irfft(X, n=N)

        t     = np.arange(N) / FS
        t_ds  = t[::4]
        x_ds  = x[::4]
        r_ds  = x_rec[::4]

        ax = Axes(
            x_range=[0, N / FS, 0.016],
            y_range=[-2.0, 2.0, 0.5],
            x_length=10, y_length=4, tips=False,
        ).shift(DOWN * 0.2)
        x_lbl = Text("Time [s]",     font_size=16).next_to(ax, DOWN, buff=0.1)
        y_lbl = Text("Amplitude",    font_size=16).next_to(ax, LEFT, buff=0.05)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl))

        orig_graph = ax.plot_line_graph(t_ds.tolist(), x_ds.tolist(),
                         line_color=BLUE, add_vertex_dots=False, stroke_width=2.0)
        orig_lbl   = Text("Original  x", font_size=18, color=BLUE).to_edge(DOWN, buff=0.9)
        self.play(Create(orig_graph), Write(orig_lbl))
        self.wait(1)

        rec_graph = ax.plot_line_graph(t_ds.tolist(), r_ds.tolist(),
                        line_color=GREEN, add_vertex_dots=False, stroke_width=1.5,
                        stroke_opacity=0.7)
        rec_lbl   = Text("Reconstructed  irfft(rfft(x))", font_size=18, color=GREEN).to_edge(DOWN, buff=0.5)
        self.play(Create(rec_graph), Write(rec_lbl))
        self.wait(0.8)

        match = np.allclose(x, x_rec)
        result_txt = Text(
            f"np.allclose(x, irfft(rfft(x))) = {match}",
            font_size=22, color=GREEN if match else RED,
        ).to_edge(DOWN, buff=0.15)
        self.play(Write(result_txt))
        self.wait(3)


class RfftSymmetryScene(Scene):
    def construct(self):
        explanation = VGroup(
            Text("Why rfft returns only N/2 + 1 bins", font_size=32, color=YELLOW),
            Text("For a real-valued signal x[n], the DFT has Hermitian symmetry:", font_size=21),
            Text("  X[k]  =  X*[N - k]   (complex conjugate)", font_size=21, color=BLUE_B),
            Text("The upper half of the spectrum is a mirror of the lower half.", font_size=21),
            Text("→ rfft stores only bins 0 … N/2  (the unique information).", font_size=21, color=GREEN),
            Text("→ irfft uses that symmetry to reconstruct all N real samples.", font_size=21, color=GREEN),
            Text("Example:  N = 512  →  rfft gives 257 complex bins", font_size=21, color=ORANGE),
            Text("          each bin covers  Δf = fs / N  =  8000 / 512 ≈ 15.6 Hz", font_size=21, color=ORANGE),
            Text("Round-trip:  irfft(rfft(x)) == x  to floating-point precision.", font_size=21, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).center()

        for mob in explanation:
            self.play(Write(mob), run_time=0.65)
            self.wait(0.25)
        self.wait(3)
