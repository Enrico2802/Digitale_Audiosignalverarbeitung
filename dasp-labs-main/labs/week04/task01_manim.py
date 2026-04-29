"""Manim animation for Task 1 — STFT: Time-Frequency Tradeoff.

Parallel render (recommended):
    python dasp-labs-main/labs/week04/render_parallel.py task01

Single scene:
    manim -qk dasp-labs-main/labs/week04/task01_manim.py STFTConceptScene
"""
from manim import *
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

config.frame_rate = 60
config.pixel_height = 2160
config.pixel_width  = 3840

FS       = 8000
DURATION = 1.0


def make_signal():
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    return np.where(t < 0.5,
                    np.sin(2 * np.pi * 440.0 * t),
                    np.sin(2 * np.pi * 880.0 * t)), t


def compute_stft(x, N_b, hop=None):
    if hop is None:
        hop = N_b // 4
    window = np.hanning(N_b)
    frames, times = [], []
    for start in range(0, len(x) - N_b + 1, hop):
        frames.append(np.abs(np.fft.rfft(x[start:start + N_b] * window)))
        times.append((start + N_b // 2) / FS)
    S     = np.array(frames).T
    S_db  = 20 * np.log10(S + 1e-9)
    S_db -= S_db.max()
    freqs = np.fft.rfftfreq(N_b, d=1 / FS)
    return np.array(times), freqs, S_db


def spectrogram_image(S_db, vmin=-80, vmax=0, max_freq_bin=None):
    if max_freq_bin is not None:
        S_db = S_db[:max_freq_bin, :]
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = (cm.inferno(norm(S_db)) * 255).astype(np.uint8)
    return rgba[::-1, :, :3]


class STFTConceptScene(Scene):
    def construct(self):
        title = Text("STFT — Short-Time Fourier Transform", font_size=36, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x, t_arr = make_signal()

        ax_t = Axes(
            x_range=[0, 1, 0.25], y_range=[-1.3, 1.3, 0.5],
            x_length=10, y_length=2.5, tips=False,
        ).shift(UP * 1.4)
        t_lbl = Text("Time [s]",      font_size=16).next_to(ax_t, DOWN, buff=0.05)
        t_hdr = Text("Signal x(t): 440 Hz → 880 Hz", font_size=18, color=WHITE).next_to(ax_t, UP, buff=0.05)

        downsample = 8
        t_ds = t_arr[::downsample]
        x_ds = x[::downsample]
        signal_graph = ax_t.plot_line_graph(
            t_ds.tolist(), x_ds.tolist(),
            line_color=BLUE_B, add_vertex_dots=False, stroke_width=1.5,
        )
        self.play(Create(ax_t), Write(t_lbl), Write(t_hdr))
        self.play(Create(signal_graph), run_time=1.2)
        self.wait(0.4)

        N_b  = 512
        hop  = N_b // 4
        win_duration = N_b / FS

        window_rect = Rectangle(
            width=ax_t.x_length * win_duration,
            height=ax_t.y_length * 0.85,
            color=YELLOW, fill_color=YELLOW, fill_opacity=0.18,
        ).move_to(ax_t.c2p(win_duration / 2, 0))

        win_lbl = Text(f"Hann window  N_b={N_b}", font_size=17, color=YELLOW).next_to(
            window_rect, DOWN, buff=0.15)

        self.play(FadeIn(window_rect), Write(win_lbl))
        self.wait(0.5)

        positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        for pos in positions:
            target_x = pos + win_duration / 2
            self.play(
                window_rect.animate.move_to(ax_t.c2p(target_x, 0)),
                win_lbl.animate.next_to(ax_t.c2p(target_x, -1.3), DOWN, buff=0.05),
                run_time=0.5,
            )
            self.wait(0.15)

        explanation = VGroup(
            Text("Each window position → one FFT → one column of the spectrogram",
                 font_size=20, color=WHITE),
            Text("Stack all columns side by side = Spectrogram", font_size=20, color=YELLOW),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.35)

        self.play(Write(explanation[0]))
        self.wait(0.8)
        self.play(Write(explanation[1]))
        self.wait(2.5)


class NbComparisonScene(Scene):
    def construct(self):
        title = Text("Effect of Frame Length N_b on the Spectrogram",
                     font_size=30, color=YELLOW)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_edge(UP))

        x, _ = make_signal()
        nb_configs = [
            (128,  RED,    "N_b=128\nΔf=62.5 Hz  Δt=16 ms\n→ blurry freq, sharp time"),
            (512,  GREEN,  "N_b=512\nΔf=15.6 Hz  Δt=64 ms\n→ good balance  ✓"),
            (2048, BLUE,   "N_b=2048\nΔf=3.9 Hz   Δt=256 ms\n→ sharp freq, smeared time"),
        ]

        max_freq_idx = int(1500 / (FS / 2) * (128 // 2 + 1))

        images_group = Group()
        for i, (N_b, color, label_text) in enumerate(nb_configs):
            _, freqs, S_db = compute_stft(x, N_b)
            cap = np.searchsorted(freqs, 1500)
            img_arr = spectrogram_image(S_db[:cap, :])
            img = ImageMobject(img_arr).set_height(2.8)

            border = Rectangle(
                width=img.width + 0.1, height=img.height + 0.1,
                color=color, stroke_width=3,
            ).move_to(img)
            lbl = Text(label_text, font_size=14, color=color).next_to(img, DOWN, buff=0.15)

            group = Group(img, border, lbl)
            images_group.add(group)

        images_group.arrange(RIGHT, buff=0.5).shift(UP * 0.2)

        x_axis_lbl = Text("← Time [s]  (0 → 1) →", font_size=14, color=GREY_A)
        y_axis_lbl = Text("↑ Freq [Hz] (0 → 1500)", font_size=14, color=GREY_A)
        x_axis_lbl.to_edge(DOWN, buff=0.05)
        y_axis_lbl.to_edge(LEFT, buff=0.05)

        for group in images_group:
            self.play(FadeIn(group), run_time=0.7)
            self.wait(1.2)

        self.play(Write(x_axis_lbl), Write(y_axis_lbl))
        self.wait(3)


class STFTTradeoffScene(Scene):
    def construct(self):
        summary = VGroup(
            Text("The Time-Frequency Tradeoff", font_size=34, color=YELLOW),
            Text("Δf = fs / N_b     (frequency resolution)", font_size=24),
            Text("Δt = N_b / fs     (time resolution of one frame)", font_size=24),
            Text("→  Large N_b :  fine frequency resolution,  poor time resolution",
                 font_size=22, color=BLUE),
            Text("→  Small N_b :  fine time resolution,       poor frequency resolution",
                 font_size=22, color=RED),
            Text("There is NO window that gives both simultaneously.",
                 font_size=24, color=RED),
            Text("This is the signal-processing analogue of the Heisenberg uncertainty principle.",
                 font_size=20, color=GREY_A),
            Text("Choose N_b based on what matters for YOUR signal.", font_size=22, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).center()

        for mob in summary:
            self.play(Write(mob), run_time=0.7)
            self.wait(0.3)
        self.wait(3)
