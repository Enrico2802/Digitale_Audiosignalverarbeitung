"""Parallel renderer for week05 Manim animations.

Run from the repository root:
    python dasp-labs-main/labs/week05/render_parallel.py

Or render a single scene:
    manim -qk dasp-labs-main/labs/week05/week05_manim.py FIRConceptScene
    manim -qk dasp-labs-main/labs/week05/week05_manim.py SinusoidProbeScene
    manim -qk dasp-labs-main/labs/week05/week05_manim.py MagnitudeResponseScene
"""
import multiprocessing
import os
import subprocess
import sys
import tempfile

import av

WEEK_DIR = "dasp-labs-main/labs/week05"
MANIM_FILE = f"{WEEK_DIR}/week05_manim.py"
QUALITY    = "2160p60"

SCENES = [
    "FIRConceptScene",
    "SinusoidProbeScene",
    "MagnitudeResponseScene",
]


def render(scene):
    print(f"[START] {scene}")
    result = subprocess.run(
        ["manim", "-qk", MANIM_FILE, scene],
        capture_output=True, text=True,
    )
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {scene}")
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode


def concat_pyav(parts, output):
    abs_parts  = [os.path.abspath(p) for p in parts]
    abs_output = os.path.abspath(output)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list = f.name
        for p in abs_parts:
            f.write(f"file '{p}'\n")

    try:
        inp        = av.open(concat_list, format="concat", options={"safe": "0"})
        in_stream  = inp.streams.video[0]
        out        = av.open(abs_output, "w")
        out_stream = out.add_stream_from_template(template=in_stream)

        for packet in inp.demux(in_stream):
            if packet.dts is None:
                continue
            packet.dts    = None
            packet.stream = out_stream
            out.mux(packet)

        inp.close()
        out.close()
    finally:
        os.remove(concat_list)


if __name__ == "__main__":
    media_dir = f"media/videos/week05_manim/{QUALITY}"
    output    = f"{media_dir}/week05_final.mp4"

    print(f"Rendering {len(SCENES)} scenes in parallel...\n")

    with multiprocessing.Pool(len(SCENES)) as pool:
        codes = pool.map(render, SCENES)

    if any(c != 0 for c in codes):
        print("\nOne or more scenes failed — aborting concat.")
        sys.exit(1)

    parts = [f"{media_dir}/{s}.mp4" for s in SCENES]

    print("\nCombining scenes with PyAV...")
    concat_pyav(parts, output)
    print(f"\nDone: {output}")
