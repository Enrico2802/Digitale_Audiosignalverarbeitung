"""Renders Part1Scene, Part2Scene, ConclusionScene in parallel, then concatenates.

Run from the repository root:
    python dasp-labs-main/labs/week03/render_parallel.py
"""
import multiprocessing
import subprocess
import sys

import av

MANIM_FILE = "dasp-labs-main/labs/week03/task03_manim.py"
SCENES     = ["Part1Scene", "Part2Scene", "ConclusionScene"]
QUALITY    = "2160p60"
MEDIA_DIR  = "media/videos/task03_manim"
OUTPUT     = f"{MEDIA_DIR}/{QUALITY}/ResolutionVsZeroPadding_final.mp4"


def render(scene_name):
    print(f"[START] {scene_name}")
    result = subprocess.run(
        ["manim", "-qk", MANIM_FILE, scene_name],
        capture_output=True, text=True,
    )
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {scene_name}")
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode


def concat_pyav(parts, output):
    import os
    import tempfile

    abs_parts  = [os.path.abspath(p) for p in parts]
    abs_output = os.path.abspath(output)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list = f.name
        for p in abs_parts:
            f.write(f"file '{p}'\n")

    try:
        inp = av.open(concat_list, format="concat", options={"safe": "0"})
        in_stream = inp.streams.video[0]

        out_container = av.open(abs_output, "w")
        out_stream = out_container.add_stream_from_template(template=in_stream)

        for packet in inp.demux(in_stream):
            if packet.dts is None:
                continue
            packet.dts = None
            packet.stream = out_stream
            out_container.mux(packet)

        inp.close()
        out_container.close()
    finally:
        os.remove(concat_list)


if __name__ == "__main__":
    print(f"Rendering {len(SCENES)} scenes in parallel...\n")

    with multiprocessing.Pool(len(SCENES)) as pool:
        codes = pool.map(render, SCENES)

    if any(c != 0 for c in codes):
        print("\nOne or more scenes failed — aborting concat.")
        sys.exit(1)

    parts = [f"{MEDIA_DIR}/{QUALITY}/{s}.mp4" for s in SCENES]

    print("\nCombining scenes with PyAV...")
    concat_pyav(parts, OUTPUT)
    print(f"\nDone: {OUTPUT}")
