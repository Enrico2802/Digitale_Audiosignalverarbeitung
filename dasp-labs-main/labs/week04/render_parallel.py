"""Generic parallel renderer for week04 Manim tasks.

Run from the repository root:
    python dasp-labs-main/labs/week04/render_parallel.py task01
    python dasp-labs-main/labs/week04/render_parallel.py task02
    python dasp-labs-main/labs/week04/render_parallel.py task03
    python dasp-labs-main/labs/week04/render_parallel.py task04
"""
import multiprocessing
import os
import subprocess
import sys
import tempfile

import av

WEEK_DIR = "dasp-labs-main/labs/week04"
QUALITY  = "2160p60"

TASK_SCENES = {
    "task01": ["STFTConceptScene",    "NbComparisonScene",       "STFTTradeoffScene"],
    "task02": ["SpeechSpectrogramScene", "GuitarSpectrogramScene", "SpeechGuitarCompareScene"],
    "task03": ["SpectrumBinsScene",   "IrfftReconScene",         "RfftSymmetryScene"],
    "task04": ["FilterPipelineScene", "FrequencyMaskScene",      "BrickWallScene"],
}


def render(args):
    task, scene = args
    manim_file  = f"{WEEK_DIR}/{task}_manim.py"
    print(f"[START] {scene}")
    result = subprocess.run(
        ["manim", "-qk", manim_file, scene],
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
    if len(sys.argv) != 2 or sys.argv[1] not in TASK_SCENES:
        print("Usage: python render_parallel.py <task>")
        print(f"Tasks: {', '.join(TASK_SCENES)}")
        sys.exit(1)

    task      = sys.argv[1]
    scenes    = TASK_SCENES[task]
    media_dir = f"media/videos/{task}_manim/{QUALITY}"
    output    = f"{media_dir}/{task}_final.mp4"

    print(f"Rendering {len(scenes)} scenes in parallel ({task})...\n")

    with multiprocessing.Pool(len(scenes)) as pool:
        codes = pool.map(render, [(task, s) for s in scenes])

    if any(c != 0 for c in codes):
        print("\nOne or more scenes failed — aborting concat.")
        sys.exit(1)

    parts = [f"{media_dir}/{s}.mp4" for s in scenes]

    print("\nCombining scenes with PyAV...")
    concat_pyav(parts, output)
    print(f"\nDone: {output}")
