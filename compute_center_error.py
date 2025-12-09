
"""
Compute centering errors and produce diagnostics/ visualizations.

Run:
  python3 compute_center_error.py [--run-dir runs/bag] [--frames-dir ...] [--centers ...] [--out-dir ...]

Requires:
  - centers.csv (from wikiclip --center-debug)
  - frames/frame_XXXXX.png
"""

import argparse
import os

from center_diagnostics import run_full_diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="Analyze centering/zoom diagnostics.")
    p.add_argument(
        "--run-dir",
        default=".",
        help="Base directory for a run (defaults to current directory).",
    )
    p.add_argument(
        "--frames-dir",
        help="Frames directory; defaults to <run-dir>/frames.",
    )
    p.add_argument(
        "--centers",
        help="centers.csv path; defaults to <run-dir>/centers.csv.",
    )
    p.add_argument(
        "--out-dir",
        help="Diagnostics output directory; defaults to <run-dir>/diagnostics.",
    )
    p.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Analyze all frames even if duplicate content is detected.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    frames_dir = args.frames_dir or os.path.join(run_dir, "frames")
    centers = args.centers or os.path.join(run_dir, "centers.csv")
    out_dir = args.out_dir or os.path.join(run_dir, "diagnostics")
    run_full_diagnostics(
        frames_dir=frames_dir,
        centers_csv=centers,
        out_dir=out_dir,
        skip_duplicates=not args.keep_duplicates,
    )


if __name__ == "__main__":
    main()
