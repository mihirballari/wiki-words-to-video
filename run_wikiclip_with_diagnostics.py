"""
Wrapper to run wikiclip into its own run folder and emit diagnostics, or re-encode existing frames.
"""

import argparse
import os
import subprocess
import sys
import shutil
import platform
from typing import Optional, Tuple

from zoom_utils import resolve_zoom_range
from video_utils import build_video_from_frames
from center_diagnostics import run_full_diagnostics
from PIL import Image

HELP_TEXT = """
Usage:
  Fresh capture + diagnostics
    python3 run_wikiclip_with_diagnostics.py --run-name NAME -- [wikiclip args...]
    e.g. -- --term bag --seconds 5 --fps 10 --zoom 1.8 --zoom-end 2.5

  Re-encode existing frames (no scraping)
    python3 run_wikiclip_with_diagnostics.py --run-name NAME --encode-only [options]
    options: --frames-dir <dir> --centers <csv> --fps 10 --zoom-start 1.0 --zoom-end 1.5
             --zoom-in 1.3 --zoom-out 1.3 --crf 18 --preset slow --out output.mp4

  Clean generated artifacts
    python3 clean_runs.py --run-name NAME [--remove-video] [--force]

  Standalone diagnostics on an existing run
    python3 compute_center_error.py --run-dir runs/NAME [--keep-duplicates]

Wrapper flags (put before '--'):
  --run-name NAME       Required. Uses runs/NAME for frames/centers/diagnostics.
  --force               Reuse existing runs/NAME (does not pass to wikiclip).
  --encode-only         Skip wikiclip; re-encode existing frames + diagnostics.
  --frames-dir PATH     Frames for encode-only (default runs/NAME/frames).
  --centers PATH        centers.csv for encode-only (default runs/NAME/centers.csv).
  --fps INT             FPS when encoding existing frames (default 10).
  --zoom-start FLOAT    Starting zoom for encode-only (default 1.0).
  --zoom-end FLOAT      Ending zoom for encode-only.
  --zoom-in FLOAT       Multiplier to zoom in across the video.
  --zoom-out FLOAT      Divider to zoom out across the video.
  --crf STR             ffmpeg CRF (default 18).
  --preset STR          ffmpeg preset (default slow).
  --out PATH            Output mp4 (default <repo>/<run-name>.mp4).
  --help, -h            Show this help.

Notes:
  - Everything after '--' is passed directly to wikiclip.py.
  - The wrapper always sets --center-debug so diagnostics have centers.csv.
  - Videos default to the repo root; runs/NAME holds frames, centers.csv, diagnostics/.
""".strip()


def parse_args():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_TEXT)
        raise SystemExit(0)

    p = argparse.ArgumentParser(
        description="Run wikiclip in an isolated folder and generate diagnostics, or re-encode existing frames.",
        add_help=False,
    )
    p.add_argument("-h", "--help", action="help", help="Show help and exit.")
    p.add_argument("--run-name", required=True, help="Name for this run (creates runs/<name>).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run folder if it exists.",
    )
    p.add_argument(
        "--encode-only",
        action="store_true",
        help="Skip wikiclip; re-encode existing frames + diagnostics.",
    )
    p.add_argument("--frames-dir", help="Frames directory (encode-only). Defaults to runs/<name>/frames.")
    p.add_argument("--centers", help="centers.csv path (encode-only). Defaults to runs/<name>/centers.csv.")
    p.add_argument("--fps", type=int, help="FPS for encoding (encode-only; default 10).")
    p.add_argument("--zoom-start", type=float, help="Start zoom for encode-only (default 1.0).")
    p.add_argument("--zoom-end", type=float, help="End zoom for encode-only.")
    p.add_argument("--zoom-in", type=float, help="Zoom-in multiplier for encode-only.")
    p.add_argument("--zoom-out", type=float, help="Zoom-out multiplier for encode-only.")
    p.add_argument("--crf", help="CRF for encode-only (default 18).")
    p.add_argument("--preset", help="Preset for encode-only (default slow).")
    p.add_argument("--out", help="Output mp4 path (default: <project-root>/<run-name>.mp4).")
    p.add_argument(
        "--",
        dest="sep",
        action="store_true",
        help="Separator; everything after is passed to wikiclip.",
    )
    args, remainder = p.parse_known_args()
    # argparse with custom separator leaves the remaining args in remainder
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    return args, remainder


def ensure_run_dir(path: str, force: bool):
    if os.path.exists(path):
        if force:
            return
        raise SystemExit(f"Run folder already exists: {path} (use --force to reuse)")
    os.makedirs(path, exist_ok=True)


def strip_unsupported_wikiclip_args(args: list) -> list:
    """Remove flags that belong to the wrapper, not wikiclip."""
    cleaned = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--force":
            print("[note] Removed unsupported flag for wikiclip: --force (wrapper handles run reuse).")
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def frame_size_from_dir(frames_dir: str) -> Optional[Tuple[int, int]]:
    first = next((p for p in sorted(os.listdir(frames_dir)) if p.startswith("frame_") and p.endswith(".png")), None)
    if not first:
        return None
    fp = os.path.join(frames_dir, first)
    with Image.open(fp) as img:
        return img.size


def open_files(paths):
    """Best-effort opener for macOS/Linux."""
    existing = [p for p in paths if p and os.path.exists(p)]
    if not existing:
        return
    opener = None
    if platform.system() == "Darwin":
        opener = "open"
    elif shutil.which("xdg-open"):
        opener = "xdg-open"
    if not opener:
        return
    for p in existing:
        try:
            subprocess.Popen([opener, p])
        except Exception:
            pass


def main():
    args, wikiclip_args = parse_args()
    run_dir = os.path.abspath(os.path.join("runs", args.run_name))
    ensure_run_dir(run_dir, args.force)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    wikiclip_path = os.path.join(script_dir, "wikiclip.py")
    compute_path = os.path.join(script_dir, "compute_center_error.py")

    wikiclip_args = strip_unsupported_wikiclip_args(wikiclip_args)

    # Default output path: project root <name>.mp4 unless user provided one.
    out_path = args.out or os.path.join(script_dir, f"{args.run_name}.mp4")

    if args.encode_only:
        frames_dir = args.frames_dir or os.path.join(run_dir, "frames")
        centers_path = args.centers or os.path.join(run_dir, "centers.csv")
        fps = args.fps or 10
        zoom_start = args.zoom_start or 1.0
        zoom_start, zoom_end = resolve_zoom_range(
            zoom_start, args.zoom_end, args.zoom_in, args.zoom_out
        )
        crf = args.crf or "18"
        preset = args.preset or "slow"

        size = frame_size_from_dir(frames_dir)
        if not size:
            raise SystemExit(f"No frames found in {frames_dir}")

        print(f"[encode-only] frames={frames_dir}, fps={fps}, zoom {zoom_start:.2f}->{zoom_end:.2f}, out={out_path}")
        build_video_from_frames(
            frames_dir,
            fps,
            out_path,
            frame_size=size,
            zoom_start=zoom_start,
            zoom_end=zoom_end,
            crf=crf,
            preset=preset,
        )
    else:
        # Force center-debug to get centers.csv
        if "--center-debug" not in wikiclip_args:
            wikiclip_args.append("--center-debug")

        # Ensure --out is set for wikiclip
        out_set = any(arg == "--out" for arg in wikiclip_args)
        if not out_set:
            wikiclip_args.extend(["--out", out_path])

        cmd = [sys.executable, wikiclip_path] + wikiclip_args
        print(f"[run] {' '.join(cmd)} (cwd={run_dir})")
        subprocess.run(cmd, check=True, cwd=run_dir)

    print(f"[diagnostics] analyzing frames in {run_dir}")
    summary = run_full_diagnostics(
        frames_dir=os.path.join(run_dir, "frames"),
        centers_csv=os.path.join(run_dir, "centers.csv"),
        out_dir=os.path.join(run_dir, "diagnostics"),
        skip_duplicates=True,
    )

    video_path = None
    video_path = out_path

    diag_dir = os.path.join(run_dir, "diagnostics")
    important = [
        video_path,
        os.path.join(diag_dir, "center_errors.png"),
        os.path.join(diag_dir, "area_trend.png"),
        os.path.join(diag_dir, "anomalies.txt"),
    ]
    open_files(important)

    suggestions = []
    if summary:
        anomaly_frames = summary.get("center_anomalies", []) or []
        anomaly_paths = [
            os.path.join(run_dir, "frames", f"frame_{idx:05d}.png")
            for idx in anomaly_frames
        ]
        print(
            f"[done] video={video_path}\n"
            f"       frames analyzed: {summary.get('frames_analyzed', 0)} "
            f"(skipped duplicates: {summary.get('skipped_frames', 0)})\n"
            f"       mean center error: {summary.get('mean_center_error_pct', 0):.2f}% "
            f"max: {summary.get('max_center_error_pct', 0):.2f}%\n"
            f"       center anomalies: {anomaly_frames}\n"
            f"       area anomalies: {summary.get('area_anomalies', [])}\n"
            f"       anomalies file: {summary.get('anomalies_path')}"
        )
        if anomaly_paths:
            print("       anomaly frame paths:")
            for p in anomaly_paths:
                print(f"         - {p}")
            suggestions.append("Inspect anomaly frames above and remove or re-encode with --encode-only.")
        if summary.get("area_anomalies"):
            suggestions.append("Check area anomalies; consider adjusting zoom range or cleaning frames.")
        if summary.get("mean_center_error_pct", 0) > 5:
            suggestions.append("Center error is high; try re-run with --max-pages higher or adjust zoom.")
    else:
        suggestions.append("Diagnostics missing; ensure centers.csv exists and re-run.")

    if suggestions:
        print("[next] Suggested follow-ups:")
        for s in suggestions:
            print(f"       - {s}")
    print(f"Done. Run folder: {run_dir}")


if __name__ == "__main__":
    main()
