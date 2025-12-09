"""
Wrapper to run wikiclip into its own run folder and emit diagnostics.

Example:
  python3 run_wikiclip_with_diagnostics.py --run-name bag -- --term bag --seconds 5

This will:
  - create runs/bag/
  - run wikiclip with frames/ and centers.csv inside that folder (forcing --center-debug)
  - save the video to runs/bag/bag.mp4 (unless overridden)
  - run compute_center_error.py against that run (skipping duplicate frames)
"""

import argparse
import os
import subprocess
import sys
import shutil
import platform


def parse_args():
    p = argparse.ArgumentParser(
        description="Run wikiclip in an isolated folder and generate diagnostics.",
        add_help=False,
    )
    p.add_argument("--run-name", required=True, help="Name for this run (creates runs/<name>).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run folder if it exists.",
    )
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

    # Force center-debug to get centers.csv
    if "--center-debug" not in wikiclip_args:
        wikiclip_args.append("--center-debug")

    # Default output path: project root <name>.mp4 unless user provided one.
    out_set = any(arg == "--out" for arg in wikiclip_args)
    if not out_set:
        out_path = os.path.join(script_dir, f"{args.run_name}.mp4")
        wikiclip_args.extend(["--out", out_path])

    cmd = [sys.executable, wikiclip_path] + wikiclip_args
    print(f"[run] {' '.join(cmd)} (cwd={run_dir})")
    subprocess.run(cmd, check=True, cwd=run_dir)

    diag_cmd = [
        sys.executable,
        compute_path,
        "--run-dir",
        run_dir,
    ]
    print(f"[diagnostics] {' '.join(diag_cmd)}")
    subprocess.run(diag_cmd, check=True, cwd=script_dir)

    video_path = None
    if out_set:
        # Find --out value if provided explicitly
        if "--out" in wikiclip_args:
            out_idx = wikiclip_args.index("--out")
            if out_idx + 1 < len(wikiclip_args):
                cand = wikiclip_args[out_idx + 1]
                video_path = cand if os.path.isabs(cand) else os.path.abspath(os.path.join(script_dir, cand))
    else:
        video_path = os.path.join(script_dir, f"{args.run_name}.mp4")

    diag_dir = os.path.join(run_dir, "diagnostics")
    important = [
        video_path,
        os.path.join(diag_dir, "center_errors.png"),
        os.path.join(diag_dir, "area_trend.png"),
        os.path.join(diag_dir, "anomalies.txt"),
    ]
    open_files(important)

    print(f"Done. Run folder: {run_dir}")


if __name__ == "__main__":
    main()
