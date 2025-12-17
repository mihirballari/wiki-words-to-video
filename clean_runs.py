"""
Clean up generated artifacts for a run.

Usage:
  python3 clean_runs.py --run-name bag [--remove-video]
"""

import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="Remove run folders and optional video output.")
    p.add_argument("--run-name", required=True, help="Run name (removes runs/<name>/).")
    p.add_argument(
        "--remove-video",
        action="store_true",
        help="Also remove <name>.mp4 from project root if present.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Do not prompt before deletion.",
    )
    return p.parse_args()


def confirm(msg: str) -> bool:
    ans = input(f"{msg} [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def main():
    args = parse_args()
    run_dir = os.path.abspath(os.path.join("runs", args.run_name))
    video_path = os.path.abspath(f"{args.run_name}.mp4")

    targets = []
    if os.path.exists(run_dir):
        targets.append(run_dir)
    if args.remove_video and os.path.exists(video_path):
        targets.append(video_path)

    if not targets:
        print("Nothing to remove.")
        return

    if not args.force:
        if not confirm(f"Delete {' ,'.join(targets)}?"):
            print("Aborted.")
            return

    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)
            print(f"Removed directory: {t}")
        else:
            try:
                os.remove(t)
                print(f"Removed file: {t}")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
