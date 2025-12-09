
# video_utils.py
import os
import subprocess
import time

from zoom_utils import build_zoompan_filter


def _count_frames(frames_dir: str) -> int:
    return len(
        [
            f
            for f in os.listdir(frames_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ]
    )


def build_video_from_frames(
    frames_dir,
    fps,
    out_path,
    *,
    frame_size=None,
    zoom_start: float = 1.0,
    zoom_end: float = 1.0,
    crf: str = "18",
    preset: str = "slow",
):
    """Call ffmpeg to build an MP4 from frames in `frames_dir` and return (seconds, size_bytes)."""
    if os.path.exists(out_path):
        os.remove(out_path)

    frame_count = _count_frames(frames_dir)
    width, height = (frame_size if frame_size else (None, None))
    zoom_filter = None
    if width and height:
        zoom_filter = build_zoompan_filter(
            frame_count, width, height, zoom_start, zoom_end, fps
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%05d.png"),
    ]

    if zoom_filter:
        print(
            f"[zoom] Interpolating zoom {zoom_start:.2f} -> {zoom_end:.2f} "
            f"across {frame_count} frames"
        )
        cmd.extend(["-vf", zoom_filter])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
    )

    print("\n[ffmpeg] Building video...")
    print(" ".join(cmd))

    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start

    size_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    return elapsed, size_bytes
