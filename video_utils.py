
# video_utils.py
import os
import subprocess
import time

def build_video_from_frames(frames_dir, fps, out_path):
    """Call ffmpeg to build an MP4 from frames in `frames_dir` and return (seconds, size_bytes)."""
    if os.path.exists(out_path):
        os.remove(out_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]

    print("\n[ffmpeg] Building video...")
    print(" ".join(cmd))

    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start

    size_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    return elapsed, size_bytes
