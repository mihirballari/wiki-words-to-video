"""
Create a short video for a sentence, showing <1s of footage per word.

Workflow:
  - For each word, fetch Wikipedia pages, grab frames highlighting that word.
  - Limit per-word duration (<1s) by capping frames based on fps.
  - Collect all frames into one folder, then encode with ffmpeg.
"""

import argparse
import math
import os
import random
import shutil
import re
from typing import List, Optional, Tuple

from playwright.sync_api import sync_playwright

from video_utils import build_video_from_frames
from zoom_utils import resolve_zoom_range
from wikiclip import (
    get_page_titles,
    save_center_debug,
    screenshot_page_with_term,
)


SYMBOL_MAP = {
    "+": "plus sign",
    "-": "minus sign",
    "−": "minus sign",
    "×": "multiplication sign",
    "*": "asterisk",
    "÷": "division sign",
    "/": "division slash",
    "=": "equals sign",
    "π": "pi",
    "√": "square root",
    "^": "exponent",
    "?": "question mark",
    "!": "exclamation mark",
}


def tokenize_sentence(sentence: str) -> List[Tuple[str, str]]:
    """
    Return list of (token, query) where token is the literal string to highlight,
    and query is what we use to search Wikipedia (mapped for symbols).
    """
    tokens = re.findall(r"[A-Za-z0-9]+|[^\\sA-Za-z0-9]", sentence)
    pairs = []
    for t in tokens:
        query = SYMBOL_MAP.get(t, t)
        pairs.append((t, query))
    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Make a video for a sentence with <1s per word using Wikipedia pages."
    )
    p.add_argument("--sentence", required=True, help="Sentence to visualize.")
    p.add_argument(
        "--word-seconds",
        type=float,
        default=0.8,
        help="Seconds per word (will be clamped to <1.0). Default: 0.8.",
    )
    p.add_argument(
        "--max-pages-per-word",
        type=int,
        default=15,
        help="Max Wikipedia pages to sample per word (default: 5).",
    )
    p.add_argument(
        "--frames-dir",
        default="sentence_frames",
        help="Directory to store captured frames (default: sentence_frames).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the video (default: 20).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output frame width (default: 1024).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=576,
        help="Output frame height (default: 576).",
    )
    p.add_argument(
        "--zoom",
        type=float,
        default=1.8,
        help="Page zoom factor during capture (default: 1.0).",
    )
    p.add_argument(
        "--zoom-in",
        type=float,
        help="Smoothly zoom IN over the final video by this multiplier.",
    )
    p.add_argument(
        "--zoom-out",
        type=float,
        help="Smoothly zoom OUT over the final video by this multiplier.",
    )
    p.add_argument(
        "--zoom-end",
        type=float,
        help="Final video zoom factor; alternative to zoom-in/out.",
    )
    p.add_argument(
        "--out",
        default="sentence.mp4",
        help="Output video filename (default: sentence.mp4).",
    )
    p.add_argument(
        "--center-debug",
        action="store_true",
        help="Collect centering stats and write centers.csv + centers.png.",
    )
    p.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Prefer case-sensitive term matching when highlighting.",
    )
    return p.parse_args()


def clamp_word_seconds(val: float) -> float:
    if val >= 1.0:
        return 0.95
    if val <= 0:
        return 0.8
    return val


def duplicate_frames(
    frames_dir: str,
    source_indices: List[int],
    target_count: int,
    centers: Optional[List[dict]],
    global_idx_start: int,
) -> int:
    """Duplicate existing word frames to reach target_count; return new total frames added."""
    added = 0
    idx = global_idx_start
    while len(source_indices) + added < target_count and source_indices:
        src_idx = random.choice(source_indices)
        src_path = os.path.join(frames_dir, f"frame_{src_idx:05d}.png")
        dst_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        shutil.copy2(src_path, dst_path)

        if centers:
            src_center = next((c for c in centers if c["frame_idx"] == src_idx), None)
            if src_center:
                clone = src_center.copy()
                clone["frame_idx"] = idx
                centers.append(clone)

        idx += 1
        added += 1
    return added


def main():
    args = parse_args()
    tokens = tokenize_sentence(args.sentence)
    if not tokens:
        print("No words found in the sentence.")
        return
    words = [t[0] for t in tokens]

    word_secs = clamp_word_seconds(args.word_seconds)
    frames_dir = args.frames_dir
    centers = [] if args.center_debug else None

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # For video zoom, start at 1.0 unless user overrides; page zoom (args.zoom) applies during capture.
    zoom_start, zoom_end = resolve_zoom_range(
        1.0, args.zoom_end, args.zoom_in, args.zoom_out
    )

    target_per_word = max(1, int(math.ceil(word_secs * args.fps)))
    print(f"Target frames per word: {target_per_word} (<= {word_secs:.2f}s at {args.fps} fps)")

    global_idx = 0
    used_words = 0

    with sync_playwright() as p:
        viewport_w = args.width * 2
        viewport_h = args.height * 2
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})

        for literal, query in tokens:
            titles = get_page_titles(query, args.max_pages_per_word)
            if not titles:
                print(f"[skip] No pages for '{literal}'")
                continue

            word_frames: List[int] = []
            print(f"[word] '{literal}' (query '{query}') -> {len(titles)} candidate pages")
            for title in titles:
                if len(word_frames) >= target_per_word:
                    break
                ok = screenshot_page_with_term(
                    page,
                    title,
                    literal,
                    global_idx,
                    frames_dir,
                    args.width,
                    args.height,
                    args.zoom,
                    centers=centers,
                    case_sensitive=args.case_sensitive,
                )
                if ok:
                    word_frames.append(global_idx)
                    global_idx += 1

            if len(word_frames) < target_per_word:
                added = duplicate_frames(
                    frames_dir,
                    word_frames,
                    target_per_word,
                    centers,
                    global_idx,
                )
                global_idx += added
                if added:
                    print(f"  -> duplicated {added} frames to reach {target_per_word} for 'w'")

            if word_frames:
                used_words += 1

        browser.close()

    total_frames = global_idx
    if total_frames == 0:
        print("No frames captured; nothing to encode.")
        return

    if args.center_debug and centers:
        save_center_debug(centers, args.width, args.height, csv_path="centers.csv", png_path="centers.png")

    encode_secs, size_bytes = build_video_from_frames(
        frames_dir,
        args.fps,
        args.out,
        frame_size=(args.width, args.height),
        zoom_start=zoom_start,
        zoom_end=zoom_end,
    )
    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0

    print("\n=== Sentence Summary ===")
    print(f"Words processed:   {used_words}/{len(words)}")
    print(f"Frames captured:   {total_frames}")
    print(f"Per-word target:   {target_per_word} frames (~{word_secs:.2f}s)")
    print(f"Zoom path:         {zoom_start:.2f} -> {zoom_end:.2f}")
    print(f"Output length:     ~{total_frames/args.fps:.2f}s at {args.fps} fps")
    print(f"Encode time:       {encode_secs:.2f}s")
    print(f"Output size:       {size_mb:.2f} MiB -> {args.out}")


if __name__ == "__main__":
    main()
