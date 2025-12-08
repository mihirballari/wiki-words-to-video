#!/usr/bin/env python3
import argparse
import os
import shutil
import textwrap
import subprocess

import wikipedia
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a newspaper-style video of a word/phrase using Wikipedia pages."
    )
    parser.add_argument(
        "--term",
        required=True,
        help="Word or short phrase to search + highlight, e.g. 'testing'.",
    )
    parser.add_argument(
        "--out",
        default="out.mp4",
        help="Output video filename (default: out.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the video (default: 10).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=15,
        help="Max Wikipedia pages to search (default: 15).",
    )
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=60,
        help="Max total snippets/frames (default: 60).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height (default: 720).",
    )
    return parser.parse_args()


def get_snippets(term, n_pages=15, max_snippets=60, context_chars=100):
    """
    Search Wikipedia for `term` and return a list of (title, snippet).
    `term` can be a word or a short phrase. We search with it as-is,
    but highlight individual words from the phrase for now.
    """
    term_lower = term.lower()
    results = wikipedia.search(term, results=n_pages)

    snippets = []
    for title in results:
        if len(snippets) >= max_snippets:
            break

        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content
        except Exception:
            continue

        lower = text.lower()
        start = 0
        while len(snippets) < max_snippets:
            idx = lower.find(term_lower, start)
            if idx == -1:
                break
            left = max(0, idx - context_chars)
            right = min(len(text), idx + len(term_lower) + context_chars)
            snippet = text[left:right].replace("\n", " ")
            snippets.append((title, snippet.strip()))
            start = idx + len(term_lower)

    return snippets


def load_fonts():
    """
    Try to load a nicer TTF font; fall back to default if not available.
    """
    try:
        # This path may or may not exist; adjust later if you want.
        return (
            ImageFont.truetype("Arial.ttf", 52),
            ImageFont.truetype("Arial.ttf", 32),
        )
    except Exception:
        # Fallback: boring but works everywhere.
        return ImageFont.load_default(), ImageFont.load_default()


def render_snippet_frame(
    idx,
    term,
    title,
    snippet,
    out_dir,
    size=(1280, 720),
    bg=(245, 245, 245),
):
    """
    Render a single frame PNG with the snippet text and highlighted term.
    Returns the path to the written file.
    """
    W, H = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    title_font, body_font = load_fonts()

    margin_x = 80
    y = 60

    # Draw the big term in the center-ish top
    big_term_font = title_font
    term_text = term
    tw, th = draw.textbbox((0, 0), term_text, font=big_term_font)[2:]
    draw.text(
        ((W - tw) / 2, y),
        term_text,
        font=big_term_font,
        fill=(30, 30, 30),
    )
    y += th + 40

    # Draw the page title smaller above the snippet
    title_text = title[:80]
    draw.text(
        (margin_x, y),
        f"From: {title_text}",
        font=body_font,
        fill=(80, 80, 80),
    )
    y += 40

    # Snippet text area
    text_top = y
    margin_y = 40
    max_width = W - 2 * margin_x

    # We highlight any word that matches one of the words in the term.
    term_words = [w.lower() for w in term.split()]
    words = snippet.split()

    # Basic metrics
    _, _, _, line_h = draw.textbbox((0, 0), "Ag", font=body_font)
    line_h = line_h + 8  # add a bit of spacing
    space_w = draw.textlength(" ", font=body_font)

    x = margin_x
    y = text_top

    highlight_color = (255, 235, 59)  # yellow

    for raw_word in words:
        # Strip punctuation just for matching.
        cleaned = raw_word.strip(".,!?;:\"'()[]{}").lower()
        w_width = draw.textlength(raw_word, font=body_font)

        if x + w_width > margin_x + max_width:
            # New line
            x = margin_x
            y += line_h

            if y + line_h > H - margin_y:
                break  # don't overflow the bottom

        # Highlight if matches any term word
        if cleaned in term_words and cleaned != "":
            # rectangle a bit larger than text
            rect_x1 = x - 3
            rect_y1 = y - 3
            rect_x2 = x + w_width + 3
            rect_y2 = y + line_h - 6
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=highlight_color)

        draw.text((x, y), raw_word, font=body_font, fill=(0, 0, 0))
        x += w_width + space_w

    os.makedirs(out_dir, exist_ok=True)
    frame_path = os.path.join(out_dir, f"frame_{idx:05d}.png")
    img.save(frame_path)
    return frame_path


def build_video_from_frames(frames_dir, fps, out_path):
    """
    Call ffmpeg to build an MP4 from frames in `frames_dir`.
    """
    # Clean any existing output
    if os.path.exists(out_path):
        os.remove(out_path)

    # ffmpeg command: frames/frame_00000.png etc.
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(frames_dir, "frame_*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    term = args.term.strip()
    out_path = args.out

    frames_dir = "frames"

    # Clear old frames
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Searching Wikipedia for '{term}'...")
    snippets = get_snippets(
        term, n_pages=args.max_pages, max_snippets=args.max_snippets
    )
    if not snippets:
        print("No snippets found for that term. Try a different word/phrase.")
        return

    print(f"Rendering {len(snippets)} frames...")
    for i, (title, snippet) in enumerate(snippets):
        render_snippet_frame(
            idx=i,
            term=term,
            title=title,
            snippet=snippet,
            out_dir=frames_dir,
            size=(args.width, args.height),
        )

    print("Building video with ffmpeg...")
    build_video_from_frames(frames_dir, args.fps, out_path)
    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
