# !/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import time
import math
import csv
import random

import wikipedia
from playwright.sync_api import sync_playwright
from PIL import Image

from video_utils import build_video_from_frames
from zoom_utils import resolve_zoom_range

HIDE_CHROME_CSS = """
/* Hide some wiki chrome to keep things cleaner */
#siteNotice,
.mw-head,
#mw-head,
#mw-page-base,
#mw-head-base,
#mw-panel,
#vector-toc,
.vector-header-container,
.vector-sticky-header,
#footer {
    display: none !important;
}
body {
    margin: 0 !important;
}
"""

HIGHLIGHT_CSS = """
.wwtv-highlight {
    background: #ffeb3b;
    padding: 2px 3px;
}
"""

HIGHLIGHT_JS = """
(term) => {
    // Remove previous highlight if present
    document.querySelectorAll('.wwtv-highlight').forEach(span => {
        const parent = span.parentNode;
        while (span.firstChild) {
            parent.insertBefore(span.firstChild, span);
        }
        parent.removeChild(span);
    });

    const root =
        document.querySelector('#mw-content-text') ||
        document.body;

    const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_TEXT,
        null
    );

    const lowerTerm = term.toLowerCase();
    const matches = [];

    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.nodeValue) continue;
        const text = node.nodeValue;
        const lower = text.toLowerCase();
        let idx = lower.indexOf(lowerTerm);
        while (idx !== -1) {
            matches.push({ node, idx });
            idx = lower.indexOf(lowerTerm, idx + lowerTerm.length);
        }
    }

    if (matches.length === 0) {
        return 0;
    }

    // Pick a random occurrence for variety
    const choice = matches[Math.floor(Math.random() * matches.length)];
    const node = choice.node;
    const i = choice.idx;

    const text = node.nodeValue;
    const before = text.slice(0, i);
    const mid = text.slice(i, i + term.length);
    const after = text.slice(i + term.length);

    const parent = node.parentNode;

    if (before.length) parent.insertBefore(document.createTextNode(before), node);

    const span = document.createElement('span');
    span.className = 'wwtv-highlight';
    span.id = 'wwtv-focus';
    span.textContent = mid;
    parent.insertBefore(span, node);

    if (after.length) parent.insertBefore(document.createTextNode(after), node);

    parent.removeChild(node);

    span.scrollIntoView({ block: 'center', inline: 'center' });
    return matches.length;
}
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a newspaper-style video of a word/phrase using real Wikipedia pages."
    )
    parser.add_argument(
        "--term",
        required=True,
        help="Word or short phrase to highlight, e.g. 'hello' or 'hello world'.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Page zoom factor (1.0 = normal, >1 = closer, <1 = further).",
    )
    zoom_group = parser.add_mutually_exclusive_group()
    zoom_group.add_argument(
        "--zoom-in",
        type=float,
        help="Smoothly zoom IN over the final video by this multiplier (e.g. 1.5).",
    )
    zoom_group.add_argument(
        "--zoom-out",
        type=float,
        help="Smoothly zoom OUT over the final video by this multiplier (e.g. 1.5).",
    )
    zoom_group.add_argument(
        "--zoom-end",
        type=float,
        help="Final video zoom factor; equivalent to using --zoom plus a target.",
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
        default=50,
        help="Max Wikipedia pages to use (default: 50).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output frame width (default: 1024).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Output frame height (default: 576).",
    )
    parser.add_argument(
        "--crf",
        default="18",
        help="ffmpeg CRF for quality (default: 18; lower = better quality/larger file).",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="ffmpeg preset for encode speed/size (default: slow).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=(
            "Target video length in seconds. If we collect fewer frames than "
            "fps * seconds, existing frames will be re-used to hit that length."
        ),
    )
    parser.add_argument(
        "--center-debug",
        action="store_true",
        help="Collect centering stats and write centers.csv + centers.png.",
    )
    return parser.parse_args()

def crop_around_highlight(raw_path, bbox, out_path, out_w, out_h):
    """
    Take a full-viewport screenshot and crop a window centered on the highlight,
    then resize to (out_w, out_h). Return the normalized (fx, fy) position of
    the highlight center in the FINAL frame (0..1 in each dimension).
    """
    img = Image.open(raw_path)
    W, H = img.size  # viewport size in pixels

    # Base crop window: some fraction of viewport, but with same aspect as output
    base_fraction = 0.6  # 60% of viewport width by default
    base_w = int(W * base_fraction)
    target_ratio = out_w / out_h
    base_h = int(base_w / target_ratio)
    if base_h > H:
        base_h = int(H * base_fraction)
        base_w = int(base_h * target_ratio)

    # How big is the highlight relative to that base window?
    highlight_ratio = bbox["width"] / base_w if base_w > 0 else 1.0

    # If highlight would fill too much of the window, zoom OUT so it only
    # fills ~70% of the width at most.
    max_fill = 0.7
    zoom_factor = 1.0
    if highlight_ratio > max_fill:
        zoom_factor = min(
            (highlight_ratio / max_fill),
            W / base_w,
            H / base_h,
        )

    crop_w = min(W, int(base_w * zoom_factor))
    crop_h = min(H, int(base_h * zoom_factor))

    # Center crop on highlight center
    cx = bbox["x"] + bbox["width"] / 2.0
    cy = bbox["y"] + bbox["height"] / 2.0

    left = int(cx - crop_w / 2.0)
    top = int(cy - crop_h / 2.0)

    # Clamp to stay inside the image
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + crop_w > W:
        left = W - crop_w
    if top + crop_h > H:
        top = H - crop_h

    box = (left, top, left + crop_w, top + crop_h)
    cropped = img.crop(box)

    # Compute highlight center in the FINAL frame
    sx = out_w / crop_w
    sy = out_h / crop_h
    fx = (cx - left) * sx
    fy = (cy - top) * sy
    fx_norm = fx / out_w
    fy_norm = fy / out_h

    resized = cropped.resize((out_w, out_h), Image.LANCZOS)
    resized.save(out_path)
    img.close()

    # normalized 0..1 coordinates of highlight center in the output frame
    return fx_norm, fy_norm

def get_page_titles(term: str, max_pages: int):
    """Search Wikipedia for `term` and return up to `max_pages` page titles."""
    return wikipedia.search(term, results=max_pages)

def screenshot_page_with_term(
    page,
    title,
    term,
    frame_idx,
    frames_dir,
    out_w,
    out_h,
    zoom,
    centers=None,
):
    """
    Load one Wikipedia page, highlight a random occurrence of `term`,
    center it, screenshot the viewport, then crop around the highlight.
    Also optionally record the final-frame center for centering diagnostics.
    """
    url_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{url_title}"
    print(f"[{frame_idx:02d}] {title} -> {url}")

    # Navigate
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
    except Exception as e:
        print(f"  -> failed to load: {e}")
        return False

    # Apply zoom (how big the page appears)
    try:
        page.evaluate("(z) => { document.body.style.zoom = z; }", zoom)
    except Exception as e:
        print(f"  -> failed to set zoom, continuing with zoom=1.0: {e}")

    # Inject CSS (hide chrome + highlight style)
    page.add_style_tag(content=HIDE_CHROME_CSS + HIGHLIGHT_CSS)

    # Highlight a random match on this page
    matches_count = page.evaluate(HIGHLIGHT_JS, term)
    if not matches_count:
        print("  -> term not found on page; skipping")
        return False

    # Get bounding box of the span
    focus = page.locator("#wwtv-focus")
    try:
        box = focus.bounding_box()
    except Exception as e:
        print(f"  -> could not get bounding box: {e}")
        return False
    if box is None:
        print("  -> highlight not visible; skipping")
        return False

    # Take full-viewport screenshot
    raw_path = os.path.join(frames_dir, f"raw_{frame_idx:05d}.png")
    final_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
    try:
        page.screenshot(path=raw_path, full_page=False)
    except Exception as e:
        print(f"  -> screenshot failed: {e}")
        return False

    # Crop around highlight and record center in final frame
    try:
        fx, fy = crop_around_highlight(raw_path, box, final_path, out_w, out_h)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    if centers is not None:
        centers.append(
            {
                "frame_idx": frame_idx,
                "page_title": title,
                "x": fx,
                "y": fy,
            }
        )

    return True

def save_center_debug(
    centers,
    frame_w: int,
    frame_h: int,
    csv_path: str = "centers.csv",
    png_path: str = "centers.png",
):
    """
    centers: list of dicts with keys:
      - frame_idx
      - page_title
      - x, y  (highlight center in FINAL frame coordinates, pixels)
    Writes:
      - centers.csv
      - centers.png (2D heatmap of normalized centers)
    """
    if not centers:
        return

    max_r = math.hypot(frame_w / 2.0, frame_h / 2.0)

    # --- CSV ---
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "page_title", "x", "y", "nx", "ny", "err_pct"])
        for rec in centers:
            x = rec["x"]
            y = rec["y"]
            nx = x / frame_w
            ny = y / frame_h
            dx = x - frame_w / 2.0
            dy = y - frame_h / 2.0
            err_pct = math.hypot(dx, dy) / max_r * 100.0
            w.writerow(
                [
                    rec["frame_idx"],
                    rec["page_title"],
                    f"{x:.2f}",
                    f"{y:.2f}",
                    f"{nx:.4f}",
                    f"{ny:.4f}",
                    f"{err_pct:.2f}",
                ]
            )

    # --- Heatmap ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping centers.png heat map.")
        return

    xs = [rec["x"] / frame_w for rec in centers]
    ys = [rec["y"] / frame_h for rec in centers]

    plt.figure(figsize=(6, 3.5))
    plt.hist2d(xs, ys, bins=50, range=[[0.0, 1.0], [0.0, 1.0]])
    plt.scatter([0.5], [0.5], marker="+", s=80, linewidths=2, label="Perfect center")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized X (0 = left, 1 = right)")
    plt.ylabel("Normalized Y (0 = top, 1 = bottom)")
    plt.title("Highlight center distribution")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # --- Console summary ---
    err_vals = []
    for rec in centers:
        x = rec["x"]
        y = rec["y"]
        dx = x - frame_w / 2.0
        dy = y - frame_h / 2.0
        err_vals.append(math.hypot(dx, dy) / max_r * 100.0)

    if err_vals:
        print(
            f"Centering stats: {len(err_vals)} frames, "
            f"mean error ~{sum(err_vals)/len(err_vals):.2f}%, "
            f"max error ~{max(err_vals):.2f}% "
            f"(see {png_path})"
        )
def main():
    args = parse_args()
    term = args.term.strip()
    zoom_start, zoom_end = resolve_zoom_range(
        args.zoom, args.zoom_end, args.zoom_in, args.zoom_out
    )

    frames_dir = "frames"
    centers = [] if args.center_debug else None
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    titles = get_page_titles(term, args.max_pages)
    if not titles:
        print("No Wikipedia pages found for that term.")
        return

    print(f"Found {len(titles)} candidate pages for '{term}'")

    used = 0
    with sync_playwright() as p:
        # Render at 2× resolution for sharper downscaled frames
        viewport_w = args.width * 2
        viewport_h = args.height * 2

        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": viewport_w, "height": viewport_h}
        )

        for title in titles:
            ok = screenshot_page_with_term(
                page, title, term, used, frames_dir, args.width, args.height, args.zoom, centers=centers,
            )
            if ok:
                used += 1

        browser.close()

        total_frames = used

        if args.seconds is not None and args.seconds > 0:
            target_frames = int(round(args.seconds * args.fps))
            print(
                f"\nTarget length: ~{args.seconds:.2f}s -> "
                f"{target_frames} frames at {args.fps} fps"
            )

            if target_frames > total_frames:
                print(
                    f"Only {total_frames} unique frames collected; "
                    f"duplicating frames to reach {target_frames}."
                )
                for i in range(total_frames, target_frames):
                    src_idx = random.randint(0, total_frames - 1)
                    src_name = os.path.join(frames_dir, f"frame_{src_idx:05d}.png")
                    dst_name = os.path.join(frames_dir, f"frame_{i:05d}.png")
                    shutil.copy2(src_name, dst_name)
                used = target_frames
            else:
                # We have at least as many frames as we need; just keep them all.
                actual = total_frames / float(args.fps)
                print(
                    f"Collected {total_frames} frames (>= {target_frames}); "
                    f"final length will be ~{actual:.2f}s."
                )
                used = total_frames
    
    if used == 0:
        print("No pages produced frames; nothing to build.")
        return

    if args.center_debug and centers:
        save_center_debug(centers, args.width, args.height)

    encode_secs, size_bytes = build_video_from_frames(
        frames_dir,
        args.fps,
        args.out,
        frame_size=(args.width, args.height),
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        crf=args.crf,
        preset=args.preset,
    )
    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0

    print("\n=== Summary ===")
    print(f"Frames written: {used}")
    if args.seconds is not None and args.seconds > 0:
        print(f"Requested length: ~{args.seconds:.2f}s at {args.fps} fps")
    print(f"Actual length:   ~{used/args.fps:.2f}s at {args.fps} fps")
    print(f"Encode time:     {encode_secs:.2f}s")
    print(f"Output size:     {size_mb:.2f} MiB -> {args.out}")


if __name__ == "__main__":
    main()
