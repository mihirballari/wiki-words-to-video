
#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import subprocess

import wikipedia
from playwright.sync_api import sync_playwright


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
        "--seconds",
        type=float,
        default=None,
        help="Target video length in seconds. If set, the script will "
             "try to generate about fps * seconds frames.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Max Wikipedia pages to use in one pass (default: 30).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="OUTPUT frame width in pixels (default: 1024).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="OUTPUT frame height in pixels (default: 576).",
    )
    parser.add_argument(
        "--viewport-scale",
        type=float,
        default=1.6,
        help="How much larger the browser viewport is than the frame "
             "(default: 1.6). Larger = more room to center/crop.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Page zoom factor. 1.0 = normal, >1 = closer (word bigger), "
             "<1 = further (more page). Default: 1.0",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated frame images after building the video.",
    )
    return parser.parse_args()


def get_page_titles(term: str, max_pages: int):
    """Search Wikipedia for `term` and return up to `max_pages` page titles."""
    return wikipedia.search(term, results=max_pages)


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
        while (span.firstChild) parent.insertBefore(span.firstChild, span);
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
        return 0; // no match on this page
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

    return matches.length;
}
"""

CENTER_JS = """
() => {
    const span = document.getElementById('wwtv-focus');
    if (!span) return null;

    const rect = span.getBoundingClientRect();

    const targetX =
        window.scrollX + rect.left + rect.width / 2 - window.innerWidth / 2;
    const targetY =
        window.scrollY + rect.top + rect.height / 2 - window.innerHeight / 2;

    window.scrollTo(targetX, targetY);

    const r2 = span.getBoundingClientRect();
    return {
        left: r2.left,
        top: r2.top,
        width: r2.width,
        height: r2.height,
        vw: window.innerWidth,
        vh: window.innerHeight
    };
}
"""


def screenshot_page_with_term(
    page,
    title,
    term,
    frame_idx,
    frames_dir,
    frame_w,
    frame_h,
    viewport_w,
    viewport_h,
    zoom,
):
    """
    Load one Wikipedia page, highlight a random occurrence of `term`,
    zoom, center it, and screenshot a FIXED-SIZE clip around the highlight.
    """
    url_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{url_title}"
    print(f"[{frame_idx:03d}] {title} -> {url}")

    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
    except Exception as e:
        print(f"  -> failed to load: {e}")
        return False

    # Apply zoom (how big the text appears on screen)
    try:
        page.evaluate(
            "(z) => { document.body.style.zoom = z; }",
            zoom,
        )
    except Exception as e:
        print(f"  -> failed to set zoom, continuing with zoom=1.0: {e}")

    # Inject CSS (every page, in case wiki layout changes)
    page.add_style_tag(content=HIDE_CHROME_CSS + HIGHLIGHT_CSS)

    # Highlight a random match on this page
    matches_count = page.evaluate(HIGHLIGHT_JS, term)
    if not matches_count:
        print("  -> term not found on page; skipping")
        return False

    # Center the highlight in the viewport
    center_info = page.evaluate(CENTER_JS)
    if center_info is None:
        print("  -> could not center highlight; skipping")
        return False

    # Get bounding box of span relative to viewport
    focus = page.locator("#wwtv-focus")
    try:
        box = focus.bounding_box()
    except Exception as e:
        print(f"  -> could not get bounding box: {e}")
        return False
    if box is None:
        print("  -> highlight not visible; skipping")
        return False

    # Compute clip rectangle so that the highlight is centered in the frame
    cx = box["x"] + box["width"] / 2.0
    cy = box["y"] + box["height"] / 2.0

    clip_x = int(cx - frame_w / 2.0)
    clip_y = int(cy - frame_h / 2.0)

    # Clamp clip rectangle to viewport so we never go "outside" the page
    if clip_x < 0:
        clip_x = 0
    if clip_y < 0:
        clip_y = 0
    if clip_x + frame_w > viewport_w:
        clip_x = max(0, viewport_w - frame_w)
    if clip_y + frame_h > viewport_h:
        clip_y = max(0, viewport_h - frame_h)

    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
    try:
        page.screenshot(
            path=frame_path,
            full_page=False,
            clip={
                "x": clip_x,
                "y": clip_y,
                "width": frame_w,
                "height": frame_h,
            },
        )
    except Exception as e:
        print(f"  -> screenshot failed: {e}")
        return False

    return True


def open_video(out_path: str):
    """Try to open the video file with the default player."""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.Popen(["open", out_path])
        elif system == "Windows":
            os.startfile(out_path)  # type: ignore[attr-defined]
        else:  # Linux / other
            subprocess.Popen(["xdg-open", out_path])
    except Exception as e:
        print(f"Could not auto-open video: {e}")


def build_video_from_frames(frames_dir, fps, out_path):
    """Call ffmpeg to build an MP4 from frames in `frames_dir`."""
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
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Auto-open the video
    open_video(out_path)


def main():
    args = parse_args()
    term = args.term.strip()

    frame_w = args.width
    frame_h = args.height
    viewport_w = int(frame_w * args.viewport_scale)
    viewport_h = int(frame_h * args.viewport_scale)

    frames_dir = "frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    titles = get_page_titles(term, args.max_pages)
    if not titles:
        print("No Wikipedia pages found for that term.")
        return

    print(f"Found {len(titles)} candidate pages for '{term}'")
    print(f"Frame size: {frame_w}x{frame_h}, viewport: {viewport_w}x{viewport_h}")
    print(f"Zoom factor: {args.zoom}")

    if args.seconds is not None and args.seconds > 0:
        target_frames = int(max(1, round(args.fps * args.seconds)))
        print(f"Target length: ~{args.seconds:.2f}s = {target_frames} frames at {args.fps} fps")
    else:
        target_frames = None

    used = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": viewport_w, "height": viewport_h}
        )

        if target_frames is None:
            # Single pass over pages
            for title in titles:
                ok = screenshot_page_with_term(
                    page,
                    title,
                    term,
                    used,
                    frames_dir,
                    frame_w,
                    frame_h,
                    viewport_w,
                    viewport_h,
                    args.zoom,
                )
                if ok:
                    used += 1
        else:
            # Multiple passes until we hit target_frames or can't make more
            passes = 0
            MAX_PASSES = 20  # safety cap
            while used < target_frames and passes < MAX_PASSES:
                passes += 1
                made_before = used
                print(f"\n=== Pass {passes} over titles (current frames: {used}) ===")
                for title in titles:
                    if used >= target_frames:
                        break
                    ok = screenshot_page_with_term(
                        page,
                        title,
                        term,
                        used,
                        frames_dir,
                        frame_w,
                        frame_h,
                        viewport_w,
                        viewport_h,
                        args.zoom,
                    )
                    if ok:
                        used += 1

                if used == made_before:
                    print("No new frames could be generated on this pass; stopping early.")
                    break

            if used < target_frames:
                actual_len = used / float(args.fps)
                print(
                    f"Warning: only {used} frames generated (~{actual_len:.2f}s at "
                    f"{args.fps} fps), fewer than requested {target_frames} frames."
                )

        browser.close()

    if used == 0:
        print("No pages produced frames; nothing to build.")
        return

    build_video_from_frames(frames_dir, args.fps, args.out)
    print(f"Done. Wrote {used} frames to {args.out}")

    if args.clean:
        try:
            shutil.rmtree(frames_dir)
            print("Cleaned up frames/ directory.")
        except Exception as e:
            print(f"Could not remove frames directory: {e}")


if __name__ == "__main__":
    main()

