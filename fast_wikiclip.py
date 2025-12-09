# !/usr/bin/env python3
"""
Drop-in faster variant of wikiclip.py that keeps the same CLI flags.
Speed gains come from:
  - API-side filtering (insource search + extract check) to skip pages without the term
  - Using lean printable desktop pages instead of the full chrome
  - Blocking fonts/analytics during navigation (keep CSS for layout + variety)
  - Running multiple pages in parallel
"""
import argparse
import asyncio
import csv
import math
import os
import random
import shutil
import time
import urllib.parse
import urllib.request
from io import BytesIO

try:
    from playwright.async_api import async_playwright
except ImportError as exc:
    async_playwright = None  # Defer error until runtime for clearer messaging.
from PIL import Image

from video_utils import build_video_from_frames


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

USER_AGENT = "wiki-words-to-video/fast (https://en.wikipedia.org)"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DESKTOP_PAGE = "https://en.wikipedia.org/wiki/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast newspaper-style video of a word/phrase using real Wikipedia pages."
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
        help="Max Wikipedia pages to use (default: 30).",
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


def http_get_json(url, params):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read().decode("utf-8")


def search_titles(term, max_pages):
    """Use insource search to bias toward pages that actually contain the term."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'insource:"{term}"',
        "srlimit": max(max_pages * 2, max_pages),
        "format": "json",
        "formatversion": "2",
        "utf8": "1",
    }
    try:
        raw = http_get_json(WIKI_API_URL, params)
        import json

        data = json.loads(raw)
        hits = data.get("query", {}).get("search", [])
        titles = [hit.get("title") for hit in hits if hit.get("title")]
        return titles[: max_pages * 2] if titles else []
    except Exception as exc:
        print(f"API search failed, continuing without fast filter: {exc}")
        return []


def filter_titles_by_extract(term, titles, max_keep):
    """Pull short extracts and keep only titles whose extract contains the term."""
    if not titles:
        return []

    term_lower = term.lower()
    filtered = []
    chunk_size = 20  # MediaWiki recommends <=50; keep smaller to stay light.
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i : i + chunk_size]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exintro": "1",
            "format": "json",
            "formatversion": "2",
            "titles": "|".join(chunk),
            "utf8": "1",
        }
        try:
            raw = http_get_json(WIKI_API_URL, params)
            import json

            data = json.loads(raw)
            pages = data.get("query", {}).get("pages", [])
            for p in pages:
                extract = p.get("extract") or ""
                if term_lower in extract.lower():
                    filtered.append(p.get("title"))
                    if len(filtered) >= max_keep:
                        return filtered
        except Exception as exc:
            print(f"Extract filter chunk failed: {exc}")
    return filtered


def crop_around_highlight_from_bytes(img_bytes, bbox, out_path, out_w, out_h):
    """
    Crop around the highlight using an in-memory screenshot.
    Returns normalized (fx, fy) center position in the final frame.
    """
    img = Image.open(BytesIO(img_bytes))
    W, H = img.size

    base_fraction = 0.6
    base_w = int(W * base_fraction)
    target_ratio = out_w / out_h
    base_h = int(base_w / target_ratio)
    if base_h > H:
        base_h = int(H * base_fraction)
        base_w = int(base_h * target_ratio)

    highlight_ratio = bbox["width"] / base_w if base_w > 0 else 1.0
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

    cx = bbox["x"] + bbox["width"] / 2.0
    cy = bbox["y"] + bbox["height"] / 2.0

    left = int(cx - crop_w / 2.0)
    top = int(cy - crop_h / 2.0)

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

    sx = out_w / crop_w
    sy = out_h / crop_h
    fx = (cx - left) * sx
    fy = (cy - top) * sy
    fx_norm = fx / out_w
    fy_norm = fy / out_h

    resized = cropped.resize((out_w, out_h), Image.LANCZOS)
    resized.save(out_path)
    img.close()

    return fx_norm, fy_norm


def save_center_debug(centers, frame_w, frame_h, csv_path="centers.csv"):
    """CSV-only center debug (no matplotlib dependency)."""
    if not centers:
        return

    max_r = math.hypot(frame_w / 2.0, frame_h / 2.0)
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
    print(f"Wrote center debug CSV -> {csv_path}")


async def process_page(
    browser_context,
    title,
    term,
    zoom,
    out_w,
    out_h,
    frames_dir,
    centers,
    counter_lock,
    counter,
):
    url_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"{DESKTOP_PAGE}{url_title}?printable=yes"
    page = await browser_context.new_page()
    try:
        await page.route(
            "**/*",
            lambda route, request: (
                route.abort()
                if request.resource_type in {"font"}  # type: ignore[attr-defined]
                else route.continue_()
            ),
        )
        await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        await page.add_style_tag(content=HIDE_CHROME_CSS + HIGHLIGHT_CSS)
        try:
            await page.evaluate("(z) => { document.body.style.zoom = z; }", zoom)
        except Exception:
            pass

        matches_count = await page.evaluate(HIGHLIGHT_JS, term)
        if not matches_count:
            print(f"  -> term not found on {title}; skipping")
            return False

        focus = page.locator("#wwtv-focus")
        box = await focus.bounding_box()
        if not box:
            print(f"  -> highlight not visible on {title}; skipping")
            return False

        img_bytes = await page.screenshot(full_page=False)

        async with counter_lock:
            idx = counter["value"]
            counter["value"] += 1

        final_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        fx_norm, fy_norm = crop_around_highlight_from_bytes(
            img_bytes, box, final_path, out_w, out_h
        )

        if centers is not None:
            centers.append(
                {
                    "frame_idx": idx,
                    "page_title": title,
                    "x": fx_norm * out_w,
                    "y": fy_norm * out_h,
                }
            )
        print(f"[{idx:02d}] {title} -> ok")
        return True
    except Exception as exc:
        print(f"  -> failed {title}: {exc}")
        return False
    finally:
        await page.close()


async def run(args):
    if async_playwright is None:
        raise SystemExit(
            "playwright is not installed. Install with `pip install playwright` "
            "and run `playwright install chromium` once."
        )
    term = args.term.strip()

    frames_dir = "frames"
    centers = [] if args.center_debug else None
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Searching for pages containing '{term}'...")
    api_titles = search_titles(term, args.max_pages)
    candidate_titles = api_titles or []
    if not candidate_titles:
        print("No titles from insource search; stopping.")
        return

    filtered_titles = filter_titles_by_extract(term, candidate_titles, args.max_pages)
    titles = filtered_titles or candidate_titles[: args.max_pages]
    print(f"Using {len(titles)} titles after filtering.")
    random.shuffle(titles)

    used = 0
    viewport_w = args.width * 2
    viewport_h = args.height * 2
    counter = {"value": 0}
    counter_lock = asyncio.Lock()

    concurrency = max(1, min(8, len(titles), (os.cpu_count() or 4)))

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": viewport_w, "height": viewport_h},
            user_agent=USER_AGENT,
        )

        sem = asyncio.Semaphore(concurrency)

        async def worker(title):
            async with sem:
                return await process_page(
                    context,
                    title,
                    term,
                    args.zoom,
                    args.width,
                    args.height,
                    frames_dir,
                    centers,
                    counter_lock,
                    counter,
                )

        results = await asyncio.gather(*(worker(t) for t in titles))
        used = sum(1 for r in results if r)
        await context.close()
        await browser.close()

    total_frames = used

    if args.seconds is not None and args.seconds > 0:
        target_frames = int(round(args.seconds * args.fps))
        print(
            f"\nTarget length: ~{args.seconds:.2f}s -> "
            f"{target_frames} frames at {args.fps} fps"
        )

        if target_frames > total_frames and total_frames > 0:
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
        elif target_frames <= total_frames:
            actual = total_frames / float(args.fps)
            print(
                f"Collected {total_frames} frames (>= {target_frames}); "
                f"final length will be ~{actual:.2f}s."
            )
            used = total_frames

    if args.center_debug and centers:
        save_center_debug(centers, args.width, args.height)

    encode_secs, size_bytes = build_video_from_frames(frames_dir, args.fps, args.out)
    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0

    print("\n=== Summary ===")
    print(f"Frames written: {used}")
    if args.seconds is not None and args.seconds > 0:
        print(f"Requested length: ~{args.seconds:.2f}s at {args.fps} fps")
    print(f"Actual length:   ~{used/args.fps:.2f}s at {args.fps} fps")
    print(f"Encode time:     {encode_secs:.2f}s")
    print(f"Output size:     {size_mb:.2f} MiB -> {args.out}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
