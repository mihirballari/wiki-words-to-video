
import os

from PIL import Image
from playwright.sync_api import Page

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


def crop_around_highlight(raw_path, bbox, out_path, out_w, out_h):
    img = Image.open(raw_path)
    W, H = img.size  # viewport size in pixels

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
    resized = cropped.resize((out_w, out_h), Image.LANCZOS)
    resized.save(out_path)
    img.close()

    # Highlight center position in the final frame (after crop + resize)
    fx = (cx - left) / crop_w * out_w
    fy = (cy - top) / crop_h * out_h
    return fx, fy

def save_center_debug(
    centers,
    frame_w,
    frame_h,
    csv_path="centers.csv",
    png_path="centers.png",
):
    """
    centers: list of dicts with keys frame_idx, page_title, x, y (final-frame coords).
    """
    if not centers:
        return

    max_r = math.hypot(frame_w / 2.0, frame_h / 2.0)

    # Write CSV
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

    # Heat map
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping centers.png heat map.")
        return

    xs = [rec["x"] / frame_w for rec in centers]
    ys = [rec["y"] / frame_h for rec in centers]

    plt.figure(figsize=(6, 3.5))
    plt.hist2d(xs, ys, bins=50, range=[[0.0, 1.0], [0.0, 1.0]])
    plt.scatter([0.5], [0.5], marker="+", s=80, linewidths=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized X (0 = left, 1 = right)")
    plt.ylabel("Normalized Y (0 = top, 1 = bottom)")
    plt.title("Highlight center distribution")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Console stats
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

def screenshot_page_with_term(page, title, term, frame_idx, frames_dir, out_w, out_h, zoom): 
    url_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{url_title}"
    print(f"[{frame_idx:02d}] {title} -> {url}")

    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
    except Exception as e:
        print(f"  -> failed to load: {e}")
        return False

    # Page zoom
    try:
        page.evaluate("(z) => { document.body.style.zoom = z; }", zoom)
    except Exception as e:
        print(f"  -> failed to set zoom, continuing with zoom=1.0: {e}")

    page.add_style_tag(content=HIDE_CHROME_CSS + HIGHLIGHT_CSS)

    matches_count = page.evaluate(HIGHLIGHT_JS, term)
    if not matches_count:
        print("  -> term not found on page; skipping")
        return False

    focus = page.locator("#wwtv-focus")
    try:
        box = focus.bounding_box()
    except Exception as e:
        print(f"  -> could not get bounding box: {e}")
        return False
    if box is None:
        print("  -> highlight not visible; skipping")
        return False

    raw_path = os.path.join(frames_dir, f"raw_{frame_idx:05d}.png")
    final_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")

    try:
        page.screenshot(path=raw_path, full_page=False)
    except Exception as e:
        print(f"  -> screenshot failed: {e}")
        return False

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
