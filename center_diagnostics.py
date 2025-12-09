"""
Diagnostics for center accuracy, highlight bounding boxes, and zoom smoothness.

Consumes:
  - centers.csv (from --center-debug)
  - frames/frame_XXXXX.png (final rendered frames)

Outputs to diagnostics/:
  - center_heatmap.png
  - center_errors.png
  - area_trend.png
  - anomalies.txt
  - annotated/frame_XXXXX.png (per-frame overlay of center + bbox)
"""

import csv
import glob
import hashlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dep
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dep
    np = None

HIGHLIGHT_COLOR = (255, 235, 59)  # CSS #ffeb3b
HIGHLIGHT_TOLERANCE = 32
IDEAL_CENTER = (0.5, 0.5)


@dataclass
class CenterRecord:
    frame_idx: int
    x: float  # normalized [0,1]
    y: float  # normalized [0,1]
    page_title: str


@dataclass
class FrameAnalysis:
    frame_idx: int
    file_path: str
    center_x: Optional[float]
    center_y: Optional[float]
    center_error: Optional[float]
    bbox: Optional[Tuple[int, int, int, int]]  # (left, top, right, bottom)
    bbox_center: Optional[Tuple[float, float]]
    bbox_area: Optional[int]
    bbox_area_norm: Optional[float]
    bbox_center_error: Optional[float]


def load_centers(csv_path: str = "centers.csv") -> List[CenterRecord]:
    centers: List[CenterRecord] = []
    if not os.path.exists(csv_path):
        print(f"centers file not found: {csv_path}")
        return centers
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                centers.append(
                    CenterRecord(
                        frame_idx=int(row["frame_idx"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        page_title=row.get("page_title", ""),
                    )
                )
            except (KeyError, ValueError):
                continue
    return centers


def compute_center_errors(centers: List[CenterRecord]) -> List[Dict]:
    records = []
    max_r = math.hypot(0.5, 0.5)
    for rec in centers:
        dx = rec.x - IDEAL_CENTER[0]
        dy = rec.y - IDEAL_CENTER[1]
        dist = math.hypot(dx, dy)
        err_pct = (dist / max_r) * 100.0
        records.append(
            {
                "frame_idx": rec.frame_idx,
                "dx": dx,
                "dy": dy,
                "error": dist,
                "error_pct": err_pct,
            }
        )
    return records


def build_heatmap(centers: List[CenterRecord], out_path: str) -> None:
    if not plt:
        print("matplotlib not available; skipping heatmap.")
        return
    if not centers:
        return

    xs = [rec.x for rec in centers]
    ys = [rec.y for rec in centers]

    plt.figure(figsize=(6, 3.5))
    plt.hist2d(xs, ys, bins=50, range=[[0.0, 1.0], [0.0, 1.0]])
    plt.scatter([0.5], [0.5], marker="+", s=80, linewidths=2, label="Frame center")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized X (0 = left, 1 = right)")
    plt.ylabel("Normalized Y (0 = top, 1 = bottom)")
    plt.title("Highlight center distribution")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _mask_from_color(img: Image.Image) -> Optional["np.ndarray"]:
    if np is None:
        return None
    arr = np.asarray(img.convert("RGB"))
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (
        (np.abs(r - HIGHLIGHT_COLOR[0]) <= HIGHLIGHT_TOLERANCE)
        & (np.abs(g - HIGHLIGHT_COLOR[1]) <= HIGHLIGHT_TOLERANCE)
        & (np.abs(b - HIGHLIGHT_COLOR[2]) <= HIGHLIGHT_TOLERANCE)
    )
    return mask


def detect_highlight_bbox(frame_path: str) -> Optional[Tuple[int, int, int, int]]:
    with Image.open(frame_path) as img:
        mask = _mask_from_color(img)
        if mask is not None:
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None
            ys = coords[:, 0]
            xs = coords[:, 1]
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        # Fallback: slow pixel scan (no numpy)
        pix = img.convert("RGB").load()
        w, h = img.size
        left, top = w, h
        right, bottom = 0, 0
        found = False
        for y in range(h):
            for x in range(w):
                r, g, b = pix[x, y]
                if (
                    abs(r - HIGHLIGHT_COLOR[0]) <= HIGHLIGHT_TOLERANCE
                    and abs(g - HIGHLIGHT_COLOR[1]) <= HIGHLIGHT_TOLERANCE
                    and abs(b - HIGHLIGHT_COLOR[2]) <= HIGHLIGHT_TOLERANCE
                ):
                    found = True
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)
        if not found:
            return None
        return left, top, right, bottom


def annotate_frame(
    frame_path: str,
    out_path: str,
    bbox: Optional[Tuple[int, int, int, int]],
    center: Optional[Tuple[float, float]],
) -> None:
    with Image.open(frame_path) as img:
        draw = ImageDraw.Draw(img)
        w, h = img.size

        if bbox:
            l, t, r, b = bbox
            draw.rectangle([l, t, r, b], outline="red", width=3)
            draw.ellipse([(l + r) / 2 - 4, (t + b) / 2 - 4, (l + r) / 2 + 4, (t + b) / 2 + 4], fill="red")

        if center:
            cx, cy = center
            draw.line([(cx * w - 6, cy * h), (cx * w + 6, cy * h)], fill="blue", width=2)
            draw.line([(cx * w, cy * h - 6), (cx * w, cy * h + 6)], fill="blue", width=2)

        img.save(out_path)


def analyze_frames(
    frames_dir: str,
    centers: List[CenterRecord],
    out_dir: str,
    skip_duplicates: bool = True,
) -> List[FrameAnalysis]:
    os.makedirs(out_dir, exist_ok=True)
    annotated_dir = os.path.join(out_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    center_map = {c.frame_idx: c for c in centers}
    results: List[FrameAnalysis] = []
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    seen_hashes = set()

    frame_area = None
    if frame_paths:
        with Image.open(frame_paths[0]) as img:
            frame_area = img.size[0] * img.size[1]

    for frame_path in frame_paths:
        name = os.path.basename(frame_path)
        try:
            idx = int(name.split("_")[1].split(".")[0])
        except ValueError:
            continue

        with open(frame_path, "rb") as fh:
            digest = hashlib.sha1(fh.read()).hexdigest()
        if skip_duplicates and digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        bbox = detect_highlight_bbox(frame_path)
        bbox_center = None
        bbox_area = None
        bbox_area_norm = None
        if bbox:
            l, t, r, b = bbox
            bbox_center = ((l + r) / 2.0, (t + b) / 2.0)
            bbox_area = (r - l + 1) * (b - t + 1)
            if frame_area:
                bbox_area_norm = bbox_area / frame_area

        center_rec = center_map.get(idx)
        center_tuple = (center_rec.x, center_rec.y) if center_rec else None

        center_error = None
        bbox_center_error = None
        if center_rec:
            dx = center_rec.x - IDEAL_CENTER[0]
            dy = center_rec.y - IDEAL_CENTER[1]
            center_error = math.hypot(dx, dy)

        if bbox_center:
            # Normalize bbox center to [0,1]
            with Image.open(frame_path) as img:
                fw, fh = img.size
                norm_center = (bbox_center[0] / fw, bbox_center[1] / fh)
                dx = norm_center[0] - IDEAL_CENTER[0]
                dy = norm_center[1] - IDEAL_CENTER[1]
                bbox_center_error = math.hypot(dx, dy)

        annotated_path = os.path.join(annotated_dir, name)
        annotate_frame(frame_path, annotated_path, bbox, center_tuple)

        results.append(
            FrameAnalysis(
                frame_idx=idx,
                file_path=frame_path,
                center_x=center_rec.x if center_rec else None,
                center_y=center_rec.y if center_rec else None,
                center_error=center_error,
                bbox=bbox,
                bbox_center=bbox_center,
                bbox_area=bbox_area,
                bbox_area_norm=bbox_area_norm,
                bbox_center_error=bbox_center_error,
            )
        )

    return results


def detect_anomalies(errors: List[Dict], min_threshold: float = 0.03) -> Tuple[float, List[int]]:
    if not errors:
        return 0.0, []
    vals = [e["error"] for e in errors]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    threshold = max(mean + 2 * std, min_threshold)
    bad = [e["frame_idx"] for e in errors if e["error"] > threshold]
    return threshold, bad


def detect_area_anomalies(
    results: List[FrameAnalysis], min_threshold: float = 0.02
) -> Tuple[float, List[int]]:
    vals = [r.bbox_area_norm for r in results if r.bbox_area_norm is not None]
    if not vals:
        return 0.0, []
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    threshold = max(mean + 2 * std, mean + min_threshold)
    bad_frames = [
        r.frame_idx
        for r in results
        if r.bbox_area_norm is not None and r.bbox_area_norm > threshold
    ]
    return threshold, bad_frames


def plot_center_errors(errors: List[Dict], anomalies: List[int], out_path: str) -> None:
    if not plt or not errors:
        if not plt:
            print("matplotlib not available; skipping center error plot.")
        return
    xs = [e["frame_idx"] for e in errors]
    ys = [e["error_pct"] for e in errors]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="Center error (%)")
    if anomalies:
        bad_y = [ys[xs.index(i)] for i in anomalies if i in xs]
        plt.scatter(anomalies, bad_y, color="red", label="Anomaly", zorder=3)
    plt.xlabel("Frame")
    plt.ylabel("Error (% of half-diagonal)")
    plt.title("Centering error per frame")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_area_trend(
    results: List[FrameAnalysis], anomalies: List[int], out_path: str
) -> None:
    if not plt or not results:
        if not plt:
            print("matplotlib not available; skipping area trend plot.")
        return
    valid = [r for r in results if r.bbox_area_norm is not None]
    if not valid:
        return
    xs = [r.frame_idx for r in valid]
    ys = [r.bbox_area_norm for r in valid]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="Normalized highlight area")
    if len(valid) >= 2:
        y0, y1 = ys[0], ys[-1]
        trend = [y0 + (y1 - y0) * i / (len(ys) - 1) for i in range(len(ys))]
        plt.plot(xs, trend, "--", label="Linear trend (expected zoom path)")
    if anomalies:
        bad_y = [ys[xs.index(i)] for i in anomalies if i in xs]
        plt.scatter(anomalies, bad_y, color="red", label="Area anomaly", zorder=3)
    plt.xlabel("Frame")
    plt.ylabel("Area (fraction of frame)")
    plt.title("Highlight area over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize(
    results: List[FrameAnalysis],
    errors: List[Dict],
    anomalies: List[int],
    area_anomalies: List[int],
    out_path: str,
    skipped_frames: int = 0,
) -> None:
    with open(out_path, "w") as f:
        if not errors:
            f.write("No center data found.\n")
            return
        mean_err = sum(e["error_pct"] for e in errors) / len(errors)
        max_err = max(e["error_pct"] for e in errors)
        f.write(f"Frames analyzed: {len(results)}\n")
        if skipped_frames:
            f.write(f"Skipped duplicate frames: {skipped_frames}\n")
        f.write(f"Mean center error: {mean_err:.2f}% of half-diagonal\n")
        f.write(f"Max center error: {max_err:.2f}% of half-diagonal\n")
        if anomalies:
            f.write(f"Anomalous frames (> threshold): {', '.join(map(str, anomalies))}\n")
        else:
            f.write("No center anomalies detected.\n")

        areas = [r.bbox_area_norm for r in results if r.bbox_area_norm is not None]
        if areas:
            mean_area = sum(areas) / len(areas)
            diffs = [abs(areas[i + 1] - areas[i]) for i in range(len(areas) - 1)]
            smoothness = sum(diffs) / len(diffs) if diffs else 0.0
            f.write(f"Mean normalized area: {mean_area:.4f}\n")
            f.write(f"Average frame-to-frame area delta: {smoothness:.6f}\n")

            if len(areas) >= 2:
                trend_diffs = []
                y0, y1 = areas[0], areas[-1]
                for i, val in enumerate(areas):
                    expected = y0 + (y1 - y0) * i / (len(areas) - 1)
                    trend_diffs.append(abs(val - expected))
                trend_error = sum(trend_diffs) / len(trend_diffs)
                f.write(f"Mean deviation from linear zoom trend: {trend_error:.6f}\n")

            if area_anomalies:
                f.write(
                    f"Area anomalies (> threshold): {', '.join(map(str, area_anomalies))}\n"
                )
            else:
                f.write("No area anomalies detected.\n")
        else:
            f.write("No highlight boxes detected; cannot evaluate area/zoom.\n")


def run_full_diagnostics(
    frames_dir: str = "frames",
    centers_csv: str = "centers.csv",
    out_dir: str = "diagnostics",
    skip_duplicates: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    centers = load_centers(centers_csv)
    if not centers:
        print("No centers found; ensure centers.csv exists (run wikiclip with --center-debug).")
        return
    errors = compute_center_errors(centers)
    threshold, anomalies = detect_anomalies(errors)

    print(f"Detected {len(anomalies)} center anomalies (threshold={threshold:.4f}).")

    heatmap_path = os.path.join(out_dir, "center_heatmap.png")
    build_heatmap(centers, heatmap_path)

    results = analyze_frames(frames_dir, centers, out_dir, skip_duplicates=skip_duplicates)
    skipped = 0
    if skip_duplicates:
        frame_count = len(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        skipped = frame_count - len(results)

    plot_center_errors(errors, anomalies, os.path.join(out_dir, "center_errors.png"))

    area_threshold, area_anomalies = detect_area_anomalies(results)
    if area_anomalies:
        print(f"Detected {len(area_anomalies)} area anomalies (threshold={area_threshold:.4f}).")
    plot_area_trend(results, area_anomalies, os.path.join(out_dir, "area_trend.png"))

    summarize(
        results,
        errors,
        anomalies,
        area_anomalies,
        os.path.join(out_dir, "anomalies.txt"),
        skipped_frames=skipped,
    )

    print(f"Diagnostics written to {out_dir}/")
