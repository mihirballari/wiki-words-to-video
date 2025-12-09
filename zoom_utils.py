"""
Helpers for building smooth zoom filters for ffmpeg.
"""

from typing import Optional, Tuple


def resolve_zoom_range(
    base_zoom: float,
    zoom_end: Optional[float] = None,
    zoom_in: Optional[float] = None,
    zoom_out: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the (start, end) zoom factors for the final video.

    Rules:
      - Start uses `base_zoom` (the capture-time zoom).
      - `zoom_end` wins if provided.
      - `zoom_in` multiplies base_zoom.
      - `zoom_out` divides base_zoom.
    """
    end = base_zoom

    if zoom_end is not None:
        end = zoom_end
    elif zoom_in is not None:
        end = base_zoom * zoom_in
    elif zoom_out is not None:
        end = base_zoom / zoom_out

    return base_zoom, end


def build_zoompan_filter(
    frame_count: int,
    width: int,
    height: int,
    zoom_start: float,
    zoom_end: float,
    fps: int,
) -> Optional[str]:
    """
    Return a zoompan filter string that linearly interpolates zoom across
    all frames while keeping the center fixed. If no change is needed, return None.
    """
    if frame_count <= 0:
        return None

    if abs(zoom_end - zoom_start) < 1e-6:
        return None

    # Avoid zoom values < 1.0, since that would show beyond the frame bounds.
    z0 = max(zoom_start, 1.0)
    z1 = max(zoom_end, 1.0)

    span = max(frame_count - 1, 1)
    zoom_expr = f"{z0} + ({z1} - {z0})*on/{span}"

    # Center the crop for every frame so the highlight stays in place.
    x_expr = "(iw - iw/zoom)/2"
    y_expr = "(ih - ih/zoom)/2"

    return (
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d=1:s={width}x{height}:fps={fps}"
    )
