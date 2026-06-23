"""
Stage 6, Steps 1-2 — Keyframe selection + GPS-guided initial pair

Step 1: Order captures by GPS flight path, select every Nth as a keyframe.
        80% forward/side overlap on agricultural missions means every ground
        point is visible in 8+ images, so dropping 2-in-3 images loses zero
        coverage while cutting SfM workload ~3x.

Step 2: Pick the COLMAP initial image pair explicitly from GPS data instead
        of letting COLMAP search all pairs for the best starting point.
        Two keyframes near the field centroid, with a baseline that's neither
        too short (weak triangulation) nor too long (matches may not exist),
        make a fast, reliable seed for incremental reconstruction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..ingestion.capture import Capture
from ..features.neighbors import haversine_distance as _haversine_distance


# Baseline window (meters) for a "good" GPS init pair.
# Too short -> triangulation angle too small, unstable.
# Too long  -> may not actually overlap / match.
DEFAULT_MIN_BASELINE_M = 5.0
DEFAULT_MAX_BASELINE_M = 40.0


def _with_gps(captures: List[Capture]) -> List[Capture]:
    return [c for c in captures if c.latitude is not None and c.longitude is not None]


def _order_by_flight_path(captures: List[Capture]) -> List[Capture]:
    """
    Order captures along the flight path using a nearest-neighbor walk
    starting from one extreme corner of the GPS bounding box.

    This is NOT the same as sorting by capture_id/filename — drone flight
    logs aren't always captured in strict numeric order, and a true
    spatial walk gives a much more meaningful "every 3rd" keyframe sample
    than filename order would on a boustrophedon (lawnmower) flight path.
    """
    gps_captures = _with_gps(captures)
    if len(gps_captures) <= 2:
        return list(gps_captures)

    # Start from the capture with the smallest (lat + lon) — a stable,
    # deterministic corner of the bounding box.
    start = min(gps_captures, key=lambda c: (c.latitude, c.longitude))

    remaining = set(c.capture_id for c in gps_captures)
    by_id = {c.capture_id: c for c in gps_captures}

    ordered = [start]
    remaining.discard(start.capture_id)
    current = start

    while remaining:
        nearest_id = min(
            remaining,
            key=lambda cid: _haversine_distance(
                current.latitude, current.longitude,
                by_id[cid].latitude, by_id[cid].longitude,
            ),
        )
        current = by_id[nearest_id]
        ordered.append(current)
        remaining.discard(nearest_id)

    return ordered


def select_keyframes(
    captures: List[Capture],
    interval: int = 3,
) -> Tuple[List[Capture], List[Capture]]:
    """
    Step 1 — Select every `interval`-th capture (by flight-path GPS order)
    as a keyframe for full SfM. The rest are registered later via fast PnP
    (Step 5) without full bundle adjustment cost.

    Args:
        captures : full Capture list (with GPS)
        interval : keep 1 in every `interval` images as a keyframe.
                   interval=3 matches the 80%-overlap assumption: every
                   ground point still visible in 8+ keyframes.

    Returns:
        (keyframes, non_keyframes) — both lists of Capture, in flight-path order.
        Captures missing GPS are excluded entirely from both lists (cannot be
        spatially ordered) and should be handled separately by the caller.
    """
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")

    ordered = _order_by_flight_path(captures)

    keyframes = [c for i, c in enumerate(ordered) if i % interval == 0]
    non_keyframes = [c for i, c in enumerate(ordered) if i % interval != 0]

    no_gps_count = len(captures) - len(ordered)
    if no_gps_count:
        print(f"[keyframes] Warning: {no_gps_count} captures have no GPS — "
              f"excluded from keyframe selection.")

    print(f"[keyframes] {len(ordered)} GPS-ordered captures -> "
          f"{len(keyframes)} keyframes + {len(non_keyframes)} non-keyframes "
          f"(interval={interval})")

    return keyframes, non_keyframes


def write_keyframe_list(keyframes: List[Capture], output_path: str) -> str:
    """
    Write keyframe image filenames to a plain text file, one per line.
    This matches the `--image_list_path` format expected by COLMAP's
    mapper / image_registrator commands, and is also used directly by
    db_importer's image-name convention (<capture_id>.jpg).
    """
    from pathlib import Path

    lines = []
    for cap in keyframes:
        ext = Path(cap.rgb).suffix if cap.rgb else ".jpg"
        lines.append(f"{cap.capture_id}{ext}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"[keyframes] Wrote {len(lines)} keyframe names to {out}")
    return str(out)


def _field_centroid(captures: List[Capture]) -> Tuple[float, float]:
    gps = _with_gps(captures)
    lat = sum(c.latitude for c in gps) / len(gps)
    lon = sum(c.longitude for c in gps) / len(gps)
    return lat, lon


def find_gps_guided_init_pair(
    keyframes: List[Capture],
    min_baseline_m: float = DEFAULT_MIN_BASELINE_M,
    max_baseline_m: float = DEFAULT_MAX_BASELINE_M,
) -> Optional[Tuple[Capture, Capture]]:
    """
    Step 2 — Pick a COLMAP initial image pair using GPS instead of letting
    COLMAP score every candidate pair (init_num_trials=200 by default).

    Strategy:
        1. Find the keyframe closest to the field centroid (image A).
        2. Among keyframes within [min_baseline_m, max_baseline_m] of A,
           pick the one closest to the midpoint of that baseline window
           (i.e. closest to (min+max)/2 meters away) as image B.
           This avoids both near-degenerate (too-short) baselines and
           weak-overlap (too-long) baselines.

    Returns:
        (capture_a, capture_b) or None if no keyframe pair satisfies the
        baseline window (caller should fall back to COLMAP's own search
        by leaving init_image_id1/2 unset).
    """
    gps_keyframes = _with_gps(keyframes)
    if len(gps_keyframes) < 2:
        print("[keyframes] Not enough GPS keyframes for GPS-guided init pair.")
        return None

    center_lat, center_lon = _field_centroid(gps_keyframes)

    # Image A: keyframe nearest the field centroid.
    cap_a = min(
        gps_keyframes,
        key=lambda c: _haversine_distance(center_lat, center_lon, c.latitude, c.longitude),
    )

    # Candidate B: keyframes within the good-baseline window from A.
    target_baseline = (min_baseline_m + max_baseline_m) / 2.0
    candidates = []
    for c in gps_keyframes:
        if c.capture_id == cap_a.capture_id:
            continue
        dist = _haversine_distance(cap_a.latitude, cap_a.longitude, c.latitude, c.longitude)
        if min_baseline_m <= dist <= max_baseline_m:
            candidates.append((abs(dist - target_baseline), dist, c))

    if not candidates:
        print(f"[keyframes] No keyframe found within baseline window "
              f"[{min_baseline_m}, {max_baseline_m}]m of field-center keyframe "
              f"{cap_a.capture_id}. Falling back to COLMAP's own pair search.")
        return None

    candidates.sort(key=lambda x: x[0])
    _, chosen_dist, cap_b = candidates[0]

    print(f"[keyframes] GPS-guided init pair: {cap_a.capture_id} <-> {cap_b.capture_id}  "
          f"(baseline: {chosen_dist:.1f}m)")

    return cap_a, cap_b