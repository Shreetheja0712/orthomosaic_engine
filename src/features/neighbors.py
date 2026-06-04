import math
from typing import List, Tuple

from ..ingestion.capture import Capture


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate ground distance in meters between two GPS coordinates.
    """
    earth_radius_m = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return earth_radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_neighbor_pairs(
    captures: List[Capture],
    n_neighbors: int = 8,
) -> List[Tuple[str, str]]:
    """
    For each capture, find N nearest captures by GPS distance.
    Returns unique pairs (capture_id_a, capture_id_b) to match.

    This reduces matching pairs from O(n^2) to O(n * n_neighbors).
    For 900 images: 810,000 pairs -> ~7,200 pairs.
    """
    with_gps = [c for c in captures if c.latitude is not None and c.longitude is not None]
    without_gps = [c for c in captures if c.latitude is None or c.longitude is None]

    if without_gps:
        print(f"[neighbors] Warning: {len(without_gps)} captures have no GPS - excluded from neighbor filter.")

    if not with_gps:
        raise ValueError("No captures with GPS coordinates found. Cannot build neighbor pairs.")

    pairs: set[Tuple[str, str]] = set()

    for i, cap_a in enumerate(with_gps):
        distances = []
        for j, cap_b in enumerate(with_gps):
            if i == j:
                continue
            dist = _haversine_distance(
                cap_a.latitude,
                cap_a.longitude,
                cap_b.latitude,
                cap_b.longitude,
            )
            distances.append((dist, cap_b.capture_id))

        for _, neighbor_id in sorted(distances, key=lambda x: x[0])[:n_neighbors]:
            pair = tuple(sorted([cap_a.capture_id, neighbor_id]))
            pairs.add(pair)

    pair_list = sorted(pairs)
    exhaustive_count = len(with_gps) * (len(with_gps) - 1) // 2
    print(f"[neighbors] {len(with_gps)} captures -> {len(pair_list)} GPS-filtered pairs "
          f"(vs {exhaustive_count} exhaustive)")
    return pair_list
