"""
Stage 6 — Structure-from-Motion pipeline.

Steps 1-2: keyframes.py    — GPS-ordered keyframe selection, GPS-guided init pair
Step  3a : pose_priors.py  — RTK/GPS position prior injection (weighted)
Step  3b+: (in progress)   — GLOMAP / COLMAP incremental, PnP registration,
                              final BA, GPS alignment
"""

from .keyframes import select_keyframes, write_keyframe_list, find_gps_guided_init_pair
from .pose_priors import inject_gps_priors, RTK_PRIOR_WEIGHT, GPS_PRIOR_WEIGHT

__all__ = [
    "select_keyframes",
    "write_keyframe_list",
    "find_gps_guided_init_pair",
    "inject_gps_priors",
    "RTK_PRIOR_WEIGHT",
    "GPS_PRIOR_WEIGHT",
]
