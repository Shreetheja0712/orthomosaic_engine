"""
Stage 6 — Structure-from-Motion pipeline.

Entry point:
    from src.sfm import run_sfm
    reconstruction = run_sfm(
        database_path = "mission/colmap.db",
        image_dir     = "mission/rgb_images/",
        output_dir    = "mission/sparse/",
        captures      = captures,
        has_rtk       = False,
    )

Internal steps:
    keyframes.py        Steps 1-2  keyframe selection + GPS init pair
    pose_priors.py      Step 3a    GPS/RTK prior injection
    glomap.py           Step 3b    GLOMAP fast path + 3-check validation
    colmap_mapper.py    Step 4     COLMAP incremental + 5 optimizations
    pnp_registration.py Step 5     PnP non-keyframe registration
    final_ba.py         Steps 6-7  final BA + GPS alignment
    pipeline.py                    orchestrator
"""

from .pipeline        import run_sfm
from .keyframes       import select_keyframes, write_keyframe_list, find_gps_guided_init_pair
from .pose_priors     import inject_gps_priors, RTK_PRIOR_WEIGHT, GPS_PRIOR_WEIGHT
from .glomap          import run_glomap, validate_glomap_reconstruction
from .colmap_mapper   import run_colmap_incremental
from .pnp_registration import register_non_keyframes
from .final_ba        import run_final_bundle_adjustment, align_to_gps

__all__ = [
    "run_sfm",
    "select_keyframes",
    "write_keyframe_list",
    "find_gps_guided_init_pair",
    "inject_gps_priors",
    "RTK_PRIOR_WEIGHT",
    "GPS_PRIOR_WEIGHT",
    "run_glomap",
    "validate_glomap_reconstruction",
    "run_colmap_incremental",
    "register_non_keyframes",
    "run_final_bundle_adjustment",
    "align_to_gps",
]
