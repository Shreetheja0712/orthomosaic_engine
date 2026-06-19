"""
Stage 6, Step 4 — COLMAP incremental SfM with agricultural optimizations.

This is the safe path — always runs when has_rtk=False,
and as fallback when GLOMAP fails validation.

5 risk-free optimizations applied:

  1. Keyframe-only mapping
     Only keyframes go through full incremental SfM.
     Non-keyframes registered separately via fast PnP (Step 5).
     Safe at 80% overlap — every ground point still in 8+ keyframes.

  2. GPS-guided init pair
     Precomputed in keyframes.py. Avoids COLMAP scoring all pairs.
     Safe because GPS baseline is physically meaningful.

  3. Bundle adjustment frequency tuning
     ba_local_num_images    = 12  (default 6)
     opt.ba_global_frames_ratio = 1.3  (default 1.1)
     Run BA less often but with more context each time.
     Safe for flat terrain + clean LightGlue matches.

  4. Tight reprojection filter
     mapper.filter_max_reproj_error = 2.0  (default 4.0)
     Safe because ALIKED+LightGlue produces clean matches.
     Would be risky with noisy SIFT matches — not risky here.

  5. All CPU cores
     num_threads = -1
     No risk. Linear speedup with core count.

GPS prior weight is set by pose_priors.py before this runs:
  RTK  → 1e6 covariance diagonal → strong anchor
  GPS  → 1e2 covariance diagonal → loose regularizer

use_prior_position = True is required to activate the priors.
This was confirmed necessary — default is False and priors are ignored
without it.

API surface confirmed against pycolmap 4.0.4:
  opt.mapper.ba_local_num_images       → int
  opt.mapper.filter_max_reproj_error   → float
  opt.ba_global_frames_ratio           → float  (renamed from ba_global_images_ratio)
  opt.init_image_id1 / init_image_id2  → int
  opt.use_prior_position               → bool
  opt.num_threads                      → int
  pycolmap.incremental_mapping()       → dict[int, Reconstruction]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..ingestion.capture import Capture


def _best_reconstruction(reconstructions: dict) -> Optional[object]:
    if not reconstructions:
        return None
    return max(reconstructions.values(), key=lambda r: r.num_reg_images)


def _resolve_image_id(db, capture: Capture) -> Optional[int]:
    """
    Look up the COLMAP image_id for a Capture by its name convention.
    db_importer names images as <capture_id><ext>.
    """
    ext  = Path(capture.rgb).suffix if capture.rgb else ".jpg"
    name = f"{capture.capture_id}{ext}"
    img  = db.read_image_with_name(name)
    return img.image_id if img is not None else None


def run_colmap_incremental(
    database_path  : str,
    image_dir      : str,
    output_dir     : str,
    keyframes      : List[Capture],
    init_pair      : Optional[Tuple[Capture, Capture]] = None,
) -> Optional[object]:
    """
    Step 4 — COLMAP incremental SfM on keyframes only.

    Args:
        database_path : COLMAP .db with features, matches, pose priors
        image_dir     : directory containing actual image files
                        (COLMAP requires real files on disk matching db names)
        output_dir    : where to write the sparse/ model
        keyframes     : keyframe Capture list for SfM
        init_pair     : (cap_a, cap_b) from find_gps_guided_init_pair()
                        or None to let COLMAP choose

    Returns:
        Best Reconstruction, or None if mapper produced nothing.
    """
    import pycolmap

    output_path = Path(output_dir) / "colmap"
    output_path.mkdir(parents=True, exist_ok=True)

    # Build keyframe image name list
    image_names = []
    for cap in keyframes:
        ext = Path(cap.rgb).suffix if cap.rgb else ".jpg"
        image_names.append(f"{cap.capture_id}{ext}")

    # ── Options ───────────────────────────────────────────────────────────────
    opt = pycolmap.IncrementalPipelineOptions()

    # Optimization 1 — keyframe image list (image_names filters what mapper sees)
    opt.image_names = image_names

    # Optimization 3 — BA frequency tuning (safe for flat terrain + clean matches)
    opt.mapper.ba_local_num_images   = 12    # default 6
    opt.ba_global_frames_ratio       = 1.3   # default 1.1

    # Optimization 4 — tight reprojection filter (safe with ALIKED+LightGlue)
    opt.mapper.filter_max_reproj_error = 2.0  # default 4.0

    # Optimization 5 — all CPU cores
    opt.num_threads = -1

    # GPS/RTK priors — activate prior position constraints
    # pose_priors.py already wrote priors into the database
    # this flag tells BA to actually use them during optimization
    opt.use_prior_position = True

    # Optimization 2 — GPS-guided init pair
    if init_pair is not None:
        db = pycolmap.Database.open(str(database_path))
        id1 = _resolve_image_id(db, init_pair[0])
        id2 = _resolve_image_id(db, init_pair[1])
        db.close()

        if id1 is not None and id2 is not None:
            opt.init_image_id1 = id1
            opt.init_image_id2 = id2
            print(f"[colmap] GPS-guided init pair: "
                  f"{init_pair[0].capture_id} (id={id1}) <-> "
                  f"{init_pair[1].capture_id} (id={id2})")
        else:
            print("[colmap] Warning: could not resolve init pair image ids. "
                  "COLMAP will select its own init pair.")

    # ── Run mapper ────────────────────────────────────────────────────────────
    print(f"[colmap] Running incremental SfM on {len(keyframes)} keyframes...")
    print(f"[colmap] Options: ba_local_num_images={opt.mapper.ba_local_num_images}, "
          f"ba_global_frames_ratio={opt.ba_global_frames_ratio}, "
          f"filter_max_reproj_error={opt.mapper.filter_max_reproj_error}, "
          f"use_prior_position={opt.use_prior_position}")

    try:
        reconstructions = pycolmap.incremental_mapping(
            database_path = str(database_path),
            image_path    = str(image_dir),
            output_path   = str(output_path),
            options       = opt,
        )
    except Exception as e:
        print(f"[colmap] incremental_mapping failed: {e}")
        return None

    recon = _best_reconstruction(reconstructions)

    if recon is None:
        print("[colmap] Mapper produced no reconstruction.")
        return None

    n_reg   = recon.num_reg_images
    n_total = len(keyframes)
    print(f"[colmap] Registered {n_reg}/{n_total} keyframes "
          f"({n_reg/n_total*100:.1f}%)")

    if n_reg < n_total * 0.70:
        print(f"[colmap] Warning: fewer than 70% of keyframes registered. "
              f"Check feature matching quality.")

    return recon
