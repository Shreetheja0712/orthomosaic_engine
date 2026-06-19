"""
Stage 5b — Geometric Verification using PoseLib

Replaces COLMAP's built-in RANSAC (pycolmap.verify_matches) with PoseLib's
LO-RANSAC + non-linear refinement. PoseLib's minimal solvers are faster and
more numerically stable than COLMAP's defaults, which matters most exactly
where nadir drone imagery is weakest: near-planar scenes (flat farmland),
where 5-point essential matrix estimation is close to degenerate.

Since intrinsics are known (single shared OPENCV camera from db_importer),
this uses poselib.estimate_relative_pose() — the essential-matrix path with
known calibration — rather than estimate_fundamental(). This is what
PoseLib's own docs recommend whenever calibration is available, and it
gives a real (R, t) pose, not just an F matrix, which COLMAP's
two_view_geometries table can store directly.

Flow:
    for each (image_a, image_b) pair in matches table:
        load keypoints for both images
        load match indices for the pair
        undistort/index into pixel coords
        poselib.estimate_relative_pose(pts_a, pts_b, camera_a, camera_b)
        keep inlier matches only
        write pose + inliers into two_view_geometries

This module is called instead of pycolmap.verify_matches() from
db_importer.py. It writes directly into the same two_view_geometries table,
so the COLMAP incremental mapper (Stage 6) cannot tell the difference.
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# COLMAP two_view_geometry "config" enum (colmap/src/colmap/scene/two_view_geometry.h)
# 0 = UNDEFINED, 1 = DEGENERATE, 2 = CALIBRATED, 3 = UNCALIBRATED,
# 4 = PLANAR, 5 = PANORAMIC, 6 = PLANAR_OR_PANORAMIC, 7 = WATERMARK, 8 = MULTIPLE
CONFIG_CALIBRATED = 2

# Default PoseLib RANSAC options tuned for nadir drone imagery.
# max_epipolar_error in pixels — LightGlue matches are already clean,
# so a tight threshold here further rejects any remaining outliers.
DEFAULT_RANSAC_OPTIONS = {
    "max_epipolar_error": 1.5,
    "success_prob": 0.9999,
    "min_iterations": 100,
    "max_iterations": 10000,
}


def _check_poselib():
    try:
        import poselib
        return poselib
    except ImportError as exc:
        raise ImportError(
            "poselib not installed.\n"
            "Run:  pip install poselib"
        ) from exc


def _unpack_params(blob: bytes) -> Tuple[float, ...]:
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", blob)


def _build_poselib_camera(poselib_module, model_id: int, width: int, height: int,
                           params: Tuple[float, ...]):
    """
    Build a poselib.Camera from COLMAP camera model id + params.
    Only OPENCV (model_id=4) is needed since db_importer always writes
    a single shared OPENCV camera, but PINHOLE/SIMPLE_RADIAL are mapped too
    for forward compatibility.
    """
    model_name_by_id = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
    }
    model_name = model_name_by_id.get(model_id, "OPENCV")
    return poselib_module.Camera(model_name, list(params), int(width), int(height))


def _pack_matches(matches: np.ndarray) -> bytes:
    return matches.astype("uint32").tobytes()


def _pack_pose_blobs(pose) -> dict:
    """
    Extract qvec (wxyz) and tvec from a poselib.CameraPose and pack them as
    float64 blobs in COLMAP's expected layout.
    """
    q = np.asarray(pose.q, dtype="float64")   # (4,) wxyz
    t = np.asarray(pose.t, dtype="float64")   # (3,)
    return {
        "qvec": q.tobytes(),
        "tvec": t.tobytes(),
    }


def verify_matches_poselib(
    db_path: Path,
    ransac_options: Optional[dict] = None,
) -> dict:
    """
    Run PoseLib geometric verification on all pairs in the `matches` table
    of a COLMAP database, writing results into `two_view_geometries`.

    Args:
        db_path        : path to database.db (already populated by db_importer
                          with cameras, images, keypoints, descriptors, matches)
        ransac_options  : override defaults in DEFAULT_RANSAC_OPTIONS

    Returns:
        dict with verification stats (pairs_verified, pairs_rejected, total_inliers)
    """
    poselib = _check_poselib()
    import pycolmap as _pycolmap
    opts = {**DEFAULT_RANSAC_OPTIONS, **(ransac_options or {})}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # ── load cameras keyed by image_id ───────────────────────────────────────
    image_camera: dict[int, dict] = {}
    for image_id, camera_id, width, height, model, params_blob in cur.execute("""
        SELECT images.image_id, images.camera_id, cameras.width, cameras.height,
               cameras.model, cameras.params
        FROM images JOIN cameras ON images.camera_id = cameras.camera_id
    """):
        params = _unpack_params(params_blob)
        image_camera[image_id] = {
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "model": model,
            "params": params,
        }

    # ── load keypoints per image_id (lazy cache) ─────────────────────────────
    keypoint_cache: dict[int, np.ndarray] = {}

    def get_keypoints(image_id: int) -> np.ndarray:
        if image_id not in keypoint_cache:
            row = cur.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)
            ).fetchone()
            if row is None:
                keypoint_cache[image_id] = np.zeros((0, 2), dtype="float64")
            else:
                rows, cols, data = row
                arr = np.frombuffer(data, dtype="float32").reshape(rows, cols)
                keypoint_cache[image_id] = arr[:, :2].astype("float64")  # x, y only
        return keypoint_cache[image_id]

    poselib_camera_cache: dict[int, "poselib.Camera"] = {}

    def get_poselib_camera(image_id: int):
        cam_id = image_camera[image_id]["camera_id"]
        if cam_id not in poselib_camera_cache:
            c = image_camera[image_id]
            poselib_camera_cache[cam_id] = _build_poselib_camera(
                poselib, c["model"], c["width"], c["height"], c["params"]
            )
        return poselib_camera_cache[cam_id]

    # ── iterate over all match pairs ──────────────────────────────────────────
    pair_rows = cur.execute("SELECT pair_id, rows, cols, data FROM matches").fetchall()

    print(f"[geometric_verification] PoseLib verifying {len(pair_rows)} pairs "
          f"(max_epipolar_error={opts['max_epipolar_error']}px)...")

    t0 = time.perf_counter()
    pairs_verified = 0
    pairs_rejected = 0
    total_inliers = 0

    for pair_id, n_matches, cols, data in pair_rows:
        # pair_id_to_image_pair is the version-correct inverse of whatever
        # encoding pycolmap.Database.write_matches() used to produce pair_id
        # in the first place — re-deriving the formula by hand is fragile
        # across COLMAP versions (the encoding constant has changed before).
        image_id_a, image_id_b = _pycolmap.pair_id_to_image_pair(pair_id)

        if image_id_a not in image_camera or image_id_b not in image_camera:
            pairs_rejected += 1
            continue

        matches = np.frombuffer(data, dtype="uint32").reshape(n_matches, cols)
        if len(matches) < 5:
            # Essential matrix needs minimum 5 correspondences
            pairs_rejected += 1
            continue

        kpts_a = get_keypoints(image_id_a)
        kpts_b = get_keypoints(image_id_b)

        pts_a = kpts_a[matches[:, 0]]
        pts_b = kpts_b[matches[:, 1]]

        cam_a = get_poselib_camera(image_id_a)
        cam_b = get_poselib_camera(image_id_b)

        try:
            pose, info = poselib.estimate_relative_pose(
                pts_a, pts_b, cam_a, cam_b, ransac_opt=opts
            )
        except Exception:
            pairs_rejected += 1
            continue

        inlier_mask = np.asarray(info.get("inliers", []), dtype=bool)
        n_inliers = int(inlier_mask.sum()) if inlier_mask.size else 0

        # Reject pairs PoseLib could not confidently verify.
        if n_inliers < 15:
            pairs_rejected += 1
            continue

        inlier_matches = matches[inlier_mask].astype("uint32")
        pose_blobs = _pack_pose_blobs(pose)

        cur.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                pair_id,
                len(inlier_matches),
                2,
                _pack_matches(inlier_matches),
                CONFIG_CALIBRATED,
                None,            # F not computed in the calibrated path
                None,            # E not exposed directly by estimate_relative_pose
                None,            # H not applicable (non-planar config)
                pose_blobs["qvec"],
                pose_blobs["tvec"],
            ),
        )

        pairs_verified += 1
        total_inliers += n_inliers

    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - t0
    print(f"[geometric_verification] Verified: {pairs_verified}  "
          f"Rejected: {pairs_rejected}  Total inliers: {total_inliers}  "
          f"Time: {elapsed:.1f}s")

    return {
        "pairs_verified": pairs_verified,
        "pairs_rejected": pairs_rejected,
        "total_inliers": total_inliers,
        "elapsed_seconds": elapsed,
    }