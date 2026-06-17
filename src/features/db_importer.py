"""
Stage 5 (bridge) — Import ALIKED + LightGlue results into a COLMAP database

COLMAP's incremental mapper (Stage 6) reads from a .db SQLite file.
This module bridges the gap:

    features/               <- written by extractor.py (ALIKED .h5 files)
    matches.h5              <- written by matcher.py   (LightGlue matches)
         |
         v
    database.db             <- COLMAP SQLite database
         |
         v
    COLMAP RANSAC geometric verification  (run inside this module via pycolmap)
         |
         v
    Stage 6: pycolmap incremental mapper reads database.db

Flow inside this module:
    1. Create COLMAP database
    2. Add ONE shared camera (OPENCV model, intrinsics from EXIF or defaults)
    3. Add all images with their camera ID
    4. Import ALIKED keypoints per image
    5. Import LightGlue match indices per pair
    6. Run COLMAP RANSAC geometric verification on all imported pairs
       (two_view_geometries table populated — mapper requires this)

After this module, database.db is indistinguishable from one produced by
COLMAP's own feature_extractor + exhaustive_matcher pipeline, except that the
keypoints are ALIKED (better) and the matches are LightGlue (fewer, cleaner).
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

from ..ingestion.capture import Capture


# ── COLMAP camera model IDs (from colmap/src/colmap/sensor/models.h) ────────
OPENCV_MODEL_ID = 4   # fx, fy, cx, cy, k1, k2, p1, p2

# Default focal length multiplier when EXIF focal length is unavailable.
# 1.2 × sensor_diagonal is a reasonable prior for drone cameras.
FOCAL_MULTIPLIER_DEFAULT = 1.2


# ── SQLite / COLMAP database helpers ─────────────────────────────────────────

def _create_database(db_path: Path) -> sqlite3.Connection:
    """
    Create a fresh COLMAP SQLite database with the standard schema.
    If the file already exists it is deleted first so Stage 6 gets a clean run.
    """
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            model       INTEGER NOT NULL,
            width       INTEGER NOT NULL,
            height      INTEGER NOT NULL,
            params      BLOB,
            prior_focal_length INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS images (
            image_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            camera_id   INTEGER NOT NULL,
            prior_qw    REAL,
            prior_qx    REAL,
            prior_qy    REAL,
            prior_qz    REAL,
            prior_tx    REAL,
            prior_ty    REAL,
            prior_tz    REAL
        );

        CREATE TABLE IF NOT EXISTS keypoints (
            image_id    INTEGER PRIMARY KEY,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB
        );

        CREATE TABLE IF NOT EXISTS descriptors (
            image_id    INTEGER PRIMARY KEY,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB
        );

        CREATE TABLE IF NOT EXISTS matches (
            pair_id     INTEGER PRIMARY KEY,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB
        );

        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id     INTEGER PRIMARY KEY,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB,
            config      INTEGER,
            F           BLOB,
            E           BLOB,
            H           BLOB,
            qvec        BLOB,
            tvec        BLOB
        );
    """)
    conn.commit()
    return conn


def _image_pair_id(image_id_1: int, image_id_2: int) -> int:
    """
    COLMAP pair ID encoding (matches colmap/src/colmap/util/misc.h):
        pair_id = image_id_1 * kMaxNumImages + image_id_2
        where kMaxNumImages = 2^20  (1048576)
    The smaller id is always image_id_1.
    """
    MAX_IMAGES = 2 ** 20
    a, b = min(image_id_1, image_id_2), max(image_id_1, image_id_2)
    return a * MAX_IMAGES + b


def _pack_params(params: Tuple[float, ...]) -> bytes:
    """Pack camera intrinsics as little-endian float64 blob."""
    return struct.pack(f"<{len(params)}d", *params)


def _pack_keypoints(kpts: np.ndarray) -> bytes:
    """
    COLMAP keypoints blob: float32 (N, 6) — x, y, scale, orientation, (unused x2)
    We only have x, y from ALIKED; scale and orientation are set to 1.0 / 0.0.
    Descriptor is stored separately; cols=6 is the COLMAP convention.
    """
    n = len(kpts)
    full = np.zeros((n, 6), dtype="float32")
    full[:, 0] = kpts[:, 0]   # x
    full[:, 1] = kpts[:, 1]   # y
    full[:, 2] = 1.0           # scale
    full[:, 3] = 0.0           # orientation
    return full.tobytes()


def _pack_matches(matches: np.ndarray) -> bytes:
    """
    COLMAP matches blob: uint32 (M, 2) — [ kp_idx_in_A,  kp_idx_in_B ]
    """
    return matches.astype("uint32").tobytes()


# ── public API ───────────────────────────────────────────────────────────────

def import_to_colmap(
    captures: List[Capture],
    output_dir: str,
    focal_length_px: Optional[float] = None,
    run_geometric_verification: bool = True,
) -> Path:
    """
    Import ALIKED features + LightGlue matches into a COLMAP database.

    Args:
        captures                  : Capture list from ingestion
        output_dir                : same root dir as extract_features / match_features
        focal_length_px           : known focal length in pixels.
                                    If None, estimated from image size using
                                    FOCAL_MULTIPLIER_DEFAULT heuristic.
        run_geometric_verification: run COLMAP RANSAC to populate
                                    two_view_geometries (required for mapper).
                                    Set False to skip during testing.

    Returns:
        Path to database.db
    """
    out = Path(output_dir)
    features_dir = out / "features"
    matches_path = out / "matches.h5"
    db_path = out / "database.db"

    if not features_dir.exists():
        raise FileNotFoundError(f"Features not found: {features_dir}. Run extract_features() first.")
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches not found: {matches_path}. Run match_features() first.")

    t0 = time.perf_counter()
    print(f"[db_importer] Importing into COLMAP database: {db_path}")

    conn = _create_database(db_path)

    # ── 1. Read image dimensions from the first valid feature file ──────────
    # All RGB images share the same camera (SINGLE camera mode).
    sample_cap = next((c for c in captures if c.rgb), None)
    if sample_cap is None:
        raise ValueError("No captures with RGB images.")

    sample_h5 = features_dir / f"{sample_cap.capture_id}.h5"
    with h5py.File(sample_h5, "r") as f:
        img_w, img_h = int(f["image_size"][0]), int(f["image_size"][1])

    # ── 2. Camera — single shared OPENCV model ──────────────────────────────
    if focal_length_px is None:
        # Heuristic prior: focal ≈ 1.2 × max(width, height)
        focal_length_px = FOCAL_MULTIPLIER_DEFAULT * max(img_w, img_h)
        print(f"[db_importer] focal_length_px not provided — using heuristic: "
              f"{focal_length_px:.1f} px  ({img_w}×{img_h} image)")
    else:
        print(f"[db_importer] focal_length_px: {focal_length_px:.1f} px")

    cx, cy = img_w / 2.0, img_h / 2.0
    # OPENCV params: fx, fy, cx, cy, k1, k2, p1, p2  (all distortion = 0 initially)
    params = (focal_length_px, focal_length_px, cx, cy, 0.0, 0.0, 0.0, 0.0)

    conn.execute(
        "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?,?,?,?,?)",
        (OPENCV_MODEL_ID, img_w, img_h, _pack_params(params), 1),
    )
    camera_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()

    # ── 3. Images ────────────────────────────────────────────────────────────
    # Map capture_id → COLMAP image_id for pair-ID calculation later.
    cap_id_to_image_id: dict[str, int] = {}

    for cap in captures:
        # Image name = what the mapper will look for in the image folder.
        # We reuse <capture_id>.jpg  (same naming as the feature .h5 files).
        ext = Path(cap.rgb).suffix if cap.rgb else ".jpg"
        name = f"{cap.capture_id}{ext}"
        conn.execute(
            "INSERT OR IGNORE INTO images "
            "(name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (name, camera_id, None, None, None, None, None, None, None),
        )
        image_id = conn.execute("SELECT image_id FROM images WHERE name=?", (name,)).fetchone()[0]
        cap_id_to_image_id[cap.capture_id] = image_id

    conn.commit()
    print(f"[db_importer] Registered {len(captures)} images  (camera_id={camera_id})")

    # ── 4. Keypoints & descriptors ───────────────────────────────────────────
    kp_count_total = 0
    missing_feature_files = 0

    for cap in captures:
        h5_path = features_dir / f"{cap.capture_id}.h5"
        if not h5_path.exists():
            missing_feature_files += 1
            continue

        with h5py.File(h5_path, "r") as f:
            kpts = f["keypoints"][:]      # (N, 2)
            desc = f["descriptors"][:]    # (N, D)

        image_id = cap_id_to_image_id[cap.capture_id]
        n_kpts = len(kpts)

        conn.execute(
            "INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (image_id, n_kpts, 6, _pack_keypoints(kpts)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO descriptors (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (image_id, n_kpts, desc.shape[1], desc.astype("float32").tobytes()),
        )
        kp_count_total += n_kpts

    conn.commit()
    print(f"[db_importer] Keypoints imported: {kp_count_total}  "
          f"(missing feature files: {missing_feature_files})")

    # ── 5. Matches ───────────────────────────────────────────────────────────
    match_pair_count = 0

    with h5py.File(matches_path, "r") as mf:
        for id_a in mf.keys():
            if id_a not in cap_id_to_image_id:
                continue
            for id_b in mf[id_a].keys():
                if id_b not in cap_id_to_image_id:
                    continue

                matches = mf[id_a][id_b]["matches0"][:]   # (M, 2)  int32
                if len(matches) == 0:
                    continue

                img_id_a = cap_id_to_image_id[id_a]
                img_id_b = cap_id_to_image_id[id_b]
                pair_id = _image_pair_id(img_id_a, img_id_b)

                # COLMAP always stores matches with smaller image_id first
                if img_id_a > img_id_b:
                    matches = matches[:, ::-1]  # flip column order

                conn.execute(
                    "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?,?,?,?)",
                    (pair_id, len(matches), 2, _pack_matches(matches)),
                )
                match_pair_count += 1

    conn.commit()
    print(f"[db_importer] Match pairs imported: {match_pair_count}")

    # ── 6. Geometric verification (COLMAP RANSAC) ────────────────────────────
    if run_geometric_verification:
        conn.close()
        _run_geometric_verification(db_path)
    else:
        print("[db_importer] Geometric verification skipped (run_geometric_verification=False).")
        conn.close()

    elapsed = time.perf_counter() - t0
    print(f"[db_importer] Done. Database: {db_path}  Time: {elapsed:.1f}s")
    return db_path


def _run_geometric_verification(db_path: Path) -> None:
    """
    Run COLMAP RANSAC on all match pairs to populate two_view_geometries.
    Uses pycolmap — this is the ONE remaining place COLMAP is called in Stage 3-5.

    RANSAC here is cheap: LightGlue already removed outlier matches.
    This step just computes F/E matrices and stores verified inlier sets.
    """
    try:
        import pycolmap
    except ImportError as exc:
        raise ImportError(
            "pycolmap required for geometric verification.\n"
            "Run:  pip install pycolmap\n"
            "Or call import_to_colmap(..., run_geometric_verification=False) "
            "and run COLMAP verification manually."
        ) from exc

    print("[db_importer] Running COLMAP RANSAC geometric verification...")
    t = time.perf_counter()

    # pycolmap.verify_matches is the correct function for this.
    # It reads matches table and writes two_view_geometries.
    if hasattr(pycolmap, "verify_matches"):
        pycolmap.verify_matches(
            database_path=str(db_path),
            # Use default TwoViewGeometryOptions — good for nadir aerial
        )
    elif hasattr(pycolmap, "geometric_verification"):
        # Older pycolmap API name
        pycolmap.geometric_verification(database_path=str(db_path))
    else:
        # Fallback: call COLMAP CLI
        import subprocess
        result = subprocess.run(
            ["colmap", "matches_importer",
             "--database_path", str(db_path),
             "--match_type", "pairs"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP geometric verification failed:\n{result.stderr}")

    print(f"[db_importer] Geometric verification done. ({time.perf_counter() - t:.1f}s)")
