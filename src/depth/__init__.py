"""
src/depth/__init__.py
=====================
Public API for Stage 8 — Depth Maps.

Single entry point: ``run_depth_pipeline()``.

Pipeline internals (in call order):
  1. ``depth_range.py``    — per-image percentile depth bounds
  2. ``image_prep.py``     — symlink images + export to OpenMVS .mvs
  3. ``openmvs_runner.py`` — run DensifyPointCloud (depth-only mode)
  4. Validate .dmap outputs exist and are non-empty
  5. Return .dmap paths for Stage 9

Stage 9 (DSM generation) consumes the list of .dmap paths:

    dmap_paths = run_depth_pipeline(reconstruction, captures, output_dir)
    dsm_path   = fuse_depth_maps(dmap_paths, output_dir)   # Stage 9

The .dmap boundary is the explicit contract between stages.
Stage 9 does not care how .dmap files were produced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .depth_range import compute_depth_ranges, print_depth_range_stats
from .dmap import DMap, read_dmap, write_dmap, dmap_from_openmvs_output
from .image_prep import prepare_openmvs_workspace
from .openmvs_runner import run_openmvs_depth_estimation

logger = logging.getLogger(__name__)

__all__ = [
    # Pipeline entry point
    "run_depth_pipeline",
    # Sub-module exports for direct access if needed
    "compute_depth_ranges",
    "print_depth_range_stats",
    "DMap",
    "read_dmap",
    "write_dmap",
    "dmap_from_openmvs_output",
    "prepare_openmvs_workspace",
    "run_openmvs_depth_estimation",
]


def run_depth_pipeline(
    reconstruction,
    captures: List,
    output_dir: str,
    colmap_sparse_dir: Optional[str] = None,
    use_gpu: bool = True,
    resolution_level: int = 1,
    num_neighbors: int = 5,
    openmvs_bin_dir: str = "",
    print_stats: bool = True,
) -> List[str]:
    """
    Run the complete Stage 8 depth map pipeline.

    Produces one .dmap file per RGB image in the reconstruction.
    Returns the list of .dmap paths for consumption by Stage 9.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        Georeferenced sparse reconstruction from Stage 6/7.
    captures : List[Capture]
        Capture list from Stage 2 (ingestion).
    output_dir : str
        Root output directory for this stage.  Will be created.
        Subdirectories created inside:
          <output_dir>/images/         symlinked RGB images
          <output_dir>/sparse/         COLMAP binary model (if exported)
          <output_dir>/scene.mvs       OpenMVS scene file
          <output_dir>/*.dmap          per-image depth maps
          <output_dir>/depth_ranges.ini per-image depth range config
    colmap_sparse_dir : str, optional
        Path to existing COLMAP binary model directory
        (cameras.bin / images.bin / points3D.bin).  If None, the
        reconstruction is exported automatically.
    use_gpu : bool
        Use GPU (CUDA) for depth estimation.  Default True.
        Setting False falls back to CPU — very slow, for debugging.
    resolution_level : int
        0 = full resolution, 1 = half resolution (default), 2 = quarter.
        Half resolution is the correct choice for a 16 GB GPU with
        ~500-900 images at agricultural drone altitudes.
    num_neighbors : int
        Number of neighbouring images used per depth estimate.
        5 is sufficient for 80% forward overlap grid flights.
    openmvs_bin_dir : str
        Path to directory containing OpenMVS binaries.  Empty string
        causes automatic search on PATH and common install locations.
    print_stats : bool
        If True, print depth range statistics after computing them.
        Useful for a quick sanity check before committing GPU hours.

    Returns
    -------
    List[str]
        Sorted list of absolute paths to produced .dmap files.
        Passes directly to Stage 9's fuse_depth_maps().

    Raises
    ------
    RuntimeError
        If OpenMVS binaries are not found, or if DensifyPointCloud
        fails, or if no .dmap files are produced.

    Example
    -------
    ::

        from src.sfm import run_sfm_pipeline
        from src.depth import run_depth_pipeline

        reconstruction, captures = run_sfm_pipeline(...)

        dmap_paths = run_depth_pipeline(
            reconstruction = reconstruction,
            captures       = captures,
            output_dir     = "outputs/depth",
        )
        # → ["outputs/depth/000.dmap", "outputs/depth/001.dmap", ...]

        # Stage 9:
        # dsm_path = fuse_depth_maps(dmap_paths, "outputs/dsm")
    """
    logger.info(
        "=== Stage 8: Depth Maps ===  "
        "%d images in reconstruction, %d captures",
        len(reconstruction.images),
        len(captures),
    )

    # ------------------------------------------------------------------
    # Step 1: Compute per-image depth ranges from SfM sparse points
    # ------------------------------------------------------------------
    logger.info("Step 1/4: Computing per-image depth ranges …")
    depth_ranges = compute_depth_ranges(reconstruction, captures)

    if print_stats:
        print_depth_range_stats(depth_ranges)

    # ------------------------------------------------------------------
    # Step 2: Prepare OpenMVS workspace (symlinks + .mvs scene file)
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Preparing OpenMVS workspace …")
    mvs_scene_path = prepare_openmvs_workspace(
        reconstruction=reconstruction,
        captures=captures,
        output_dir=output_dir,
        colmap_sparse_dir=colmap_sparse_dir,
        openmvs_bin_dir=openmvs_bin_dir,
    )

    # ------------------------------------------------------------------
    # Step 3: Run DensifyPointCloud in depth-estimation mode
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Running OpenMVS depth estimation …")
    dmap_paths = run_openmvs_depth_estimation(
        mvs_scene_path=mvs_scene_path,
        output_dir=output_dir,
        depth_ranges=depth_ranges,
        resolution_level=resolution_level,
        num_neighbors=num_neighbors,
        use_gpu=use_gpu,
        openmvs_bin_dir=openmvs_bin_dir,
    )

    # ------------------------------------------------------------------
    # Step 4: Validate outputs
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Validating .dmap outputs …")
    dmap_paths = _validate_dmap_outputs(dmap_paths, output_dir)

    logger.info(
        "=== Stage 8 complete: %d .dmap files ready for Stage 9 ===",
        len(dmap_paths),
    )

    return dmap_paths


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_dmap_outputs(dmap_paths: List[str], output_dir: str) -> List[str]:
    """
    Validate that .dmap files exist, are non-empty, and re-scan the
    output directory in case OpenMVS wrote them to a different path than
    expected.

    Parameters
    ----------
    dmap_paths : List[str]
        Paths returned by run_openmvs_depth_estimation().
    output_dir : str
        Root output directory (scanned as fallback).

    Returns
    -------
    List[str]
        Validated, sorted list of .dmap paths.

    Raises
    ------
    RuntimeError
        If no valid .dmap files are found at all.
    """
    valid: List[str] = []
    missing: List[str] = []
    empty: List[str] = []

    for p in dmap_paths:
        path = Path(p)
        if not path.exists():
            missing.append(p)
        elif path.stat().st_size == 0:
            empty.append(p)
        else:
            valid.append(p)

    if missing:
        logger.warning(
            "%d .dmap files reported by OpenMVS but not found on disk: %s",
            len(missing),
            missing[:5],
        )
    if empty:
        logger.warning(
            "%d .dmap files are empty (likely OpenMVS skipped them): %s",
            len(empty),
            empty[:5],
        )

    # If runner returned nothing (e.g. older OpenMVS that writes without
    # reporting paths), re-scan the output directory
    if not valid:
        logger.info(
            "No .dmap paths from runner — scanning %s for .dmap files …",
            output_dir,
        )
        valid = sorted(str(p) for p in Path(output_dir).glob("*.dmap") if p.stat().st_size > 0)

    if not valid:
        raise RuntimeError(
            f"Stage 8 produced no valid .dmap files in {output_dir}.\n"
            "Check DensifyPointCloud logs above for errors.\n"
            "Common causes:\n"
            "  - OpenMVS not installed or wrong binary path\n"
            "  - scene.mvs not found or malformed\n"
            "  - GPU OOM (try resolution_level=2 for a test run)\n"
            "  - All images failed depth estimation (check image quality)"
        )

    logger.info("Validated %d .dmap files.", len(valid))
    return sorted(valid)
