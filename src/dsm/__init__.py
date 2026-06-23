"""
src/dsm/__init__.py

Stage 9 — DSM Generation. Single entry point: run_dsm_pipeline().

Call order:
  1. run_fusion()            .dmap files (via .mvs scene) -> fused.ply
  2. rasterize_pointcloud()  fused.ply -> dsm_raw.tif (has gaps)
  3. fill_dsm_gaps()         dsm_raw.tif -> dsm.tif (gaps filled)
  4. check_gap_coverage()    log final coverage %
  5. delete fused.ply unless keep_pointcloud=True
  6. return dsm.tif path

Stage 10 (Orthorectification) needs only the returned dsm.tif path — no
OpenMVS objects or .ply files cross this boundary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .fusion import run_fusion
from .interpolate import check_gap_coverage, fill_dsm_gaps
from .rasterize import DEFAULT_TARGET_GSD_M, rasterize_pointcloud

DEFAULT_NUM_VIEWS_FUSE = 3

__all__ = ["run_dsm_pipeline", "run_fusion", "rasterize_pointcloud", "fill_dsm_gaps"]


def run_dsm_pipeline(
    dmap_paths: List[str],
    mvs_scene_path: str,
    reconstruction,
    output_dir: str,
    target_gsd_m: float = DEFAULT_TARGET_GSD_M,
    num_views_fuse: int = DEFAULT_NUM_VIEWS_FUSE,
    keep_pointcloud: bool = False,
    openmvs_bin_dir: str = "",
    crs: str = "EPSG:4326",
) -> str:
    """
    Run the full Stage 9 DSM generation pipeline.

    Parameters
    ----------
    dmap_paths : .dmap file paths from Stage 8's run_depth_pipeline().
        Not passed directly to OpenMVS (it reads them via the .mvs scene),
        but validated here so a missing/incomplete depth stage fails loudly
        at the Stage 9 boundary rather than silently inside OpenMVS.
    mvs_scene_path : .mvs scene file from Stage 8's image_prep.py.
    reconstruction : pycolmap.Reconstruction from SfM, passed through to
        rasterize_pointcloud() for CRS/georef context.
    output_dir : working directory for fusion + final DSM output.
    target_gsd_m : output DSM resolution in meters/pixel. Must match the
        downstream orthomosaic target GSD.
    num_views_fuse : minimum consistent views for a fused 3D point.
    keep_pointcloud : if False (default), the intermediate fused .ply is
        deleted after dsm.tif is written.
    openmvs_bin_dir : optional explicit path to OpenMVS binaries.
    crs : output CRS for the DSM GeoTIFF (e.g. "EPSG:32644" for UTM zone 44N).
        Defaults to "EPSG:4326" which triggers auto-detection from the
        reconstruction's GPS priors via rasterize_pointcloud(). Pass the
        correct UTM EPSG explicitly when auto-detection is not possible.

    Returns
    -------
    str : path to the final dsm.tif.
    """
    if not dmap_paths:
        raise ValueError(
            "run_dsm_pipeline received an empty dmap_paths list — Stage 8 "
            "must produce at least one .dmap file before Stage 9 can run."
        )
    missing = [p for p in dmap_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} .dmap file(s) from Stage 8 are missing on disk, "
            f"e.g.: {missing[:3]}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fusion
    fused_ply = run_fusion(
        mvs_scene_path=mvs_scene_path,
        output_dir=str(out_dir / "fusion"),
        num_views_fuse=num_views_fuse,
        geometric_consistent=True,
        openmvs_bin_dir=openmvs_bin_dir,
    )

    # 2. Rasterize
    dsm_raw_path = str(out_dir / "dsm_raw.tif")
    rasterize_pointcloud(
        ply_path=fused_ply,
        output_path=dsm_raw_path,
        reconstruction=reconstruction,
        target_gsd_m=target_gsd_m,
        crs=crs,
    )

    # 3. Gap fill
    dsm_path = str(out_dir / "dsm.tif")
    fill_dsm_gaps(dsm_path=dsm_raw_path, output_path=dsm_path)

    # 4. Log final coverage (fill_dsm_gaps already logs internally too;
    #    explicit call here keeps this entry point self-documenting).
    check_gap_coverage(dsm_path=dsm_path)

    # 5. Clean up intermediate point cloud
    if not keep_pointcloud:
        try:
            os.remove(fused_ply)
        except OSError:
            pass

    # 6. Return final DSM
    return dsm_path
