"""
src/dsm/fusion.py

Stage 9a — Depth Map Fusion.

Calls OpenMVS's DensifyPointCloud binary in *fusion* mode (--dense-mode 1)
on the .dmap files produced by Stage 8 (src/depth/openmvs_runner.py, which
uses --dense-mode 0, *estimation* mode). Same binary, different mode flag —
see dsm_stage_context.md.

Fusion cross-checks per-pixel depths across multiple views and keeps only
points consistent across `num_views_fuse` or more views, rejecting noise
and false matches. The result is an intermediate fused point cloud (.ply).
This file does not rasterize that point cloud — see rasterize.py for that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_NUM_VIEWS_FUSE = 3

# Common install locations to probe if the binary isn't on PATH or in an
# explicitly given bin_dir. Mirrors the helper Stage 8's openmvs_runner.py
# uses for the same binary family — worth promoting both to a shared
# src/dsm_common/openmvs_utils.py if/when Stage 8 and Stage 9 are touched
# in the same change.
_COMMON_OPENMVS_LOCATIONS = [
    "/usr/local/bin",
    "/usr/bin",
    "/opt/openMVS/bin",
    str(Path.home() / "openMVS_build" / "bin"),
]


def _find_openmvs_binary(name: str, bin_dir: str = "") -> str:
    """
    Locate an OpenMVS binary (e.g. "DensifyPointCloud").

    Search order: explicit bin_dir -> PATH -> common install locations.
    Raises FileNotFoundError if not found anywhere.
    """
    if bin_dir:
        candidate = Path(bin_dir) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise FileNotFoundError(
            f"OpenMVS binary '{name}' not found or not executable in "
            f"explicitly given bin_dir: {bin_dir}"
        )

    on_path = shutil.which(name)
    if on_path:
        return on_path

    for loc in _COMMON_OPENMVS_LOCATIONS:
        candidate = Path(loc) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise FileNotFoundError(
        f"Could not locate OpenMVS binary '{name}'. Checked PATH and "
        f"{_COMMON_OPENMVS_LOCATIONS}. Pass openmvs_bin_dir explicitly, "
        f"or install OpenMVS (conda install -c conda-forge openmvs, or "
        f"build from github.com/cdcseacave/openMVS)."
    )


def run_fusion(
    mvs_scene_path: str,
    output_dir: str,
    num_views_fuse: int = DEFAULT_NUM_VIEWS_FUSE,
    geometric_consistent: bool = True,
    openmvs_bin_dir: str = "",
) -> str:
    """
    Run OpenMVS DensifyPointCloud in fusion mode against an existing set of
    .dmap files (produced by Stage 8 and referenced inside mvs_scene_path).

    Parameters
    ----------
    mvs_scene_path : path to the .mvs scene file written by Stage 8's
        image_prep.py (already has per-image .dmap files associated).
    output_dir : directory OpenMVS should write its fused output into.
    num_views_fuse : minimum number of views a 3D point must be consistent
        in to be kept. 3 is the standard/safe default for ~80% overlap
        missions (every ground point visible in 8+ images).
    geometric_consistent : if True, pass --geometric-consistent 1 so OpenMVS
        cross-checks normals as well as depth for consistency.
    openmvs_bin_dir : optional explicit directory containing the OpenMVS
        binaries.

    Returns
    -------
    str : path to the fused .ply point cloud.
    """
    if not Path(mvs_scene_path).is_file():
        raise FileNotFoundError(f".mvs scene file not found: {mvs_scene_path}")

    binary = _find_openmvs_binary("DensifyPointCloud", openmvs_bin_dir)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary,
        str(mvs_scene_path),
        "--dense-mode", "1",
        "--number-views-fuse", str(num_views_fuse),
        "--geometric-consistent", "1" if geometric_consistent else "0",
        "--output-dir", str(out_dir),
    ]

    result = subprocess.run(
        cmd, cwd=str(out_dir), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenMVS DensifyPointCloud (fusion mode) failed "
            f"(exit code {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # OpenMVS names the fused cloud after the input scene by convention,
    # e.g. scene.mvs -> scene_dense.ply. Search the output dir for the
    # most plausible match rather than hardcoding the exact naming rule,
    # since it can vary slightly across OpenMVS versions/flags.
    scene_stem = Path(mvs_scene_path).stem
    candidates = sorted(out_dir.glob(f"{scene_stem}*dense*.ply"))
    if not candidates:
        candidates = sorted(out_dir.glob("*.ply"))
    if not candidates:
        raise RuntimeError(
            f"DensifyPointCloud reported success but no .ply file was "
            f"found in {out_dir}.\nstdout:\n{result.stdout}"
        )

    return str(candidates[0])
