"""
openmvs_runner.py
=================
Run OpenMVS DensifyPointCloud in depth-estimation mode (``--dense-mode 0``).

This is the **only file in Stage 8 that changes** in a future swap to a
different depth estimator (e.g. ACMMP).  Everything else — depth_range.py,
dmap.py, image_prep.py, __init__.py — remains untouched.

What DensifyPointCloud does (depth-estimation mode)
----------------------------------------------------
For each RGB image it runs PatchMatch Stereo:
  - Compares small patches across ``num_neighbors`` neighbouring views.
  - GPU-accelerated (CUDA via SiftGPU).
  - Writes one .dmap file per image into ``output_dir/``.

Stage 9 then reads those .dmap files and runs OpenMVS in fusion mode
(``--dense-mode 1``) to produce the DSM.  That fusion call is entirely
in src/dsm/ — this file only handles depth estimation.

Depth range injection
---------------------
OpenMVS respects per-image ``depth-min`` / ``depth-max`` values stored in
the .mvs scene file.  We patch those values by writing a per-image
``depth-ranges.ini`` config that DensifyPointCloud reads via
``--depth-map-config``, if that flag is supported (OpenMVS ≥ 2.1).

For older OpenMVS builds that do not support per-image config we fall
back to global ``--min-depth`` / ``--max-depth`` derived from the
overall percentile range across all images — still far better than the
OpenMVS default (which does no range estimation at all).

Usage
-----
    from src.depth.openmvs_runner import run_openmvs_depth_estimation

    dmap_paths = run_openmvs_depth_estimation(
        mvs_scene_path  = "outputs/depth/scene.mvs",
        output_dir      = "outputs/depth",
        depth_ranges    = {"000.jpg": (94.2, 148.7), ...},
        resolution_level= 1,
        num_neighbors   = 5,
        use_gpu         = True,
    )
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution level guide (documented here so callers don't need to guess)
# ---------------------------------------------------------------------------
# resolution_level = 0 : full resolution — most accurate, slow, high VRAM
# resolution_level = 1 : half resolution — good accuracy, fast  ← default
# resolution_level = 2 : quarter resolution — fast preview only
#
# For flat agricultural fields at ~120 m AGL, half resolution (level=1)
# gives sufficient DSM accuracy for NDVI and yield prediction.
# Full resolution (level=0) would consume ~12-16 GB VRAM for 500 images
# which is near the 16 GB limit and risks OOM during the OS overhead peaks.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_openmvs_depth_estimation(
    mvs_scene_path: str,
    output_dir: str,
    depth_ranges: dict[str, tuple[float, float]],
    resolution_level: int = 1,
    num_neighbors: int = 5,
    use_gpu: bool = True,
    openmvs_bin_dir: str = "",
) -> List[str]:
    """
    Run OpenMVS DensifyPointCloud to produce per-image .dmap files.

    Parameters
    ----------
    mvs_scene_path : str
        Path to the .mvs scene file produced by image_prep.py.
    output_dir : str
        Directory where .dmap files will be written.
    depth_ranges : dict[str, tuple[float, float]]
        Per-image depth ranges from depth_range.py.
        Keys are image names (e.g. "000.jpg"), values are
        (depth_min_metres, depth_max_metres).
    resolution_level : int
        0 = full res, 1 = half res (default), 2 = quarter res.
        Half resolution is correct for 16 GB GPU and agriculture.
    num_neighbors : int
        Number of neighbouring images to compare per image.
        5 is sufficient for 80% forward overlap grid flights.
    use_gpu : bool
        If False, forces CPU mode (very slow — for debugging only).
    openmvs_bin_dir : str
        Directory containing OpenMVS binaries.  Empty = search PATH.

    Returns
    -------
    List[str]
        Sorted list of absolute paths to the .dmap files that were
        produced.  Empty list if estimation produced no output (signals
        a problem to the caller).

    Raises
    ------
    RuntimeError
        If DensifyPointCloud binary is not found.
        If the subprocess exits with a non-zero return code.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    binary = _find_openmvs_binary("DensifyPointCloud", openmvs_bin_dir)

    # Write per-image depth range config
    config_path = _write_depth_range_config(depth_ranges, output_path)

    # Compute global fallback range (used for older OpenMVS without per-image config)
    global_min, global_max = _global_depth_range(depth_ranges)

    cmd = _build_command(
        binary=binary,
        mvs_scene_path=mvs_scene_path,
        output_dir=str(output_path),
        resolution_level=resolution_level,
        num_neighbors=num_neighbors,
        use_gpu=use_gpu,
        global_min=global_min,
        global_max=global_max,
        config_path=config_path,
    )

    logger.info("Running OpenMVS depth estimation:")
    logger.info("  %s", " ".join(cmd))
    logger.info(
        "  images: %d  resolution_level: %d  num_neighbors: %d  gpu: %s",
        len(depth_ranges),
        resolution_level,
        num_neighbors,
        use_gpu,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.debug("DensifyPointCloud stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("DensifyPointCloud stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"DensifyPointCloud failed (exit {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    dmap_paths = _collect_dmap_files(output_path)
    logger.info(
        "Depth estimation complete: %d .dmap files written to %s",
        len(dmap_paths),
        output_path,
    )

    return dmap_paths


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_command(
    binary: str,
    mvs_scene_path: str,
    output_dir: str,
    resolution_level: int,
    num_neighbors: int,
    use_gpu: bool,
    global_min: float,
    global_max: float,
    config_path: str,
) -> List[str]:
    """
    Assemble the DensifyPointCloud subprocess command.

    ``--dense-mode 0`` = depth estimation only (no fusion).
    Fusion is done in Stage 9 via a separate DensifyPointCloud call
    with ``--dense-mode 1``.

    Parameters
    ----------
    binary : str
        Absolute path to DensifyPointCloud executable.
    mvs_scene_path : str
        .mvs scene file.
    output_dir : str
        Directory for .dmap output files.
    resolution_level : int
    num_neighbors : int
    use_gpu : bool
    global_min : float
        Global depth_min fallback (metres).
    global_max : float
        Global depth_max fallback (metres).
    config_path : str
        Path to per-image depth config file (may be empty string if
        OpenMVS doesn't support it — handled at call site).

    Returns
    -------
    List[str]
        Fully assembled command list ready for subprocess.run().
    """
    cmd = [
        binary,
        mvs_scene_path,
        "--dense-mode",      "0",          # estimate only, no fusion
        "--resolution-level", str(resolution_level),
        "--min-resolution",  "640",         # minimum image dimension after downscaling
        "--number-views",    str(num_neighbors),
        "--number-views-fuse", "3",         # min views needed for multi-view consistency
        "--output-dir",      output_dir,
        "--min-depth",       f"{global_min:.4f}",
        "--max-depth",       f"{global_max:.4f}",
    ]

    if not use_gpu:
        cmd += ["--cuda-device", "-1"]      # -1 = force CPU
        logger.warning("GPU disabled — depth estimation will be very slow.")

    # Per-image config (OpenMVS ≥ 2.1, may silently ignore on older versions)
    if config_path and Path(config_path).exists():
        cmd += ["--depth-map-config", config_path]

    return cmd


def _write_depth_range_config(
    depth_ranges: dict[str, tuple[float, float]],
    output_dir: Path,
) -> str:
    """
    Write a per-image depth range config file for OpenMVS.

    Format (INI-style, one image per line)::

        [images]
        000.jpg = 94.2000,148.7000
        001.jpg = 91.5000,151.3000
        ...

    This file is passed via ``--depth-map-config`` to DensifyPointCloud.
    OpenMVS ≥ 2.1 reads it and applies per-image depth bounds instead of
    the global ``--min-depth`` / ``--max-depth``.

    Parameters
    ----------
    depth_ranges : dict[str, tuple[float, float]]
    output_dir : Path

    Returns
    -------
    str
        Path to the written config file.
    """
    config_path = output_dir / "depth_ranges.ini"

    lines = ["[images]\n"]
    for name, (d_min, d_max) in sorted(depth_ranges.items()):
        lines.append(f"{name} = {d_min:.4f},{d_max:.4f}\n")

    config_path.write_text("".join(lines), encoding="utf-8")
    logger.debug("Wrote per-image depth range config to %s", config_path)
    return str(config_path)


def _global_depth_range(
    depth_ranges: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """
    Compute global (min, max) depth range across all images.

    Used as the ``--min-depth`` / ``--max-depth`` CLI arguments which
    serve as a fallback when per-image config is not supported.

    The global min is the minimum of all per-image depth_min values.
    The global max is the maximum of all per-image depth_max values.

    Parameters
    ----------
    depth_ranges : dict[str, tuple[float, float]]

    Returns
    -------
    tuple[float, float]
        (global_min, global_max) in metres.
    """
    if not depth_ranges:
        return (80.0, 200.0)  # safe agricultural default

    all_mins = [v[0] for v in depth_ranges.values()]
    all_maxs = [v[1] for v in depth_ranges.values()]
    return (min(all_mins), max(all_maxs))


def _collect_dmap_files(output_dir: Path) -> List[str]:
    """
    Return sorted list of .dmap file paths found in *output_dir*.

    Parameters
    ----------
    output_dir : Path

    Returns
    -------
    List[str]
        Sorted absolute paths.  Empty if none found.
    """
    dmap_files = sorted(output_dir.glob("*.dmap"))
    return [str(p) for p in dmap_files]


def _find_openmvs_binary(name: str, bin_dir: str = "") -> str:
    """
    Locate an OpenMVS binary by name.

    Search order:
      1. Explicit bin_dir.
      2. System PATH.
      3. Common installation locations.

    Parameters
    ----------
    name : str
        Binary name (e.g. "DensifyPointCloud").
    bin_dir : str
        Optional explicit directory.

    Returns
    -------
    str
        Absolute path to the binary.

    Raises
    ------
    RuntimeError
        If not found anywhere.
    """
    import os

    if bin_dir:
        candidate = Path(bin_dir) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    which_result = shutil.which(name)
    if which_result:
        return which_result

    common_dirs = [
        "/usr/local/bin/OpenMVS",
        "/usr/local/bin",
        "/opt/openmvs/bin",
        str(Path.home() / "OpenMVS" / "bin"),
        str(Path.home() / "openMVS" / "bin"),
    ]
    for d in common_dirs:
        candidate = Path(d) / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate)

    raise RuntimeError(
        f"OpenMVS binary '{name}' not found.\n"
        f"Install OpenMVS: https://github.com/cdcseacave/openMVS\n"
        f"  conda install -c conda-forge openmvs\n"
        f"Or pass openmvs_bin_dir to point at the directory containing '{name}'."
    )
