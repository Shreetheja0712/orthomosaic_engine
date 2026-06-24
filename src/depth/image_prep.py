"""
image_prep.py
=============
Prepare the OpenMVS workspace for depth map estimation.

OpenMVS EstimateDepthmaps requires:
  1. A flat directory of image files whose filenames match the names
     stored in the COLMAP database (e.g. "000.jpg", "001.jpg", …).
  2. A .mvs scene file that embeds camera poses, intrinsics, and image
     paths — produced by the InterfaceCOLMAP tool (the COLMAP-to-OpenMVS
     bridge binary that ships with OpenMVS).

What this module does
---------------------
1. Create ``<output_dir>/images/`` and symlink (or copy on Windows)
   each keyframe RGB image with its canonical name (``<capture_id>.jpg``).
2. Run ``InterfaceCOLMAP`` to convert the COLMAP sparse reconstruction
   and image directory into a .mvs scene file.
3. Return the path to the .mvs file for use by openmvs_runner.py.

Depth range injection
---------------------
Per-image depth ranges (from depth_range.py) are passed to OpenMVS via
the ``--min-depth`` / ``--max-depth`` flags **per image** when calling
DensifyPointCloud, not by patching the .mvs file directly.  The .mvs
binary format (MVArchive) is not designed for external patching, so we
pass them as CLI arguments in openmvs_runner.py instead.

Usage
-----
    from src.depth.image_prep import prepare_openmvs_workspace

    mvs_path = prepare_openmvs_workspace(
        reconstruction=recon,
        captures=captures,
        output_dir="/data/mission_001/depth",
    )
    # returns "/data/mission_001/depth/scene.mvs"
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_openmvs_workspace(
    reconstruction,
    captures: List,
    output_dir: str,
    colmap_sparse_dir: Optional[str] = None,
    openmvs_bin_dir: str = "",
) -> str:
    """
    Set up the OpenMVS workspace and produce the .mvs scene file.

    Parameters
    ----------
    reconstruction : pycolmap.Reconstruction
        The georeferenced sparse reconstruction from Stage 6/7.
    captures : List[Capture]
        Capture list from ingestion. Used to resolve the original RGB
        file paths.
    output_dir : str
        Root output directory for Stage 8 (e.g. "outputs/depth/").
        Will be created if it does not exist.
    colmap_sparse_dir : str, optional
        Path to the directory containing the COLMAP binary model files
        (cameras.bin, images.bin, points3D.bin).  If None, the
        reconstruction is exported to a temporary subdirectory first.
    openmvs_bin_dir : str
        Directory containing OpenMVS binaries.  Empty string = search
        PATH and common install locations.

    Returns
    -------
    str
        Absolute path to the produced ``scene.mvs`` file.

    Raises
    ------
    RuntimeError
        If InterfaceCOLMAP binary cannot be found or the conversion
        fails.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    image_dir = output_path / "images"
    image_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Symlink (or copy) keyframe images into the flat images/ directory
    # ------------------------------------------------------------------ #
    capture_map = {c.capture_id: c for c in captures}
    linked = 0

    for image_id, image in reconstruction.images.items():
        # image.name is the canonical name (e.g. "000.jpg") set during
        # ingestion.  Strip extension to get capture_id.
        stem = Path(image.name).stem  # e.g. "000"
        capture_id = stem

        if capture_id not in capture_map:
            logger.warning(
                "Image '%s' in reconstruction has no matching Capture "
                "(capture_id='%s'). Skipping.",
                image.name,
                capture_id,
            )
            continue

        src = Path(capture_map[capture_id].rgb).resolve()
        dst = image_dir / image.name

        if dst.exists():
            continue  # already linked / copied

        _symlink_or_copy(str(src), str(dst))
        linked += 1

    logger.info("Linked %d keyframe images to %s", linked, image_dir)

    # ------------------------------------------------------------------ #
    # 2. Export COLMAP reconstruction to binary model files (if needed)
    # ------------------------------------------------------------------ #
    if colmap_sparse_dir is None:
        sparse_dir = output_path / "sparse"
        if sparse_dir.exists():
            # A previous run (or an earlier interrupted attempt) may have left
            # stale cameras.bin/images.bin/rigs.bin/frames.bin here. mkdir(...,
            # exist_ok=True) below does NOT clear existing files, so without
            # this the OLD rigs.bin/frames.bin can survive even after this
            # run's export+cleanup, and InterfaceCOLMAP chokes on them again.
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        colmap_sparse_dir = str(sparse_dir)
        _export_reconstruction_to_colmap_safe_mvs(reconstruction, str(sparse_dir))
        logger.info("Exported COLMAP sparse model to %s", sparse_dir)

    # ------------------------------------------------------------------ #
    # 3. Run InterfaceCOLMAP to produce the .mvs scene file
    # ------------------------------------------------------------------ #
    mvs_scene_path = str(output_path / "scene.mvs")

    _run_interface_colmap(
        colmap_sparse_dir=colmap_sparse_dir,
        image_dir=str(image_dir),
        output_mvs=mvs_scene_path,
        bin_dir=openmvs_bin_dir,
    )

    logger.info("OpenMVS scene file written to %s", mvs_scene_path)
    return mvs_scene_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _symlink_or_copy(src: str, dst: str) -> None:
    """
    Create a symbolic link at *dst* pointing to *src*.

    Falls back to ``shutil.copy2`` on Windows (where symlinks require
    elevated privileges) or if symlink creation fails for any reason.

    Parameters
    ----------
    src : str
        Absolute source path (the original RGB image).
    dst : str
        Absolute destination path inside the workspace images/ directory.
    """
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        # Windows, or filesystem that doesn't support symlinks
        logger.debug(
            "Symlink failed for '%s' → '%s', falling back to copy.", src, dst
        )
        shutil.copy2(src, dst)


from .text_export import write_colmap_text

def _export_reconstruction_to_colmap_safe_mvs(reconstruction, sparse_dir: str) -> None:
    """
    Write COLMAP text model files from a pycolmap.Reconstruction.
    
    OpenMVS InterfaceCOLMAP crashes silently (exit 1) when parsing COLMAP 4.0+
    binary models due to structural changes like pose_prior. Deleting unsupported
    files like frames.bin is not enough because images.bin itself changed.
    
    We bypass this by writing legacy text format and removing frames/rigs files.
    """
    # 1. Ensure the directory is completely empty
    p = Path(sparse_dir)
    for f in p.iterdir():
        if f.is_file():
            f.unlink()
            
    # 2. Write text format
    if hasattr(reconstruction, "write_text"):
        logger.debug("Writing COLMAP sparse model using pycolmap.Reconstruction.write_text")
        reconstruction.write_text(sparse_dir)
    else:
        logger.debug("Writing COLMAP sparse model using custom Python write_colmap_text")
        write_colmap_text(reconstruction, sparse_dir)

    # 3. Delete unsupported files for OpenMVS compatibility
    for f in ["rigs.txt", "frames.txt", "rigs.bin", "frames.bin"]:
        file_path = p / f
        if file_path.exists():
            file_path.unlink()
            
    logger.debug(
        "Wrote COLMAP sparse model (OpenMVS-safe custom TEXT) to %s",
        sparse_dir
    )



def _run_interface_colmap(
    colmap_sparse_dir: str,
    image_dir: str,
    output_mvs: str,
    bin_dir: str = "",
) -> None:
    """
    Call the ``InterfaceCOLMAP`` OpenMVS binary to convert a COLMAP
    sparse model into a .mvs scene file.

    InterfaceCOLMAP is the official COLMAP → OpenMVS bridge and ships
    with every standard OpenMVS build.

    Parameters
    ----------
    colmap_sparse_dir : str
        Directory containing cameras.bin / images.bin / points3D.bin.
    image_dir : str
        Directory containing the flat image files.
    output_mvs : str
        Destination path for the output .mvs file.
    bin_dir : str
        Directory containing OpenMVS binaries (empty = search PATH).

    Raises
    ------
    RuntimeError
        If the binary is not found or exits with a non-zero code.
    """
    binary = _find_openmvs_binary("InterfaceCOLMAP", bin_dir)

    cmd = [
        binary,
        "--working-folder", str(Path(output_mvs).parent),
        "--input-file", colmap_sparse_dir,
        "--image-folder", image_dir,
        "--output-file", output_mvs,
    ]

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log_content = "<No InterfaceCOLMAP-*.log found>"
        log_files = list(Path(output_mvs).parent.glob("InterfaceCOLMAP-*.log"))
        if log_files:
            try:
                # OpenMVS logs can be verbose; grab the last 50 lines.
                with open(max(log_files, key=os.path.getctime), "r") as f:
                    lines = f.readlines()
                    log_content = "".join(lines[-50:])
            except Exception as e:
                log_content = f"<Failed to read log: {e}>"

        raise RuntimeError(
            f"InterfaceCOLMAP failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"--- InterfaceCOLMAP Log ---\n{log_content}"
        )

    logger.debug("InterfaceCOLMAP stdout:\n%s", result.stdout)


def _find_openmvs_binary(name: str, bin_dir: str = "") -> str:
    """
    Locate an OpenMVS binary by name.

    Search order (fixed so explicit bin_dir always wins):
      1. Explicit *bin_dir* if provided — checked FIRST.
      2. System PATH (``shutil.which``).
      3. Common installation locations.

    Parameters
    ----------
    name : str
        Binary name without extension (e.g. ``"InterfaceCOLMAP"``).
    bin_dir : str
        Explicit directory to check first.

    Returns
    -------
    str
        Absolute path to the binary.

    Raises
    ------
    RuntimeError
        If the binary cannot be found anywhere.
    """
    # 1. Explicit bin_dir takes priority (was previously checked AFTER PATH — fixed).
    if bin_dir:
        candidate = Path(bin_dir) / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            logger.debug("Found OpenMVS binary via bin_dir: %s", candidate)
            return str(candidate)
        # bin_dir given but binary absent — warn and fall through rather than
        # hard-failing so callers with a partially-populated bin_dir still work.
        logger.warning(
            "OpenMVS binary '%s' not found in explicit bin_dir '%s'; "
            "falling back to PATH and common locations.",
            name, bin_dir,
        )

    # 2. System PATH
    which_result = shutil.which(name)
    if which_result:
        return which_result

    # 3. Common install locations
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
            logger.debug("Found OpenMVS binary: %s", candidate)
            return str(candidate)

    raise RuntimeError(
        f"OpenMVS binary '{name}' not found.\n"
        f"Install OpenMVS: https://github.com/cdcseacave/openMVS\n"
        f"  conda install -c conda-forge openmvs\n"
        f"Or set openmvs_bin_dir to the directory containing '{name}'."
    )
