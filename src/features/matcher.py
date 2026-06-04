import subprocess
from pathlib import Path
from typing import List, Tuple

from ..ingestion.capture import Capture
from .neighbors import build_neighbor_pairs


def _check_pycolmap():
    try:
        import pycolmap
        return pycolmap
    except ImportError as exc:
        raise ImportError("pycolmap not installed. Run: pip install pycolmap") from exc


def _pycolmap_device(pycolmap, use_gpu: bool):
    if not hasattr(pycolmap, "Device"):
        return None
    return pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu


def _pycolmap_matching_options(pycolmap):
    if hasattr(pycolmap, "FeatureMatchingOptions"):
        return pycolmap.FeatureMatchingOptions()
    if hasattr(pycolmap, "SiftMatchingOptions"):
        return pycolmap.SiftMatchingOptions()
    return None


def _pycolmap_verification_options(pycolmap):
    if hasattr(pycolmap, "TwoViewGeometryOptions"):
        return pycolmap.TwoViewGeometryOptions()
    return None


def match_features(
    captures: List[Capture],
    database_path: str,
    n_neighbors: int = 8,
    use_gpu: bool = True,
) -> str:
    """
    Match features between GPS-filtered image pairs only.

    The generated pair file is passed directly to COLMAP/PyCOLMAP, so this stage
    avoids exhaustive matching and keeps the pair count near O(n * n_neighbors).
    """
    pycolmap = _check_pycolmap()

    if not hasattr(pycolmap, "match_image_pairs"):
        raise RuntimeError(
            "This pipeline requires pycolmap.match_image_pairs for GPS-filtered matching. "
            "Install pycolmap >= 4.0 or call run_feature_pipeline(..., use_cli_fallback=True)."
        )

    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run extract_features first.")

    pairs = build_neighbor_pairs(captures, n_neighbors=n_neighbors)
    pairs_file = db_path.parent / "match_pairs.txt"
    _write_pairs_file(captures, pairs, pairs_file)

    print(f"[matcher] Matching {len(pairs)} GPS-filtered pairs (GPU: {use_gpu})...")

    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = str(pairs_file)

    kwargs = {
        "database_path": str(db_path),
        "pairing_options": pairing_options,
    }

    matching_options = _pycolmap_matching_options(pycolmap)
    if matching_options is not None:
        if hasattr(matching_options, "use_gpu"):
            matching_options.use_gpu = use_gpu
        kwargs["matching_options"] = matching_options

    verification_options = _pycolmap_verification_options(pycolmap)
    if verification_options is not None:
        kwargs["verification_options"] = verification_options

    device = _pycolmap_device(pycolmap, use_gpu)
    if device is not None:
        kwargs["device"] = device

    pycolmap.match_image_pairs(**kwargs)

    print("[matcher] Matching and geometric verification complete.")
    return str(db_path)


def match_features_cli_fallback(
    captures: List[Capture],
    database_path: str,
    n_neighbors: int = 8,
    colmap_bin: str = "colmap",
    use_gpu: bool = True,
) -> str:
    """
    CLI fallback for Windows or environments without a compatible pycolmap.
    """
    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run extract_features first.")

    pairs = build_neighbor_pairs(captures, n_neighbors=n_neighbors)
    pairs_file = db_path.parent / "match_pairs.txt"
    _write_pairs_file(captures, pairs, pairs_file)

    print(f"[matcher_cli] Matching {len(pairs)} GPS-filtered pairs via COLMAP CLI...")

    cmd = [
        colmap_bin,
        "matches_importer",
        "--database_path",
        str(db_path),
        "--match_list_path",
        str(pairs_file),
        "--match_type",
        "pairs",
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP pair matching failed:\n{result.stderr}")

    print("[matcher_cli] Matching and geometric verification complete.")
    return str(db_path)


def _write_pairs_file(
    captures: List[Capture],
    pairs: List[Tuple[str, str]],
    output_path: Path,
) -> None:
    """
    Write pairs file in COLMAP format:
    Each line: image_a_filename image_b_filename
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    id_to_filename = {}
    for cap in captures:
        ext = Path(cap.rgb).suffix
        id_to_filename[cap.capture_id] = f"{cap.capture_id}{ext}"

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for id_a, id_b in pairs:
            if id_a in id_to_filename and id_b in id_to_filename:
                f.write(f"{id_to_filename[id_a]} {id_to_filename[id_b]}\n")
                written += 1

    print(f"[matcher] Wrote {written} pairs to {output_path}")
