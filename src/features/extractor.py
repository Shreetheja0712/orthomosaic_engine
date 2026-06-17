import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from ..ingestion.capture import Capture


def _check_pycolmap():
    try:
        import pycolmap
        return pycolmap
    except ImportError as exc:
        raise ImportError(
            "pycolmap not installed. Run: pip install pycolmap\n"
            "Or use run_feature_pipeline(..., use_cli_fallback=True) with COLMAP on PATH."
        ) from exc


def _prepare_rgb_image_dir(captures: List[Capture], database_path: str) -> tuple[Path, Path]:
    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    image_dir = db_path.parent / "rgb_images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    for cap in captures:
        if not cap.rgb:
            raise ValueError(f"Capture {cap.capture_id} has no RGB image.")

        ext = Path(cap.rgb).suffix
        dest = image_dir / f"{cap.capture_id}{ext}"
        try:
            os.symlink(os.path.abspath(cap.rgb), dest)
        except (OSError, NotImplementedError):
            shutil.copy2(cap.rgb, dest)

    return db_path, image_dir


def _pycolmap_device(pycolmap, use_gpu: bool):
    if not hasattr(pycolmap, "Device"):
        return None
    return pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu


def extract_features(
    captures: List[Capture],
    database_path: str,
    use_gpu: bool = True,
    max_keypoints: int = 8192,
    max_image_size: int = 3200,  # <-- Added safeguard for 4GB VRAM
    camera_model: str = "OPENCV",
) -> str:
    """
    Extract SIFT features from all RGB images using COLMAP/PyCOLMAP.

    The mission is imported in SINGLE camera mode so all RGB captures share one
    calibration, which is the expected setup for one fixed drone camera.
    """
    pycolmap = _check_pycolmap()
    db_path, image_dir = _prepare_rgb_image_dir(captures, database_path)

    print(f"[extractor] Extracting features from {len(captures)} RGB images...")
    print(f"[extractor] GPU: {use_gpu} | Camera model: {camera_model} | Max keypoints: {max_keypoints} | Max image size: {max_image_size}")
    print("[extractor] SINGLE camera mode ON - shared calibration across all captures")

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model

    if hasattr(pycolmap, "FeatureExtractionOptions"):
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = use_gpu
        # Prevent CUDA out-of-memory by limiting threads and max image size on GPU
        if use_gpu:
            extraction_options.num_threads = 1
            extraction_options.max_image_size = 3200 # downscale very large drone images for GPU

        extraction_options.sift.max_num_features = max_keypoints
        extraction_options.sift.max_image_size = max_image_size  # Limit resolution to prevent OOM
        if hasattr(pycolmap, "Normalization"):
            extraction_options.sift.normalization = pycolmap.Normalization.L1_ROOT

        kwargs = {
            "database_path": str(db_path),
            "image_path": str(image_dir),
            "camera_mode": pycolmap.CameraMode.SINGLE,
            "reader_options": reader_options,
            "extraction_options": extraction_options,
        }

        device = _pycolmap_device(pycolmap, use_gpu)
        if device is not None:
            kwargs["device"] = device

        pycolmap.extract_features(**kwargs)
    else:
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.use_gpu = use_gpu
        sift_options.max_num_features = max_keypoints
        if use_gpu:
            if hasattr(sift_options, "max_image_size"):
                sift_options.max_image_size = 3200
            if hasattr(sift_options, "num_threads"):
                sift_options.num_threads = 1
                
        if hasattr(pycolmap, "Normalization"):
            sift_options.normalization = pycolmap.Normalization.L1_ROOT

        reader_options.single_camera = True
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            image_options=reader_options,
            sift_options=sift_options,
        )

    print(f"[extractor] Feature extraction complete. Database: {db_path}")
    return str(db_path)


def extract_features_cli_fallback(
    captures: List[Capture],
    database_path: str,
    colmap_bin: str = "colmap",
    use_gpu: bool = True,
    max_keypoints: int = 8192,
) -> str:
    """
    CLI fallback for environments where PyCOLMAP is unavailable.
    """
    db_path, image_dir = _prepare_rgb_image_dir(captures, database_path)

    print(f"[extractor_cli] Extracting features from {len(captures)} images via COLMAP CLI...")

    cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
        "--SiftExtraction.max_num_features",
        str(max_keypoints),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP feature extraction failed:\n{result.stderr}")

    print(f"[extractor_cli] Done. Database: {db_path}")
    return str(db_path)
