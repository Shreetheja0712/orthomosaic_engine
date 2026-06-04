from .extractor import extract_features, extract_features_cli_fallback
from .matcher import match_features, match_features_cli_fallback
from .neighbors import build_neighbor_pairs


def run_feature_pipeline(
    captures,
    database_path: str,
    use_gpu: bool = True,
    n_neighbors: int = 8,
    max_keypoints: int = 8192,
    use_cli_fallback: bool = False,
) -> str:
    """
    Full feature detection and matching pipeline.

    Steps:
    1. Extract GPU SIFT features from all RGB images (SINGLE camera mode)
    2. Build GPS-filtered neighbor pairs
    3. Match only neighbor pairs (not all pairs)
    4. RANSAC geometric verification

    Args:
        captures        : list of Capture objects from ingestion
        database_path   : where to store COLMAP database
        use_gpu         : enable CUDA GPU acceleration
        n_neighbors     : GPS neighbors to match per image (8 = default, 12 for high overlap)
        max_keypoints   : SIFT keypoints per image (8192 = good for aerial)
        use_cli_fallback: use subprocess CLI instead of pycolmap (Windows fallback)

    Returns:
        database_path
    """
    if use_cli_fallback:
        extract_features_cli_fallback(captures, database_path, use_gpu=use_gpu, max_keypoints=max_keypoints)
        match_features_cli_fallback(captures, database_path, n_neighbors=n_neighbors, use_gpu=use_gpu)
    else:
        extract_features(captures, database_path, use_gpu=use_gpu, max_keypoints=max_keypoints)
        match_features(captures, database_path, n_neighbors=n_neighbors, use_gpu=use_gpu)

    return database_path


__all__ = [
    "run_feature_pipeline",
    "extract_features",
    "extract_features_cli_fallback",
    "match_features",
    "match_features_cli_fallback",
    "build_neighbor_pairs",
]