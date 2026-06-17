import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import gps_summary, load_rgb_captures
from src.features.extractor import extract_features, extract_features_cli_fallback
from src.features.matcher import match_features, match_features_cli_fallback


def _table_count(database_path: Path, table: str) -> int | None:
    if not database_path.exists():
        return None

    with sqlite3.connect(database_path) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not exists:
            return None
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _print_database_summary(database_path: Path) -> None:
    print("\n[benchmark] COLMAP database summary")
    for table in ("cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"):
        count = _table_count(database_path, table)
        if count is not None:
            print(f"  {table}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RGB feature extraction and matching.")
    parser.add_argument("--rgb-dir", required=True, help="Folder containing RGB images only.")
    parser.add_argument(
        "--output-dir",
        default="output/feature_benchmark",
        help="Folder where database.db, rgb_images, and match_pairs.txt are written.",
    )
    parser.add_argument("--n-neighbors", type=int, default=8, help="GPS neighbors per image for matching.")
    parser.add_argument("--max-keypoints", type=int, default=8192, help="SIFT keypoints per image.")
    parser.add_argument("--cpu", action="store_true", help="Disable COLMAP GPU feature extraction/matching.")
    parser.add_argument("--skip-matching", action="store_true", help="Run extraction only.")
    parser.add_argument("--check-gps-only", action="store_true", help="Only report GPS availability.")
    parser.add_argument(
        "--pycolmap",
        action="store_true",
        help="Use PyCOLMAP instead of the COLMAP CLI. On Python 3.14, CLI fallback is recommended.",
    )
    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP executable for CLI fallback.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    database_path = output_dir / "database.db"
    use_gpu = not args.cpu

    start_total = time.perf_counter()

    captures = load_rgb_captures(args.rgb_dir)
    with_gps, without_gps = gps_summary(captures)

    print(f"[benchmark] RGB images: {len(captures)}")
    print(f"[benchmark] With GPS: {with_gps} | Without GPS: {without_gps}")
    print(f"[benchmark] GPU Acceleration Enabled: {use_gpu}")

    if args.check_gps_only:
        for cap in captures:
            status = "GPS" if cap.latitude is not None and cap.longitude is not None else "NO GPS"
            print(f"  {cap.capture_id}: {status} ({cap.latitude}, {cap.longitude}, {cap.altitude})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = extract_features if args.pycolmap else extract_features_cli_fallback
    matcher = match_features if args.pycolmap else match_features_cli_fallback

    print(f"[benchmark] Output dir: {output_dir}")
    print(f"[benchmark] Database: {database_path}")

    start = time.perf_counter()
    if args.pycolmap:
        extractor(captures, str(database_path), use_gpu=use_gpu, max_keypoints=args.max_keypoints)
    else:
        extractor(
            captures,
            str(database_path),
            colmap_bin=args.colmap_bin,
            use_gpu=use_gpu,
            max_keypoints=args.max_keypoints,
        )
    extraction_seconds = time.perf_counter() - start
    print(f"[benchmark] Extraction time: {extraction_seconds:.2f}s")

    matching_seconds = None
    if not args.skip_matching:
        if without_gps:
            raise ValueError(
                "Matching needs GPS on every RGB image for the current GPS-filtered matcher. "
                "Use --skip-matching for extraction-only benchmarking."
            )

        start = time.perf_counter()
        if args.pycolmap:
            matcher(captures, str(database_path), n_neighbors=args.n_neighbors, use_gpu=use_gpu)
        else:
            matcher(
                captures,
                str(database_path),
                n_neighbors=args.n_neighbors,
                colmap_bin=args.colmap_bin,
                use_gpu=use_gpu,
            )
        matching_seconds = time.perf_counter() - start
        print(f"[benchmark] Matching time: {matching_seconds:.2f}s")

    total_seconds_components = extraction_seconds + (matching_seconds or 0.0)
    total_pipeline_seconds = time.perf_counter() - start_total
    
    print(f"[benchmark] Extraction + Matching time sum: {total_seconds_components:.2f}s")
    print(f"[benchmark] Total feature-stage pipeline time: {total_pipeline_seconds:.2f}s")
    _print_database_summary(database_path)

    print("\n[benchmark] Outputs")
    print(f"  Database: {database_path}")
    print(f"  RGB working images: {output_dir / 'rgb_images'}")
    print(f"  Match pairs: {output_dir / 'match_pairs.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
