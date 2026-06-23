"""
run_pipeline.py  —  Full Agri Orthomosaic Engine pipeline
Usage: python run_pipeline.py --mission /path/to/your_mission --output /path/to/outputs
"""

import argparse
import time
from pathlib import Path

from src.ingestion import load_mission
from src.quality import filter_quality
from src.features import extract_features, match_features, import_to_colmap
from src.sfm import run_sfm
from src.depth import run_depth_pipeline
from src.dsm import run_dsm_pipeline
from src.ortho import run_ortho_pipeline
from src.mosaic import run_rgb_mosaic, run_ms_mosaic

def main():
    parser = argparse.ArgumentParser(description="Agri Orthomosaic Engine")
    parser.add_argument("--mission", required=True,
                        help="Path to mission folder (contains rgb/ and multi/)")
    parser.add_argument("--output",  required=True,
                        help="Output directory for all results")
    parser.add_argument("--gsd",     type=float, default=0.05,
                        help="Target ground sampling distance in metres/pixel (default 0.05 = 5cm)")
    parser.add_argument("--rtk",     action="store_true",
                        help="Set this flag if your drone has RTK GPS")
    parser.add_argument("--no-gpu",  action="store_true",
                        help="Disable GPU (run on CPU only — much slower)")
    args = parser.parse_args()

    mission_dir = args.mission
    output_dir  = Path(args.output)
    use_gpu     = not args.no_gpu

    total_start = time.time()
    
    # Helper to print elapsed time
    def print_stage_time(stage_name, start_time):
        elapsed = time.time() - start_time
        print(f"  [Time] {stage_name} took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        return time.time()

    # ── Stage 1+2: Ingest + quality filter ───────────────────────────────────
    print("\n=== Stage 1+2: Ingestion & Quality Filter ===")
    stage_start = time.time()
    captures = load_mission(mission_dir)
    captures = filter_quality(captures)
    print(f"  {len(captures)} valid captures ready")
    stage_start = print_stage_time("Stage 1+2", stage_start)

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    print("\n=== Stage 3: Feature Extraction (ALIKED) ===")
    extract_features(captures, output_dir=str(output_dir), use_gpu=use_gpu)
    stage_start = print_stage_time("Stage 3", stage_start)

    # ── Stage 4: Feature matching ─────────────────────────────────────────────
    print("\n=== Stage 4: Feature Matching (LightGlue) ===")
    match_features(captures, output_dir=str(output_dir), n_neighbors=8, use_gpu=use_gpu)
    stage_start = print_stage_time("Stage 4", stage_start)

    # ── Stage 5: Geometric verification ──────────────────────────────────────
    print("\n=== Stage 5: Geometric Verification (PoseLib) ===")
    # import_to_colmap runs PoseLib RANSAC internally by default
    db_path = import_to_colmap(captures, output_dir=str(output_dir))
    stage_start = print_stage_time("Stage 5", stage_start)

    # ── Stage 6+7: SfM + Georeferencing ──────────────────────────────────────
    print("\n=== Stage 6+7: SfM Mapping + Georeferencing (COLMAP) ===")
    reconstruction = run_sfm(
        database_path = str(db_path),
        image_dir     = str(Path(mission_dir) / "rgb"),
        output_dir    = str(output_dir / "sparse"),
        captures      = captures,
        has_rtk       = args.rtk,
    )
    if reconstruction is None:
        print("ERROR: SfM failed. Check your images have GPS and enough overlap.")
        return
    stage_start = print_stage_time("Stage 6+7", stage_start)

    # ── Stage 8: Depth maps ──────────────────────────────────────────
    print("\n=== Stage 8: Depth Maps (OpenMVS) ===")
    dmap_paths, mvs_scene = run_depth_pipeline(
        reconstruction = reconstruction,
        captures       = captures,
        output_dir     = str(output_dir / "depth"),
        use_gpu        = use_gpu,
    )
    stage_start = print_stage_time("Stage 8", stage_start)

    # ── Stage 9: DSM ──────────────────────────────────────────
    print("\n=== Stage 9: DSM Generation (OpenMVS fusion) ===")
    dsm_path = run_dsm_pipeline(
        dmap_paths     = dmap_paths,
        mvs_scene_path = mvs_scene,
        reconstruction = reconstruction,
        output_dir     = str(output_dir / "dsm"),
        target_gsd_m   = args.gsd,
    )
    print(f"  DSM written to: {dsm_path}")
    stage_start = print_stage_time("Stage 9", stage_start)

    # ── Stage 10: Orthorectification ─────────────────────────────────────────
    print("\n=== Stage 10: Orthorectification (CuPy) ===")
    ortho_result = run_ortho_pipeline(
        reconstruction        = reconstruction,
        captures              = captures,
        dsm_path              = dsm_path,
        output_dir            = str(output_dir / "ortho"),
        target_gsd_m          = args.gsd,
        process_multispectral = True,
    )
    print(f"  {len(ortho_result.rgb_tile_paths)} RGB tiles written")
    stage_start = print_stage_time("Stage 10", stage_start)

    # ── Stage 11: RGB Mosaicking ──────────────────────────────────────────────
    print("\n=== Stage 11: RGB Mosaicking (OpenCV) ===")
    rgb_mosaic_path, seamlines = run_rgb_mosaic(
        tile_paths        = ortho_result.rgb_tile_paths,
        output_path       = str(output_dir / "rgb_orthomosaic.tif"),
        seamlines_save_dir= str(output_dir / "seamlines"),
        target_gsd_m      = args.gsd,
    )
    print(f"  RGB mosaic: {rgb_mosaic_path}")
    stage_start = print_stage_time("Stage 11", stage_start)

    # ── Stage 12: Multispectral Mosaicking ───────────────────────────────────
    print("\n=== Stage 12: Multispectral Mosaicking (NumPy) ===")
    ms_mosaic_path = run_ms_mosaic(
        multi_tile_paths = ortho_result.multi_tile_paths,
        captures         = captures,
        seamline_set     = seamlines,
        output_path      = str(output_dir / "multispectral_orthomosaic.tif"),
        target_gsd_m     = args.gsd,
    )
    print(f"  MS mosaic: {ms_mosaic_path}")
    stage_start = print_stage_time("Stage 12", stage_start)

    # ── Done ──────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n=== Pipeline Complete ===")
    print(f"  [Time] TOTAL PIPELINE TIME: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes).")
    print(f"  RGB orthomosaic:            {output_dir}/rgb_orthomosaic.tif")
    print(f"  Multispectral orthomosaic:  {output_dir}/multispectral_orthomosaic.tif")
    print(f"  DSM:                        {output_dir}/dsm/dsm.tif")

if __name__ == "__main__":
    main()
