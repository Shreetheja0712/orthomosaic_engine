"""
run_pipeline.py  —  Full Agri Orthomosaic Engine pipeline
Usage: python run_pipeline.py --mission /path/to/your_mission --output /path/to/outputs
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    parser.add_argument("--n-neighbors", type=int, default=8,
                        help="GPS neighbors per image (default 8, use 12 for >80% overlap)")
    parser.add_argument(
        "--start-from-stage", type=int, default=1,
        choices=range(1, 13), metavar="N",
        help=(
            "Resume pipeline from stage N (1-12). "
            "Stages 1-2 (ingestion) always run — they are fast and required. "
            "Stages that were skipped must have already written their outputs to --output. "
            "Example: --start-from-stage 5  resumes from geometric verification."
        ),
    )
    args = parser.parse_args()

    start = args.start_from_stage
    mission_dir = args.mission
    output_dir  = Path(args.output)
    use_gpu     = not args.no_gpu

    if start > 1:
        print(f"\n[pipeline] Resuming from stage {start}. "
              f"Stages 1-{start - 1} will be skipped (outputs must exist in {output_dir}).")

    total_start = time.time()
    
    # Helper to print elapsed time
    def print_stage_time(stage_name, start_time):
        elapsed = time.time() - start_time
        print(f"  [Time] {stage_name} took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        return time.time()

    # ── Stages 1+2: Ingest + quality filter (always runs — fast, needed everywhere) ──
    print("\n=== Stage 1+2: Ingestion & Quality Filter ===")
    stage_start = time.time()
    captures = load_mission(mission_dir)

    # Detect if GPS is completely missing from EXIF
    sfm_keyframe_interval = 3
    has_any_gps = any(c.latitude is not None and c.longitude is not None for c in captures)
    if not has_any_gps:
        print("\n[pipeline] WARNING: No GPS metadata found in any image EXIF.")
        print("[pipeline] Automatically generating sequential mock GPS coordinates to allow relative reconstruction.")
        print("[pipeline] Matching will fall back to EXHAUSTIVE to ensure correct stitching without GPS.")
        for i, cap in enumerate(captures):
            cap.latitude = 45.0 + i * 0.000045  # spacing of approx 5 meters
            cap.longitude = 9.0
            cap.altitude = 120.0
        args.n_neighbors = len(captures)
        sfm_keyframe_interval = 1

    captures = filter_quality(captures)
    print(f"  {len(captures)} valid captures ready")
    stage_start = print_stage_time("Stage 1+2", stage_start)

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    if start <= 3:
        print("\n=== Stage 3: Feature Extraction (ALIKED) ===")
        extract_features(captures, output_dir=str(output_dir), use_gpu=use_gpu)
        stage_start = print_stage_time("Stage 3", stage_start)
    else:
        print(f"\n=== Stage 3: Feature Extraction — SKIPPED (start-from-stage={start}) ===")

    # ── Stage 4: Feature matching ─────────────────────────────────────────────
    if start <= 4:
        print("\n=== Stage 4: Feature Matching (LightGlue) ===")
        match_features(captures, output_dir=str(output_dir), n_neighbors=args.n_neighbors, use_gpu=use_gpu)
        stage_start = print_stage_time("Stage 4", stage_start)
    else:
        print(f"\n=== Stage 4: Feature Matching — SKIPPED (start-from-stage={start}) ===")

    # ── Stage 5: DB import + Geometric verification ───────────────────────────
    if start <= 5:
        print("\n=== Stage 5: Geometric Verification (PoseLib) ===")
        db_path = import_to_colmap(captures, output_dir=str(output_dir))
        stage_start = print_stage_time("Stage 5", stage_start)
    else:
        print(f"\n=== Stage 5: DB Import + Geometric Verification — SKIPPED (start-from-stage={start}) ===")
        db_path = output_dir / "database.db"
        if not db_path.exists():
            print(f"ERROR: database.db not found at {db_path}. Run from stage 5 or earlier first.")
            return
        print(f"  Using existing database: {db_path}")

    # ── Stage 6+7: SfM + Georeferencing ──────────────────────────────────────
    if start <= 7:
        print("\n=== Stage 6+7: SfM Mapping + Georeferencing (COLMAP) ===")
        reconstruction = run_sfm(
            database_path = str(db_path),
            image_dir     = str(Path(mission_dir) / "rgb"),
            output_dir    = str(output_dir / "sparse"),
            captures      = captures,
            has_rtk       = args.rtk,
            keyframe_interval = sfm_keyframe_interval,
            use_prior_position = has_any_gps,
        )
        if reconstruction is None:
            print("ERROR: SfM failed. Check your images have GPS and enough overlap.")
            return
        stage_start = print_stage_time("Stage 6+7", stage_start)
    else:
        print(f"\n=== Stage 6+7: SfM — SKIPPED (start-from-stage={start}) ===")
        reconstruction = _load_reconstruction(output_dir / "sparse" / "colmap")
        if reconstruction is None:
            print(f"ERROR: Could not load reconstruction from {output_dir / 'sparse' / 'colmap'}. Run from stage 6 or earlier first.")
            return
        print(f"  Loaded reconstruction: {reconstruction.num_reg_images} registered images")

    # ── Stage 8: Depth maps ───────────────────────────────────────────────────
    depth_dir = output_dir / "depth"
    if start <= 8:
        print("\n=== Stage 8: Depth Maps (OpenMVS) ===")
        dmap_paths, mvs_scene = run_depth_pipeline(
            reconstruction = reconstruction,
            captures       = captures,
            output_dir     = str(depth_dir),
            use_gpu        = use_gpu,
        )
        stage_start = print_stage_time("Stage 8", stage_start)
    else:
        print(f"\n=== Stage 8: Depth Maps — SKIPPED (start-from-stage={start}) ===")
        dmap_paths = sorted(str(p) for p in depth_dir.glob("*.dmap") if p.stat().st_size > 0)
        mvs_scene  = str(depth_dir / "scene.mvs")
        if not dmap_paths:
            print(f"ERROR: No .dmap files found in {depth_dir}. Run from stage 8 or earlier first.")
            return
        if not Path(mvs_scene).exists():
            print(f"ERROR: scene.mvs not found at {mvs_scene}. Run from stage 8 or earlier first.")
            return
        print(f"  Found {len(dmap_paths)} existing .dmap files + scene.mvs")

    # ── Stage 9: DSM ──────────────────────────────────────────────────────────
    dsm_dir = output_dir / "dsm"
    if start <= 9:
        print("\n=== Stage 9: DSM Generation (OpenMVS fusion) ===")
        dsm_path = run_dsm_pipeline(
            dmap_paths     = dmap_paths,
            mvs_scene_path = mvs_scene,
            reconstruction = reconstruction,
            output_dir     = str(dsm_dir),
            target_gsd_m   = args.gsd,
        )
        print(f"  DSM written to: {dsm_path}")
        stage_start = print_stage_time("Stage 9", stage_start)
    else:
        print(f"\n=== Stage 9: DSM Generation — SKIPPED (start-from-stage={start}) ===")
        dsm_path = str(dsm_dir / "dsm.tif")
        if not Path(dsm_path).exists():
            print(f"ERROR: dsm.tif not found at {dsm_path}. Run from stage 9 or earlier first.")
            return
        print(f"  Using existing DSM: {dsm_path}")

    # ── Stage 10: Orthorectification ─────────────────────────────────────────
    ortho_dir = output_dir / "ortho"
    if start <= 10:
        print("\n=== Stage 10: Orthorectification (CuPy) ===")
        ortho_result = run_ortho_pipeline(
            reconstruction        = reconstruction,
            captures              = captures,
            dsm_path              = dsm_path,
            output_dir            = str(ortho_dir),
            target_gsd_m          = args.gsd,
            process_multispectral = True,
        )
        print(f"  {len(ortho_result.rgb_tile_paths)} RGB tiles written")
        stage_start = print_stage_time("Stage 10", stage_start)
    else:
        print(f"\n=== Stage 10: Orthorectification — SKIPPED (start-from-stage={start}) ===")
        ortho_result = _load_ortho_result(ortho_dir)
        if not ortho_result.rgb_tile_paths:
            print(f"ERROR: No ortho RGB tiles found under {ortho_dir / 'rgb'}. Run from stage 10 or earlier first.")
            return
        print(f"  Found {len(ortho_result.rgb_tile_paths)} existing RGB tiles")

    # ── Stage 11: RGB Mosaicking ──────────────────────────────────────────────
    seamlines_dir = str(output_dir / "seamlines")
    if start <= 11:
        print("\n=== Stage 11: RGB Mosaicking (OpenCV) ===")
        rgb_mosaic_path, seamlines = run_rgb_mosaic(
            tile_paths        = ortho_result.rgb_tile_paths,
            output_path       = str(output_dir / "rgb_orthomosaic.tif"),
            seamlines_save_dir= seamlines_dir,
            target_gsd_m      = args.gsd,
        )
        print(f"  RGB mosaic: {rgb_mosaic_path}")
        stage_start = print_stage_time("Stage 11", stage_start)
    else:
        print(f"\n=== Stage 11: RGB Mosaicking — SKIPPED (start-from-stage={start}) ===")
        seamlines = _load_seamlines(seamlines_dir)
        if seamlines is None:
            print(f"ERROR: Seamlines not found in {seamlines_dir}. Run from stage 11 or earlier first.")
            return
        rgb_mosaic_path = str(output_dir / "rgb_orthomosaic.tif")
        print(f"  Loaded existing seamlines, RGB mosaic: {rgb_mosaic_path}")

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


# ── Resume helpers ────────────────────────────────────────────────────────────

def _load_reconstruction(sparse_path: Path):
    """Load a pycolmap Reconstruction from a binary sparse model on disk."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction()
        recon.read(str(sparse_path))
        return recon
    except Exception as e:
        print(f"[pipeline] Could not load reconstruction from {sparse_path}: {e}")
        return None


def _load_ortho_result(ortho_dir: Path):
    """Reconstruct an OrthoResult by globbing the tile directories."""
    from src.ortho import OrthoResult
    result = OrthoResult()
    rgb_dir = ortho_dir / "rgb"
    if rgb_dir.exists():
        result.rgb_tile_paths = sorted(str(p) for p in rgb_dir.glob("*.tif"))
    for band in ("GRE", "RED", "REG", "NIR"):
        band_dir = ortho_dir / "multi" / band
        if band_dir.exists():
            result.multi_tile_paths[band] = sorted(str(p) for p in band_dir.glob("*.tif"))
        else:
            result.multi_tile_paths[band] = []
    return result


def _load_seamlines(seamlines_dir: str):
    """Load a saved SeamlineSet from the seamlines directory."""
    try:
        from src.mosaic.seam_finder import SeamlineSet
        import numpy as np
        seamlines_path = Path(seamlines_dir) / "seamlines.npz"
        if not seamlines_path.exists():
            # Try any .npz in the directory
            candidates = list(Path(seamlines_dir).glob("*.npz"))
            if not candidates:
                return None
            seamlines_path = candidates[0]
        data = np.load(seamlines_path, allow_pickle=True)
        return SeamlineSet(masks=list(data["masks"]))
    except Exception as e:
        print(f"[pipeline] Could not load seamlines from {seamlines_dir}: {e}")
        return None


if __name__ == "__main__":
    main()
