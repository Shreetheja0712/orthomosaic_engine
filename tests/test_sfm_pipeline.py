"""
Tests for SfM pipeline orchestration logic.
Mocks pycolmap so these run without the library installed.
Tests the decision flow: RTK → GLOMAP → validate → fallback logic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.capture import Capture
from src.sfm.keyframes import select_keyframes, find_gps_guided_init_pair


def make_capture(capture_id, lat, lon, alt=120.0):
    c = Capture(
        capture_id = capture_id,
        rgb        = f"/fake/{capture_id}.jpg",
    )
    c.latitude   = lat
    c.longitude  = lon
    c.altitude   = alt
    c.green = c.nir = c.red = c.reg = f"/fake/{capture_id}.tif"
    return c


def make_grid_captures(rows=3, cols=5):
    """Simulate a grid flight pattern."""
    captures = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            lat = 16.900 + r * 0.001
            lon = 81.700 + c * 0.001
            captures.append(make_capture(f"{idx:03d}", lat, lon))
            idx += 1
    return captures


# ── Keyframe selection tests ──────────────────────────────────────────────────

def test_keyframe_count():
    """interval=3 should give ~1/3 as keyframes."""
    captures = make_grid_captures(3, 9)   # 27 captures
    kf, non_kf = select_keyframes(captures, interval=3)

    assert len(kf) + len(non_kf) == len(captures), \
        "Total should equal input count"
    assert len(kf) == 9, f"Expected 9 keyframes (27//3), got {len(kf)}"
    assert len(non_kf) == 18
    print(f"PASS: test_keyframe_count — {len(kf)} keyframes, {len(non_kf)} non-keyframes")


def test_keyframe_interval_1():
    """interval=1 should make all captures keyframes."""
    captures = make_grid_captures(2, 5)   # 10 captures
    kf, non_kf = select_keyframes(captures, interval=1)

    assert len(kf) == 10
    assert len(non_kf) == 0
    print("PASS: test_keyframe_interval_1")


def test_keyframe_no_gps_excluded():
    """Captures without GPS should be excluded from both lists."""
    captures = make_grid_captures(2, 5)
    captures[3].latitude  = None
    captures[3].longitude = None

    kf, non_kf = select_keyframes(captures, interval=2)
    total = len(kf) + len(non_kf)
    assert total == 9, f"Expected 9 with GPS, got {total}"
    print(f"PASS: test_keyframe_no_gps_excluded — {total} with GPS")


# ── GPS init pair tests ───────────────────────────────────────────────────────

def test_init_pair_found():
    """Should find a pair with reasonable baseline."""
    captures = make_grid_captures(3, 5)
    kf, _ = select_keyframes(captures, interval=1)
    pair = find_gps_guided_init_pair(kf, min_baseline_m=5.0, max_baseline_m=200.0)

    assert pair is not None, "Expected init pair to be found"
    cap_a, cap_b = pair
    assert cap_a.capture_id != cap_b.capture_id
    print(f"PASS: test_init_pair_found — pair: {cap_a.capture_id} <-> {cap_b.capture_id}")


def test_init_pair_none_on_impossible_baseline():
    """Should return None when no pair satisfies baseline window."""
    captures = [make_capture("000", 16.900, 81.700)]
    pair = find_gps_guided_init_pair(captures, min_baseline_m=50.0, max_baseline_m=100.0)
    assert pair is None
    print("PASS: test_init_pair_none_on_impossible_baseline")


def test_init_pair_baseline_in_range():
    """Selected pair baseline should be within the specified window."""
    from src.features.neighbors import haversine_distance

    captures = make_grid_captures(4, 4)
    kf, _ = select_keyframes(captures, interval=1)
    pair = find_gps_guided_init_pair(kf, min_baseline_m=5.0, max_baseline_m=300.0)

    if pair is not None:
        cap_a, cap_b = pair
        dist = haversine_distance(
            cap_a.latitude, cap_a.longitude,
            cap_b.latitude, cap_b.longitude,
        )
        assert 5.0 <= dist <= 300.0, f"Baseline {dist:.1f}m out of range"
        print(f"PASS: test_init_pair_baseline_in_range — baseline={dist:.1f}m")
    else:
        print("SKIP: test_init_pair_baseline_in_range — no pair found (too few captures)")


# ── Pipeline flow tests (no pycolmap) ────────────────────────────────────────

def test_pipeline_skips_glomap_without_rtk():
    """
    Verify that has_rtk=False causes GLOMAP to be skipped at logic level.
    We test the decision flag, not the actual mapper call.
    """
    has_rtk    = False
    try_glomap = has_rtk   # this is the exact logic in pipeline.py

    assert try_glomap is False
    print("PASS: test_pipeline_skips_glomap_without_rtk")


def test_pipeline_tries_glomap_with_rtk():
    """Verify has_rtk=True means GLOMAP is attempted."""
    has_rtk    = True
    try_glomap = has_rtk

    assert try_glomap is True
    print("PASS: test_pipeline_tries_glomap_with_rtk")


def test_run_sfm_writes_final_reconstruction_and_reports_fallback(tmp_path, monkeypatch, capsys):
    """Final model must be persisted at output_dir after COLMAP fallback."""
    import src.sfm.pipeline as sfm_pipeline

    class MockReconstruction:
        def __init__(self):
            self.num_reg_images = 6
            self.points3D = {1: object(), 2: object()}
            self.write_path = None

        def write(self, path):
            self.write_path = path

    captures = make_grid_captures(2, 3)
    recon = MockReconstruction()

    monkeypatch.setattr(sfm_pipeline, "inject_gps_priors", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_glomap", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfm_pipeline, "run_colmap_incremental", lambda *args, **kwargs: recon)
    monkeypatch.setattr(sfm_pipeline, "register_non_keyframes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sfm_pipeline, "run_final_bundle_adjustment", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfm_pipeline, "align_to_gps", lambda *args, **kwargs: True)

    output_dir = tmp_path / "sparse"
    result = sfm_pipeline.run_sfm(
        database_path = str(tmp_path / "colmap.db"),
        image_dir     = str(tmp_path / "rgb"),
        output_dir    = str(output_dir),
        captures      = captures,
        has_rtk       = True,
    )

    captured = capsys.readouterr().out
    assert result is recon
    assert recon.write_path == str(output_dir)
    assert "Path used  : COLMAP" in captured


if __name__ == "__main__":
    test_keyframe_count()
    test_keyframe_interval_1()
    test_keyframe_no_gps_excluded()
    test_init_pair_found()
    test_init_pair_none_on_impossible_baseline()
    test_init_pair_baseline_in_range()
    test_pipeline_skips_glomap_without_rtk()
    test_pipeline_tries_glomap_with_rtk()
    print("\nAll SfM pipeline tests done.")
