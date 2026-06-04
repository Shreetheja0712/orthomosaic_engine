import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.matcher import match_features
from src.features.neighbors import build_neighbor_pairs
from src.features.rgb_only import load_rgb_captures
from src.ingestion.capture import Capture

WORK_DIR = Path(__file__).parent / "_work" / "features"


def make_capture(capture_id, lat, lon):
    c = Capture(capture_id=capture_id, rgb=f"/fake/{capture_id}.jpg")
    c.latitude = lat
    c.longitude = lon
    c.altitude = 120.0
    c.green = c.nir = c.red = c.reg = f"/fake/{capture_id}.tif"
    return c


def test_neighbor_count():
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
        make_capture("002", 16.902, 81.700),
        make_capture("003", 16.903, 81.700),
        make_capture("004", 16.904, 81.700),
        make_capture("005", 16.905, 81.700),
        make_capture("006", 16.906, 81.700),
        make_capture("007", 16.907, 81.700),
        make_capture("008", 16.908, 81.700),
        make_capture("009", 16.909, 81.700),
    ]

    pairs = build_neighbor_pairs(captures, n_neighbors=3)

    assert len(pairs) == len(set(pairs)), "Duplicate pairs found"
    for a, b in pairs:
        assert a < b, f"Pair not sorted: ({a}, {b})"


def test_pair_reduction():
    n = 50
    captures = [
        make_capture(str(i).zfill(3), 16.9 + i * 0.001, 81.7)
        for i in range(n)
    ]

    exhaustive = n * (n - 1) // 2
    pairs = build_neighbor_pairs(captures, n_neighbors=8)

    assert len(pairs) < exhaustive


def test_no_gps_warning():
    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
    ]
    captures[1].latitude = None
    captures[1].longitude = None

    pairs = build_neighbor_pairs(captures, n_neighbors=8)

    assert pairs == []


def test_distance_ordering():
    center = make_capture("000", 16.900, 81.700)
    near = make_capture("001", 16.901, 81.700)
    far = make_capture("002", 16.950, 81.700)

    pairs = build_neighbor_pairs([center, near, far], n_neighbors=1)

    assert ("000", "001") in pairs, f"Expected (000,001) in pairs, got {pairs}"


def test_match_features_uses_filtered_pairs(monkeypatch):
    calls = {}

    class FakeImportedPairingOptions:
        def __init__(self):
            self.match_list_path = None

    class FakeFeatureMatchingOptions:
        def __init__(self):
            self.use_gpu = True

    class FakeTwoViewGeometryOptions:
        pass

    fake_pycolmap = types.SimpleNamespace(
        Device=types.SimpleNamespace(auto="auto", cpu="cpu"),
        FeatureMatchingOptions=FakeFeatureMatchingOptions,
        ImportedPairingOptions=FakeImportedPairingOptions,
        TwoViewGeometryOptions=FakeTwoViewGeometryOptions,
    )

    def fake_match_image_pairs(**kwargs):
        calls["kwargs"] = kwargs

    fake_pycolmap.match_image_pairs = fake_match_image_pairs
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    db_path = WORK_DIR / "database.db"
    db_path.write_bytes(b"sqlite placeholder")

    captures = [
        make_capture("000", 16.900, 81.700),
        make_capture("001", 16.901, 81.700),
        make_capture("002", 16.950, 81.700),
    ]

    result = match_features(captures, str(db_path), n_neighbors=1, use_gpu=False)

    assert result == str(db_path)
    assert calls["kwargs"]["database_path"] == str(db_path)
    assert calls["kwargs"]["device"] == "cpu"
    assert calls["kwargs"]["pairing_options"].match_list_path == str(WORK_DIR / "match_pairs.txt")
    assert (WORK_DIR / "match_pairs.txt").read_text(encoding="utf-8").strip().splitlines() == [
        "000.jpg 001.jpg",
        "001.jpg 002.jpg",
    ]


def test_load_rgb_captures_skips_ingestion(monkeypatch):
    rgb_dir = WORK_DIR / "rgb_only"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    image_path = rgb_dir / "IMG_0001_123_RGB.jpg"
    image_path.write_bytes(b"\xff\xd8\xff")

    monkeypatch.setattr("src.features.rgb_only.read_gps", lambda path: (16.9, 81.7, 120.0))

    captures = load_rgb_captures(str(rgb_dir))

    assert len(captures) == 1
    assert captures[0].capture_id == "123"
    assert captures[0].rgb == str(image_path)
    assert captures[0].latitude == 16.9
