import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import load_mission, ValidationError


def make_dummy_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff")  # minimal fake image bytes


def test_complete_mission():
    with tempfile.TemporaryDirectory() as tmpdir:
        captures = ["000", "001", "002"]

        for cid in captures:
            make_dummy_file(os.path.join(tmpdir, "rgb",   f"IMG_0001_{cid}_RGB.jpg"))
            make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0002_{cid}_GRE.tiff"))
            make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0003_{cid}_NIR.tiff"))
            make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0004_{cid}_RED.tiff"))
            make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0005_{cid}_REG.tiff"))

        result = load_mission(tmpdir)
        assert len(result) == 3, f"Expected 3 captures, got {len(result)}"
        assert result[0].capture_id == "000"
        assert result[0].is_complete()
        print("PASS: test_complete_mission")


def test_incomplete_strict():
    with tempfile.TemporaryDirectory() as tmpdir:
        cid = "000"
        make_dummy_file(os.path.join(tmpdir, "rgb",   f"IMG_0001_{cid}_RGB.jpg"))
        make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0002_{cid}_GRE.tiff"))
        # NIR, RED, REG missing

        try:
            load_mission(tmpdir, strict=True)
            print("FAIL: should have raised ValidationError")
        except ValidationError as e:
            print(f"PASS: test_incomplete_strict — caught: {e}")


def test_incomplete_lenient():
    with tempfile.TemporaryDirectory() as tmpdir:
        # capture 000: complete
        make_dummy_file(os.path.join(tmpdir, "rgb",   "IMG_0001_000_RGB.jpg"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0002_000_GRE.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0003_000_NIR.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0004_000_RED.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0005_000_REG.tiff"))

        # capture 001: missing NIR
        make_dummy_file(os.path.join(tmpdir, "rgb",   "IMG_0001_001_RGB.jpg"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0002_001_GRE.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0004_001_RED.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", "IMG_0005_001_REG.tiff"))

        result = load_mission(tmpdir, strict=False)
        assert len(result) == 1, f"Expected 1 complete capture, got {len(result)}"
        assert result[0].capture_id == "000"
        print("PASS: test_incomplete_lenient")


def test_unknown_files_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        cid = "000"
        make_dummy_file(os.path.join(tmpdir, "rgb",   f"IMG_0001_{cid}_RGB.jpg"))
        make_dummy_file(os.path.join(tmpdir, "rgb",   "thumbnail.jpg"))   # should be ignored
        make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0002_{cid}_GRE.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0003_{cid}_NIR.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0004_{cid}_RED.tiff"))
        make_dummy_file(os.path.join(tmpdir, "multi", f"IMG_0005_{cid}_REG.tiff"))

        result = load_mission(tmpdir)
        assert len(result) == 1
        print("PASS: test_unknown_files_ignored")


if __name__ == "__main__":
    test_complete_mission()
    test_incomplete_strict()
    test_incomplete_lenient()
    test_unknown_files_ignored()
    print("\nAll tests done.")