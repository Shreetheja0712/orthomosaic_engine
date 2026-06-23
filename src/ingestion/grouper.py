import os
import re
from typing import Optional
from .capture import Capture
from .exif_reader import read_gps


# Matches: IMG_<optional_prefix>_<capture>_<band>.<ext>
# Supports formats like:
# - IMG_0001_000_RGB.jpg (test format)
# - IMG_260315_083045_0000_RGB.JPG (user format with date and time)
FILENAME_PATTERN = re.compile(
    r"^IMG_(?:\d+_)*(\d+)_(RGB|GRE|NIR|RED|REG)\.(jpg|jpeg|tiff|tif)$",
    re.IGNORECASE
)

BAND_MAP = {
    "RGB": "rgb",
    "GRE": "green",
    "NIR": "nir",
    "RED": "red",
    "REG": "reg",
}


def _parse_filename(filename: str) -> Optional[tuple[str, str]]:
    """
    Parse filename and return (capture_id, band) or None if no match.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    capture_id = match.group(1)
    band = match.group(2).upper()
    return capture_id, band


def group_captures(mission_dir: str) -> dict[str, Capture]:
    """
    Scan rgb/ and multi/ folders inside mission_dir.
    Group files by capture ID.
    Returns dict of capture_id -> Capture.
    """
    rgb_dir   = os.path.join(mission_dir, "rgb")
    multi_dir = os.path.join(mission_dir, "multi")

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"rgb/ folder not found in {mission_dir}")
    if not os.path.isdir(multi_dir):
        raise FileNotFoundError(f"multi/ folder not found in {mission_dir}")

    captures: dict[str, Capture] = {}

    # Scan rgb/
    for filename in sorted(os.listdir(rgb_dir)):
        result = _parse_filename(filename)
        if result is None:
            continue
        capture_id, band = result
        if band != "RGB":
            print(f"[grouper] Warning: unexpected band {band} in rgb/ folder: {filename}")
            continue

        filepath = os.path.join(rgb_dir, filename)

        if capture_id not in captures:
            captures[capture_id] = Capture(capture_id=capture_id)

        captures[capture_id].rgb = filepath

        # Read GPS from RGB image
        lat, lon, alt = read_gps(filepath)
        captures[capture_id].latitude  = lat
        captures[capture_id].longitude = lon
        captures[capture_id].altitude  = alt

    # Scan multi/
    for filename in sorted(os.listdir(multi_dir)):
        result = _parse_filename(filename)
        if result is None:
            continue
        capture_id, band = result
        if band == "RGB":
            print(f"[grouper] Warning: RGB file found in multi/ folder: {filename}")
            continue

        filepath = os.path.join(multi_dir, filename)

        if capture_id not in captures:
            captures[capture_id] = Capture(capture_id=capture_id)
            # Warn: this capture has no RGB image yet. GPS is only read from
            # RGB images, so latitude/longitude will remain None for this
            # capture, which will cause it to be rejected at the quality
            # filter or keyframe selection stage.
            print(
                f"[grouper] Warning: capture '{capture_id}' found in multi/ "
                f"but not in rgb/. GPS will be None (no RGB to read EXIF from). "
                f"Ensure RGB file follows the naming convention IMG_<frame>_{capture_id}_RGB.<ext>."
            )

        field = BAND_MAP[band]
        setattr(captures[capture_id], field, filepath)

    return captures