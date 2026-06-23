from typing import Optional


def _convert_dms_to_decimal(dms_values, ref: str) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    d = float(dms_values[0].num) / float(dms_values[0].den)
    m = float(dms_values[1].num) / float(dms_values[1].den)
    s = float(dms_values[2].num) / float(dms_values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def read_gps(image_path: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Read GPS coordinates from image EXIF.
    Returns (latitude, longitude, altitude) or (None, None, None) if not found.
    """
    try:
        import exifread  # lazy import: only required at runtime, not at module load
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        lat = None
        lon = None
        alt = None

        if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
            lat = _convert_dms_to_decimal(
                tags["GPS GPSLatitude"].values,
                str(tags["GPS GPSLatitudeRef"])
            )

        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            lon = _convert_dms_to_decimal(
                tags["GPS GPSLongitude"].values,
                str(tags["GPS GPSLongitudeRef"])
            )

        if "GPS GPSAltitude" in tags:
            a = tags["GPS GPSAltitude"].values[0]
            alt = float(a.num) / float(a.den)

        return lat, lon, alt

    except Exception as e:
        print(f"[exif_reader] Warning: could not read GPS from {image_path}: {e}")
        return None, None, None