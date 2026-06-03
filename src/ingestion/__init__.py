from .capture import Capture
from .grouper import group_captures
from .validator import validate_captures, ValidationError


def load_mission(mission_dir: str, strict: bool = True) -> list[Capture]:
    """
    Main entry point for ingestion.

    Scans mission_dir/rgb/ and mission_dir/multi/,
    groups files by capture ID, validates completeness.

    Returns sorted list of complete Capture objects.
    """
    captures = group_captures(mission_dir)
    return validate_captures(captures, strict=strict)


__all__ = [
    "Capture",
    "load_mission",
    "group_captures",
    "validate_captures",
    "ValidationError",
]