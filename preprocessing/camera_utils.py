from __future__ import annotations

import math


VIEW_NAMES = ("top", "front", "back", "left", "right")


def camera_intrinsics(width: int, height: int, field_of_view_degrees: float) -> dict:
    focal = (width / 2.0) / math.tan(math.radians(field_of_view_degrees) / 2.0)
    return {
        "matrix": [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        "focal_length_pixels": focal,
        "principal_point": [width / 2.0, height / 2.0],
        "field_of_view_degrees": field_of_view_degrees,
    }