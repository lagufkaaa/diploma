from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AdaptiveVerticalCropSelection:
    used_crop_height: float
    crop_decision: str
    lowest_distance_to_top: Optional[float]
    scaled_lowest_distance: Optional[float]


def resolve_adaptive_vertical_crop(
    *,
    height: float,
    base_crop_height: float,
    lowest_unpacked_y: Optional[float],
    crop_zero_tolerance: float,
    crop_lowest_multiplier: float,
) -> AdaptiveVerticalCropSelection:
    height_effective = max(0.0, float(height))
    base_crop_height_effective = max(
        0.0,
        min(height_effective, float(base_crop_height)),
    )
    crop_zero_tolerance_effective = max(0.0, float(crop_zero_tolerance))
    crop_lowest_multiplier_effective = max(0.0, float(crop_lowest_multiplier))

    if lowest_unpacked_y is None:
        return AdaptiveVerticalCropSelection(
            used_crop_height=base_crop_height_effective,
            crop_decision="adaptive_no_candidates_fallback_crop_height",
            lowest_distance_to_top=None,
            scaled_lowest_distance=None,
        )

    lowest_unpacked_y_effective = float(lowest_unpacked_y)
    if lowest_unpacked_y_effective <= crop_zero_tolerance_effective + 1e-9:
        return AdaptiveVerticalCropSelection(
            used_crop_height=base_crop_height_effective,
            crop_decision="adaptive_lowest_near_zero_fallback_crop_height",
            lowest_distance_to_top=None,
            scaled_lowest_distance=None,
        )

    lowest_distance_to_top = max(
        0.0,
        height_effective - lowest_unpacked_y_effective,
    )
    scaled_lowest_distance = (
        lowest_distance_to_top * crop_lowest_multiplier_effective
    )
    scaled_cut_y = height_effective - scaled_lowest_distance

    if scaled_cut_y <= crop_zero_tolerance_effective + 1e-9:
        return AdaptiveVerticalCropSelection(
            used_crop_height=base_crop_height_effective,
            crop_decision="adaptive_scaled_near_zero_fallback_crop_height",
            lowest_distance_to_top=lowest_distance_to_top,
            scaled_lowest_distance=scaled_lowest_distance,
        )

    used_crop_height = max(
        base_crop_height_effective,
        float(scaled_lowest_distance),
    )
    used_crop_height = max(0.0, min(height_effective, float(used_crop_height)))
    return AdaptiveVerticalCropSelection(
        used_crop_height=used_crop_height,
        crop_decision="adaptive_scaled_distance_vs_crop_height",
        lowest_distance_to_top=lowest_distance_to_top,
        scaled_lowest_distance=scaled_lowest_distance,
    )
