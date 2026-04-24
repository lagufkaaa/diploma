import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from solvers.hybrid_crop import resolve_adaptive_vertical_crop


def test_adaptive_vertical_crop_falls_back_when_lowest_is_near_zero():
    selection = resolve_adaptive_vertical_crop(
        height=45000.0,
        base_crop_height=15000.0,
        lowest_unpacked_y=0.0,
        crop_zero_tolerance=1e-6,
        crop_lowest_multiplier=1.5,
    )

    assert selection.used_crop_height == pytest.approx(15000.0)
    assert selection.crop_decision == "adaptive_lowest_near_zero_fallback_crop_height"
    assert selection.lowest_distance_to_top is None
    assert selection.scaled_lowest_distance is None


def test_adaptive_vertical_crop_falls_back_when_scaled_window_reaches_zero_line():
    selection = resolve_adaptive_vertical_crop(
        height=45000.0,
        base_crop_height=15000.0,
        lowest_unpacked_y=577.5017613138743,
        crop_zero_tolerance=1e-6,
        crop_lowest_multiplier=1.5,
    )

    assert selection.used_crop_height == pytest.approx(15000.0)
    assert selection.crop_decision == "adaptive_scaled_near_zero_fallback_crop_height"
    assert selection.lowest_distance_to_top == pytest.approx(44422.498238686126)
    assert selection.scaled_lowest_distance == pytest.approx(66633.74735802919)


def test_adaptive_vertical_crop_keeps_scaled_window_when_it_stays_above_zero_line():
    selection = resolve_adaptive_vertical_crop(
        height=45000.0,
        base_crop_height=15000.0,
        lowest_unpacked_y=30000.0,
        crop_zero_tolerance=1e-6,
        crop_lowest_multiplier=1.2,
    )

    assert selection.used_crop_height == pytest.approx(18000.0)
    assert selection.crop_decision == "adaptive_scaled_distance_vs_crop_height"
    assert selection.lowest_distance_to_top == pytest.approx(15000.0)
    assert selection.scaled_lowest_distance == pytest.approx(18000.0)
