import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from solvers.free_space_improvement import resolve_free_space_improvement_requirement


def test_true_requires_any_improvement_not_one_percent():
    requirement = resolve_free_space_improvement_requirement(
        True,
        greedy_objective=50.0,
        container_area=100.0,
    )

    assert requirement.require_improvement is True
    assert requirement.mode == "any"
    assert requirement.required_improvement_percent is None
    assert requirement.min_improvement_area == pytest.approx(1e-6)
    assert requirement.min_total_objective == pytest.approx(50.000001)


def test_positive_number_requires_that_many_percentage_points():
    requirement = resolve_free_space_improvement_requirement(
        3,
        greedy_objective=50.0,
        container_area=100.0,
    )

    assert requirement.require_improvement is True
    assert requirement.mode == "percent"
    assert requirement.required_improvement_percent == pytest.approx(3.0)
    assert requirement.min_improvement_area == pytest.approx(3.0)
    assert requirement.min_total_objective == pytest.approx(53.0)
    assert requirement.target_free_space_percent == pytest.approx(47.0)


def test_other_values_do_not_require_improvement():
    for value in (False, None, 0, -1, "3"):
        requirement = resolve_free_space_improvement_requirement(
            value,
            greedy_objective=50.0,
            container_area=100.0,
        )

        assert requirement.require_improvement is False
        assert requirement.mode == "disabled"
        assert requirement.min_total_objective is None
        assert requirement.min_improvement_area == 0.0
