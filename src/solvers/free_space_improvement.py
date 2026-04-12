import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional


@dataclass(frozen=True)
class FreeSpaceImprovementRequirement:
    require_improvement: bool
    mode: str
    min_total_objective: Optional[float]
    min_improvement_area: float
    target_free_space_percent: Optional[float]
    required_improvement_percent: Optional[float]


def resolve_free_space_improvement_requirement(
    value: object,
    *,
    greedy_objective: float,
    container_area: float,
) -> FreeSpaceImprovementRequirement:
    """Resolve the user-facing free-space improvement setting.

    Semantics:
    - True: require any positive improvement over greedy.
    - positive number: require that many free-space percentage points of improvement.
    - anything else: do not add an improvement constraint.
    """

    greedy_objective = float(greedy_objective)
    container_area = float(container_area)

    if value is True:
        min_improvement_area = 0.0
        min_total_objective = None
        target_free_space_percent = _free_space_percent(container_area, greedy_objective)

        if container_area > 0.0:
            # "Any improvement" is enforced as a tiny positive area delta over greedy.
            # Use a scale-aware epsilon and cap it by the free area left in the container.
            eps_area = max(1e-6, container_area * 1e-12)
            slack_by_capacity = max(0.0, container_area - greedy_objective)
            min_improvement_area = min(eps_area, slack_by_capacity)
            if min_improvement_area > 1e-12:
                min_total_objective = greedy_objective + min_improvement_area
                target_free_space_percent = _free_space_percent(
                    container_area,
                    min_total_objective,
                )

        return FreeSpaceImprovementRequirement(
            require_improvement=True,
            mode="any",
            min_total_objective=min_total_objective,
            min_improvement_area=float(min_improvement_area),
            target_free_space_percent=target_free_space_percent,
            required_improvement_percent=None,
        )

    if isinstance(value, Real) and not isinstance(value, bool):
        required_improvement_percent = float(value)
        if (
            math.isfinite(required_improvement_percent)
            and required_improvement_percent > 0.0
            and container_area > 0.0
        ):
            min_improvement_area = container_area * required_improvement_percent / 100.0
            min_total_objective = greedy_objective + min_improvement_area
            return FreeSpaceImprovementRequirement(
                require_improvement=True,
                mode="percent",
                min_total_objective=float(min_total_objective),
                min_improvement_area=float(min_improvement_area),
                target_free_space_percent=_free_space_percent(
                    container_area,
                    min_total_objective,
                ),
                required_improvement_percent=float(required_improvement_percent),
            )

    return FreeSpaceImprovementRequirement(
        require_improvement=False,
        mode="disabled",
        min_total_objective=None,
        min_improvement_area=0.0,
        target_free_space_percent=None,
        required_improvement_percent=None,
    )


def _free_space_percent(container_area: float, packed_area: float) -> float:
    if container_area <= 0.0:
        return 0.0
    free = max(0.0, container_area - float(packed_area))
    return 100.0 * free / container_area
