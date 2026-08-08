"""Routing-plan helpers for the CPU MPS engine.

The optional ``_mps_native`` module contains the same planner in C++ so large
circuits can avoid Python dictionaries and list churn during route planning.
The Python implementation remains the portable fallback used from source
checkouts and on platforms where a compiler is unavailable.
"""
from __future__ import annotations

from typing import Sequence


def _candidate_route_swaps(
    first_site: int, second_site: int, move_first: bool
) -> list[int]:
    if first_site < second_site:
        if move_first:
            return list(range(first_site, second_site - 1))
        return list(range(second_site - 1, first_site, -1))
    if move_first:
        return list(range(first_site - 1, second_site, -1))
    return list(range(second_site, first_site - 1))


def _simulate_layout_swaps(order: list[int], swaps: Sequence[int]) -> list[int]:
    candidate = list(order)
    for site in swaps:
        candidate[site], candidate[site + 1] = (
            candidate[site + 1],
            candidate[site],
        )
    return candidate


def _lookahead_cost(
    order: list[int], lookahead: Sequence[tuple[int, int]]
) -> float:
    positions = {logical: site for site, logical in enumerate(order)}
    cost = 0.0
    for index, (first, second) in enumerate(lookahead):
        distance = abs(positions[int(first)] - positions[int(second)])
        cost += max(0, distance - 1) / float(index + 1)
    return cost


def _python_plan_routing(
    n_qubits: int,
    pairs: Sequence[tuple[int, int]],
    lookahead: int,
) -> tuple[list[list[int]], list[int], int]:
    order = list(range(int(n_qubits)))
    plan: list[list[int]] = []
    routed_swaps = 0
    normalized = [(int(first), int(second)) for first, second in pairs]
    for index, (first, second) in enumerate(normalized):
        positions = {logical: site for site, logical in enumerate(order)}
        first_site = positions[first]
        second_site = positions[second]
        first_swaps = _candidate_route_swaps(first_site, second_site, True)
        second_swaps = _candidate_route_swaps(first_site, second_site, False)
        first_order = _simulate_layout_swaps(order, first_swaps)
        second_order = _simulate_layout_swaps(order, second_swaps)
        future = normalized[index + 1:index + 1 + max(0, int(lookahead))]
        if _lookahead_cost(first_order, future) <= _lookahead_cost(
            second_order, future
        ):
            selected = first_swaps
            order = first_order
        else:
            selected = second_swaps
            order = second_order
        plan.append(selected)
        routed_swaps += len(selected)
    return plan, order, routed_swaps


try:  # pragma: no cover - exercised when the optional extension is built
    from ._mps_native import plan_routing as _native_plan_routing
except ImportError:  # pragma: no cover - the fallback is covered in tests
    _native_plan_routing = None


def plan_routing(
    n_qubits: int,
    pairs: Sequence[tuple[int, int]],
    lookahead: int,
) -> tuple[list[list[int]], list[int], int, str]:
    """Return route swaps, final layout, and swap count.

    The final tuple element identifies whether the native planner or portable
    Python fallback was used; it is included in diagnostics for benchmark
    reproducibility.
    """
    if _native_plan_routing is not None:
        plan, order, count = _native_plan_routing(
            int(n_qubits), list(pairs), int(lookahead)
        )
        return plan, order, int(count), "native"
    plan, order, count = _python_plan_routing(n_qubits, pairs, lookahead)
    return plan, order, count, "python"
