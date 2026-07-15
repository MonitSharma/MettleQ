import argparse

import pytest

from tools.benchmark_mps_limits import build_circuit, parse_case, topology_pairs


def test_line_and_ring_topologies_are_distinct():
    line = topology_pairs("line_brickwork", 8, 0)
    ring = topology_pairs("ring_brickwork", 8, 0)
    assert line == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert ring[:-1] == line
    assert ring[-1] == (7, 0)


def test_grid_topology_requires_square_and_has_nonlocal_vertical_edges():
    with pytest.raises(ValueError, match="perfect-square"):
        topology_pairs("grid_2d", 15, 0)
    assert (0, 4) in topology_pairs("grid_2d", 16, 1)


def test_random_long_range_is_deterministic_and_disjoint():
    first = topology_pairs("random_long_range", 20, 3)
    second = topology_pairs("random_long_range", 20, 3)
    assert first == second
    assert len({wire for pair in first for wire in pair}) == 20


def test_limit_circuit_operation_count_reflects_topology_and_depth():
    circuit = build_circuit("all_to_all", 6, 2)
    two_qubit = [instruction for instruction in circuit.data if len(instruction.qubits) == 2]
    assert len(two_qubit) == 2 * (6 * 5 // 2)


def test_custom_limit_case_parser_validates_topology_shape_and_size():
    assert parse_case("rainbow:48:2") == ("rainbow", 48, 2)
    with pytest.raises(argparse.ArgumentTypeError, match="perfect-square"):
        parse_case("grid_2d:18:2")
