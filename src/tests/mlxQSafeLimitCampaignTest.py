from types import SimpleNamespace

import tools.benchmark_safe_dense_limits as dense_limits
import tools.benchmark_sdk_crossover_large as crossover


GIB = 2**30


def test_full_state_validation_preflight_refuses_30q_on_36gib(monkeypatch):
    monkeypatch.setattr(
        crossover,
        "_system_memory",
        lambda: {"total_bytes": 36 * GIB, "available_bytes": 24 * GIB},
    )
    allowed = crossover._validation_memory_preflight(
        28, maximum_fraction=0.50, minimum_headroom_bytes=6 * GIB
    )
    refused = crossover._validation_memory_preflight(
        30, maximum_fraction=0.50, minimum_headroom_bytes=6 * GIB
    )
    assert allowed["allowed"]
    assert not refused["allowed"]
    assert refused["projected_peak_bytes"] == 36 * GIB


def test_scalar_dense_preflight_admits_29q_and_refuses_30q(monkeypatch):
    monkeypatch.setattr(
        dense_limits,
        "_memory",
        lambda: {"total": 36 * GIB, "available": 24 * GIB},
    )
    args = SimpleNamespace(
        maximum_memory_fraction=0.45,
        minimum_headroom_gib=6.0,
    )
    for engine in dense_limits.ENGINES:
        assert dense_limits._preflight(engine, 29, args)["allowed"]
        assert not dense_limits._preflight(engine, 30, args)["allowed"]
