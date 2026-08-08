from importlib import metadata


def test_canonical_distribution_and_import_name_are_mettleq():
    import mettleq

    assert metadata.version("mettleq") == "0.3.0rc1"
    assert mettleq.__version__ == "0.3.0rc1"


def test_legacy_mlxq_namespace_resolves_during_migration():
    import mettleq
    import mlxq
    from mettleq.integrations.qiskit import MettleQBackend, QupertinoBackend

    assert mlxq.__version__ == mettleq.__version__
    assert QupertinoBackend is MettleQBackend


def test_pennylane_registers_canonical_and_migration_device_names():
    import pennylane as qml

    canonical = qml.device("mettleq", wires=2)
    migration = qml.device("mettleq.compat", wires=2)
    assert canonical.name == "mettleq"
    assert migration.name == "mettleq"
    assert type(canonical).__name__ == "MettleQDevice"
