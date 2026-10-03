"""Validate release archives without importing the source package.

Usage: python tools/check_distribution.py dist/ [--version 0.3.0]
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import struct
import tarfile
import zipfile

REQUIRED = {
    "mettleq/__init__.py", "mettleq/py.typed", "mlxq/__init__.py",
    "mettleq/_mps_native.cpp", "mettleq/shaders/single_qubit.py",
    "mettleq/integrations/pennylane_capabilities.toml",
    "mettleq/datasets/qasm_local/bell.qasm",
    "mettleq/datasets/peaked_circuit_P9_Hqap_56x1917.qasm",
    "mettleq/datasets/ATTRIBUTION.md",
    "mettleq/datasets/LICENSE-APACHE-2.0.txt",
    "mettleq/_vendor/peaked_mpo/NOTICE.md",
    "mettleq/_vendor/peaked_mpo/LICENSE-APACHE-2.0.txt",
}


def check_metadata(data: bytes, expected_version: str | None) -> str:
    metadata = BytesParser().parsebytes(data)
    assert metadata["Name"] == "mettleq", metadata["Name"]
    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata["License-Expression"] == "MIT AND Apache-2.0"
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert len(metadata.get_all("License-File", [])) == 3
    assert "Apple Silicon" in metadata.get_payload()
    if expected_version:
        assert metadata["Version"] == expected_version, metadata["Version"]
    return metadata["Version"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--apple-silicon", action="store_true",
                        help="Require macOS 14 arm64 tags and an arm64 native binary")
    args = parser.parse_args()
    files = sorted(args.directory.iterdir())
    archives = [p for p in files if p.name.endswith((".whl", ".tar.gz"))]
    assert archives, f"No distributions in {args.directory}"
    versions = set()
    for path in archives:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                assert REQUIRED <= names, REQUIRED - names
                metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
                assert len(metadata_paths) == 1
                versions.add(check_metadata(archive.read(metadata_paths[0]), args.version))
                prefix = metadata_paths[0].removesuffix("METADATA")
                entries = archive.read(prefix + "entry_points.txt").decode()
                assert "mettleq-mpo = mettleq.midpoint_mpo:main" in entries
                assert "mettleq = mettleq.integrations.pennylane:MettleQDevice" in entries
                licenses = [n for n in names if n.startswith(prefix + "licenses/")]
                assert len(licenses) == 3, licenses
                native = [n for n in names if n.startswith("mettleq/_mps_native.")
                          and n.endswith((".so", ".pyd"))]
                if args.require_native or args.apple_silicon:
                    assert native, f"Native extension missing in {path}"
                if args.apple_silicon:
                    assert path.name.endswith("-macosx_14_0_arm64.whl"), path.name
                    wheel_metadata = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
                    tags = wheel_metadata.get_all("Tag", [])
                    assert tags and all(tag.endswith("-macosx_14_0_arm64")
                                        for tag in tags), tags
                    for name in native:
                        # Mach-O 64-bit magic and CPU_TYPE_ARM64. Reject fat or
                        # Intel binaries even if a filename claims arm64.
                        header = archive.read(name)[:8]
                        assert len(header) == 8, name
                        assert struct.unpack("<II", header) == (0xFEEDFACF, 0x0100000C), name
                assert not any(n.split("/")[0] in {"tests", "examples", "tools"}
                               for n in names)
                assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
        else:
            with tarfile.open(path) as archive:
                members = archive.getnames()
                root = members[0].split("/")[0]
                names = {n.removeprefix(root + "/") for n in members}
                required = {"src/" + n for n in REQUIRED} | {
                    "pyproject.toml", "setup.py", "LICENSE", "PYPI_README.md",
                    "tools/check_distribution.py", "tools/smoke_installed.py",
                    "src/tests/test_packaging.py", "src/examples/qpe_energy_estimation.py",
                    "docs/releasing.md",
                }
                assert required <= names, required - names
                metadata_file = archive.extractfile(root + "/PKG-INFO")
                assert metadata_file is not None
                versions.add(check_metadata(metadata_file.read(), args.version))
                assert not any(n.endswith((".so", ".pyd", ".pyc")) for n in names)
        print(f"Verified {path.name}")
    assert len(versions) == 1, versions


if __name__ == "__main__":
    main()
