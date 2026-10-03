"""Run the complete pytest suite, preserving its failure exit status."""
from pathlib import Path
import sys


def main() -> int:
    import pytest

    tests = Path(__file__).resolve().parent
    return int(pytest.main([str(tests), *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
