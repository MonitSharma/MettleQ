#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from mettleq.benchmark_environment import capture_performance_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture_performance_environment(), indent=2) + "\n")


if __name__ == "__main__":
    main()
