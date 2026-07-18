#!/usr/bin/env python3
"""Fail a release audit on common secrets, private paths, or oversized files."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private_home_path": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    "github_token": re.compile(rb"gh[ps]_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret_assignment": re.compile(
        rb"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}"
    ),
}


def _git_files():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-mib", type=float, default=50.0)
    args = parser.parse_args()
    findings = []
    for path in _git_files():
        if not path.is_file():
            continue
        size = path.stat().st_size
        relative = path.relative_to(ROOT)
        if size > args.max_file_mib * 1024**2:
            findings.append(f"oversized_file: {relative} ({size / 1024**2:.1f} MiB)")
        if size > 50 * 1024**2:
            continue
        data = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append(f"{label}: {relative}")
    if findings:
        print("Public-release audit failed:")
        print("\n".join(f"- {finding}" for finding in sorted(set(findings))))
        return 1
    print("Public-release audit passed for the current tracked tree.")
    print("Run a separate history scan/rewrite before changing repository visibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
