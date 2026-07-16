#!/usr/bin/env python3
"""Run the published P9 solver with an explicit sampling seed.

The published CLI seeds Sabre routing but calls ``mps.sample`` without its
available seed argument. This narrow harness supplies the requested seed only
when the solver omits it, leaving compression and all other CLI behavior
unchanged. It lets MettleQ and the published solver share an explicit sampling
contract during same-machine comparisons.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampling-seed", type=int, required=True)
    known, remaining = parser.parse_known_args()

    from quimb.tensor import MatrixProductState
    from p9solver.cli import main as published_main

    original_sample = MatrixProductState.sample

    def sample_with_contract(self, count, seed=None, *args, **kwargs):
        return original_sample(
            self,
            count,
            seed=known.sampling_seed if seed is None else seed,
            *args,
            **kwargs,
        )

    MatrixProductState.sample = sample_with_contract
    result = published_main(remaining)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
