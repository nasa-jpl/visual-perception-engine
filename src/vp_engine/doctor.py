"""Validate Visual Perception Engine configs without loading GPU models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vp_engine.config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Visual Perception Engine JSON config (schema only, no GPU)."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.json",
        help="Path to engine config JSON",
    )
    args = parser.parse_args(argv)
    path = Path(args.config)
    if not path.exists():
        print(f"error: config not found: {path}", file=sys.stderr)
        return 1
    try:
        Config(str(path))
    except Exception as exc:
        print(f"NOT READY: {path}\n  {exc}", file=sys.stderr)
        return 2
    print(f"READY: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
