#!/usr/bin/env python3
"""Run a low-cost render smoke test for every scene in the catalog."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "assets/scenes/catalog.json"
DEFAULT_BINARY = PROJECT_ROOT / "build/CGAssignment4"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--integrator", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    scenes = catalog.get("scenes", [])
    if not args.binary.exists():
        print(f"ERROR: binary does not exist: {args.binary}", file=sys.stderr)
        return 1

    failures = []
    for item in scenes:
        scene_id = item["id"]
        command = [
            str(args.binary),
            str(scene_id),
            str(args.integrator),
            "--bench",
            "--width",
            str(args.width),
            "--spp",
            str(args.spp),
            "--runs",
            "1",
            "--seed",
            str(args.seed),
            "--threads",
            str(args.threads),
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            failures.append((scene_id, result.stderr.strip()))
            print(f"FAIL scene {scene_id}: {result.stderr.strip()}",
                  file=sys.stderr)
            if args.stop_on_failure:
                break
        else:
            print(f"OK scene {scene_id}")

    if failures:
        print(f"Catalog smoke failed: {len(failures)} / {len(scenes)} scenes.")
        return 1

    print(f"Catalog smoke passed: {len(scenes)} scene(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
