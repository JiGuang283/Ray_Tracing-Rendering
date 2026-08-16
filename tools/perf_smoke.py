#!/usr/bin/env python3
"""Structured performance smoke runner for CGAssignment4.

Each preset executes the application with --bench-json and reports the median
wall/device seconds plus sample throughput.  The script intentionally does not
assert hard timing thresholds by default: it produces a stable, machine-readable
report that CI or a later regression policy can compare against baselines.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    "cpu_scene23": {
        "description": "CPU MIS path, Cornell-style scene 23",
        "build_dir": "build",
        "args": ["23", "4", "--bench", "--width", "400", "--spp", "32",
                 "--max-depth", "50", "--seed", "123"],
    },
    "cpu_packed_scene23": {
        "description": "CPU packed backend prototype, scene 23",
        "build_dir": "build",
        "args": ["23", "4", "--bench", "--cpu-packed",
                 "--width", "400", "--spp", "32", "--max-depth", "50",
                 "--seed", "123"],
    },
    "cpu_scene59": {
        "description": "CPU mesh path, Suzanne scene 59",
        "build_dir": "build",
        "args": ["59", "4", "--bench", "--width", "400", "--spp", "16",
                 "--max-depth", "50", "--seed", "123"],
    },
    "cuda_scene23_depth4": {
        "description": "CUDA wavefront, scene 23, short depth loop",
        "build_dir": "build-cuda",
        "args": ["23", "4", "--bench", "--backend", "cuda",
                 "--width", "400", "--spp", "16", "--max-depth", "4",
                 "--seed", "123"],
    },
    "cuda_scene23_depth50": {
        "description": "CUDA wavefront, scene 23, full depth loop",
        "build_dir": "build-cuda",
        "args": ["23", "4", "--bench", "--backend", "cuda",
                 "--width", "400", "--spp", "16", "--max-depth", "50",
                 "--seed", "123"],
    },
    "cuda_restir_di": {
        "description": "CUDA ReSTIR DI, many-light scene 65",
        "build_dir": "build-cuda",
        "args": ["65", "5", "--bench", "--backend", "cuda",
                 "--width", "320", "--spp", "8", "--max-depth", "2",
                 "--seed", "123"],
    },
}

GROUPS = {
    "review-baseline": [
        "cpu_scene23",
        "cpu_scene59",
        "cuda_scene23_depth4",
        "cuda_scene23_depth50",
        "cuda_restir_di",
    ],
}


def binary_path(build_dir):
    return os.path.join(ROOT, build_dir, "CGAssignment4")


def run_preset(name, build_dir, runs, timeout):
    if name not in PRESETS:
        raise ValueError(f"unknown preset '{name}'")
    preset = PRESETS[name]
    effective_build = build_dir or preset["build_dir"]
    executable = binary_path(effective_build)
    if not os.path.isfile(executable):
        raise RuntimeError(f"benchmark binary not found: {executable}")

    with tempfile.NamedTemporaryFile(
            prefix=f"perf_smoke_{name}_", suffix=".json",
            delete=False) as tmp:
        report_path = tmp.name

    command = [executable] + preset["args"] + [
        "--runs", str(runs), "--bench-json", report_path]

    started = time.time()
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"preset {name} failed with exit code {exc.returncode}: "
            f"{exc.stderr.decode(errors='replace')[-500:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"preset {name} timed out after {timeout}s") from exc
    finally:
        if not os.path.exists(report_path):
            try:
                os.unlink(report_path)
            except OSError:
                pass

    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"preset {name} produced no parseable benchmark JSON") from exc
    finally:
        try:
            os.unlink(report_path)
        except OSError:
            pass

    wall_seconds = [run["stats"]["seconds"] for run in report["runs"]]
    device_seconds = [run["stats"]["device_seconds"]
                      for run in report["runs"]]
    if not wall_seconds:
        raise RuntimeError(f"preset {name} produced no benchmark runs")

    return {
        "preset": name,
        "description": preset["description"],
        "binary": executable,
        "elapsed_seconds": time.time() - started,
        "runs": len(wall_seconds),
        "median_seconds": statistics.median(wall_seconds),
        "median_device_seconds": statistics.median(device_seconds),
        "wall_seconds": wall_seconds,
        "device_seconds": device_seconds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", default=[],
                        help="preset name or group name; may be repeated")
    parser.add_argument("--list", action="store_true",
                        help="list available presets and groups")
    parser.add_argument("--build-dir", default="",
                        help="build directory to use for all presets")
    parser.add_argument("--runs", type=int, default=3,
                        help="benchmark runs per preset (default 3)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-preset timeout in seconds")
    parser.add_argument("--output", default="",
                        help="write JSON report to this path")
    args = parser.parse_args(argv)

    if args.list:
        print("Presets:")
        for name, preset in sorted(PRESETS.items()):
            print(f"  {name}: {preset['description']}")
        print("Groups:")
        for name, members in sorted(GROUPS.items()):
            print(f"  {name}: {', '.join(members)}")
        return 0

    selected = []
    for item in args.preset or ["review-baseline"]:
        if item in GROUPS:
            selected.extend(GROUPS[item])
        else:
            selected.append(item)

    report = {"generator": "tools/perf_smoke.py",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "results": []}
    failures = []
    for name in selected:
        try:
            result = run_preset(name, args.build_dir, args.runs,
                                args.timeout)
            report["results"].append(result)
            print(f"PASS {name:24s} "
                  f"median={result['median_seconds']:.6f}s "
                  f"device={result['median_device_seconds']:.6f}s "
                  f"runs={result['runs']}")
        except Exception as exc:  # noqa: BLE001 - report all preset errors
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    if failures:
        print(f"{len(failures)} preset(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
