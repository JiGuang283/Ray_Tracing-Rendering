#!/usr/bin/env python3
"""Generate reproducible CPU/CUDA baselines for ReSTIR development.

The suite definition is tracked, while rendered PNG/PFM files and result JSON
stay below output/. This keeps large reference artifacts out of Git without
losing the exact commands, hashes, linear statistics, or memory measurements.
"""

from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "assets/scenes/restir_baseline_suite.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=DEFAULT_SUITE,
        help="baseline suite JSON (default: %(default)s)")
    parser.add_argument(
        "--mode", choices=("regression", "reference", "memory", "all"),
        default="regression", help="suite section to run")
    parser.add_argument(
        "--backend", choices=("manifest", "cpu", "cuda", "both"),
        default="manifest", help="override per-case backend list")
    parser.add_argument(
        "--case", action="append", default=[],
        help="run only a named case; may be repeated")
    parser.add_argument(
        "--cpu-executable", type=Path,
        default=PROJECT_ROOT / "build/CGAssignment4")
    parser.add_argument(
        "--cuda-executable", type=Path,
        default=PROJECT_ROOT / "build-cuda/CGAssignment4")
    parser.add_argument(
        "--output", type=Path,
        help="artifact root override (default comes from suite)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="reuse an existing PNG/PFM pair instead of rendering")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print commands without executing them")
    parser.add_argument(
        "--write-lock", type=Path,
        help="write hashes and recorded memory sizes to a lock JSON")
    parser.add_argument(
        "--verify-lock", type=Path,
        help="verify generated or reused artifacts against a lock JSON")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_scalar(value: str) -> Any:
    try:
        if value.lower() in {"nan", "inf", "+inf", "-inf"}:
            return float(value)
        if not any(character in value for character in ".eE"):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_bench_line(output: str, prefix: str) -> Dict[str, Any]:
    matching = [line for line in output.splitlines()
                if line.startswith(prefix + " ")]
    if not matching:
        raise RuntimeError(f"renderer did not emit {prefix}")
    fields: Dict[str, Any] = {}
    for token in matching[-1].split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = parse_scalar(value)
    return fields


def quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1,
                max(0, int(math.ceil(fraction * len(sorted_values))) - 1))
    return sorted_values[index]


def pfm_statistics(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        dimensions = stream.readline().split()
        scale_line = stream.readline().strip()
        if magic != b"PF" or len(dimensions) != 2:
            raise ValueError(f"{path}: expected an RGB PFM file")
        width, height = (int(value) for value in dimensions)
        scale = float(scale_line)
        payload = stream.read()

    expected_values = width * height * 3
    values = array.array("f")
    values.frombytes(payload)
    if len(values) != expected_values:
        raise ValueError(
            f"{path}: expected {expected_values} floats, got {len(values)}")
    file_little_endian = scale < 0.0
    host_little_endian = sys.byteorder == "little"
    if file_little_endian != host_little_endian:
        values.byteswap()

    rgb_sum = [0.0, 0.0, 0.0]
    luminance: List[float] = []
    invalid = 0
    for index in range(0, len(values), 3):
        red = float(values[index])
        green = float(values[index + 1])
        blue = float(values[index + 2])
        if not (math.isfinite(red) and math.isfinite(green) and
                math.isfinite(blue)):
            invalid += 1
            continue
        rgb_sum[0] += red
        rgb_sum[1] += green
        rgb_sum[2] += blue
        luminance.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)

    valid = len(luminance)
    luminance.sort()
    inverse_valid = 1.0 / valid if valid else 0.0
    return {
        "width": width,
        "height": height,
        "valid_pixels": valid,
        "invalid_pixels": invalid,
        "mean_rgb": [channel * inverse_valid for channel in rgb_sum],
        "mean_luminance": sum(luminance) * inverse_valid,
        "p99_luminance": quantile(luminance, 0.99),
        "p999_luminance": quantile(luminance, 0.999),
        "max_luminance": luminance[-1] if luminance else 0.0,
    }


def selected_sections(mode: str) -> Iterable[str]:
    if mode == "all":
        return ("regression", "reference", "memory")
    return (mode,)


def selected_backends(case: Dict[str, Any], override: str) -> List[str]:
    if override == "cpu":
        return ["cpu"]
    if override == "cuda":
        return ["cuda"]
    if override == "both":
        return ["cpu", "cuda"]
    backends = case.get("backends")
    if not isinstance(backends, list) or not backends:
        raise ValueError(f"case {case.get('name')}: missing backends")
    return [str(backend) for backend in backends]


def executable_for(backend: str, args: argparse.Namespace) -> Path:
    if backend == "cpu":
        return args.cpu_executable.resolve()
    if backend == "cuda":
        return args.cuda_executable.resolve()
    raise ValueError(f"unsupported backend '{backend}'")


def build_command(executable: Path, case: Dict[str, Any], backend: str,
                  png_path: Path | None, pfm_path: Path | None) -> List[str]:
    command = [
        str(executable), str(case["scene_id"]), str(case["integrator"]),
        "--bench", "--backend", backend,
        "--width", str(case["width"]),
        "--spp", str(case["spp"]),
        "--max-depth", str(case["max_depth"]),
        "--seed", str(case["seed"]),
        "--sample-clamp", "0",
    ]
    if backend == "cpu":
        command.extend(["--threads", str(case.get("threads", 0))])
    if png_path is not None:
        command.extend(["--save-image", str(png_path)])
    if pfm_path is not None:
        command.extend(["--save-linear", str(pfm_path)])
    return command


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=True, text=True, capture_output=True)
    return result.stdout.strip()


def result_key(result: Dict[str, Any]) -> str:
    return ":".join(
        (str(result["section"]), str(result["name"]),
         str(result["backend"])))


def build_lock(report: Dict[str, Any]) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    for result in report["results"]:
        entry: Dict[str, Any] = {"settings": result["settings"]}
        artifacts = result.get("artifacts")
        if artifacts is not None:
            entry["png_sha256"] = artifacts["png_sha256"]
            entry["pfm_sha256"] = artifacts["pfm_sha256"]
            entry["linear_statistics"] = result["linear_statistics"]
        summary = result.get("summary")
        if summary is not None:
            entry["recorded_memory"] = {
                "scene_bytes": summary.get("scene_bytes", 0),
                "workspace_bytes": summary.get("workspace_bytes", 0),
                "workspace_pixel_capacity":
                    summary.get("workspace_pixel_capacity", 0),
                "workspace_path_capacity":
                    summary.get("workspace_path_capacity", 0),
            }
        entries[result_key(result)] = entry
    return {
        "format": "restir_baseline_lock_v1",
        "suite": report["suite"],
        "source_revision": report["git_revision"],
        "entries": entries,
    }


def verify_lock(results: Sequence[Dict[str, Any]], lock: Dict[str, Any]) -> None:
    if lock.get("format") != "restir_baseline_lock_v1":
        raise ValueError("unsupported ReSTIR baseline lock format")
    entries = lock.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("baseline lock is missing entries")
    failures: List[str] = []
    for result in results:
        key = result_key(result)
        expected = entries.get(key)
        if not isinstance(expected, dict):
            failures.append(f"{key}: missing lock entry")
            continue
        if expected.get("settings") != result.get("settings"):
            failures.append(f"{key}: render settings differ")
        artifacts = result.get("artifacts")
        if artifacts is not None:
            for field in ("png_sha256", "pfm_sha256"):
                if expected.get(field) != artifacts.get(field):
                    failures.append(f"{key}: {field} differs")
        recorded_memory = expected.get("recorded_memory")
        summary = result.get("summary")
        if isinstance(recorded_memory, dict) and isinstance(summary, dict):
            for field in ("scene_bytes", "workspace_bytes",
                          "workspace_pixel_capacity",
                          "workspace_path_capacity"):
                if recorded_memory.get(field) != summary.get(field, 0):
                    failures.append(f"{key}: {field} differs")
    if failures:
        raise RuntimeError("baseline lock verification failed:\n  " +
                           "\n  ".join(failures))


def run_case(section: str, case: Dict[str, Any], backend: str,
             artifact_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    name = str(case["name"])
    case_root = artifact_root / section / name / backend
    save_artifacts = bool(case.get("save_artifacts", True))
    png_path = case_root / "beauty.png" if save_artifacts else None
    pfm_path = case_root / "beauty.pfm" if save_artifacts else None
    executable = executable_for(backend, args)
    command = build_command(executable, case, backend, png_path, pfm_path)

    print(f"[{section}:{name}:{backend}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return {"section": section, "name": name, "backend": backend,
                "command": command, "dry_run": True}
    if not executable.is_file():
        raise FileNotFoundError(f"renderer executable not found: {executable}")

    reuse = (args.skip_existing and png_path is not None and
             pfm_path is not None and png_path.is_file() and
             pfm_path.is_file())
    output = ""
    if not reuse:
        case_root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        output = completed.stdout
        print(output, end="" if output.endswith("\n") else "\n")
        if completed.returncode != 0:
            raise RuntimeError(
                f"baseline render failed with exit code {completed.returncode}")

    result: Dict[str, Any] = {
        "section": section,
        "name": name,
        "scene_id": case["scene_id"],
        "backend": backend,
        "settings": {
            key: case[key] for key in
            ("width", "spp", "max_depth", "seed", "integrator")
        },
        "command": command,
        "reused_artifacts": reuse,
    }
    if output:
        result["preparation"] = parse_bench_line(output, "BENCH_PREP")
        result["summary"] = parse_bench_line(output, "BENCH_SUMMARY")
    if png_path is not None and pfm_path is not None:
        if not png_path.is_file() or not pfm_path.is_file():
            raise RuntimeError(f"missing baseline artifacts for {name}/{backend}")
        result["artifacts"] = {
            "png": str(png_path.relative_to(PROJECT_ROOT)),
            "png_sha256": sha256(png_path),
            "pfm": str(pfm_path.relative_to(PROJECT_ROOT)),
            "pfm_sha256": sha256(pfm_path),
            "pfm_bytes": pfm_path.stat().st_size,
        }
        result["linear_statistics"] = pfm_statistics(pfm_path)
    return result


def main() -> int:
    args = parse_args()
    suite_path = args.suite.resolve()
    suite = load_json(suite_path)
    if suite.get("format") != "restir_baseline_suite_v1":
        raise ValueError(f"{suite_path}: unsupported suite format")

    artifact_root = args.output
    if artifact_root is None:
        artifact_root = PROJECT_ROOT / str(suite["artifact_root"])
    artifact_root = artifact_root.resolve()
    selected_names = set(args.case)

    results: List[Dict[str, Any]] = []
    for section in selected_sections(args.mode):
        cases = suite.get(section)
        if not isinstance(cases, list):
            raise ValueError(f"{suite_path}: section '{section}' is not a list")
        for case in cases:
            if not isinstance(case, dict) or "name" not in case:
                raise ValueError(f"{suite_path}: invalid case in '{section}'")
            if selected_names and case["name"] not in selected_names:
                continue
            for backend in selected_backends(case, args.backend):
                results.append(run_case(
                    section, case, backend, artifact_root, args))

    if selected_names:
        found = {result["name"] for result in results}
        missing = selected_names - found
        if missing:
            raise ValueError("unknown or excluded case(s): " +
                             ", ".join(sorted(missing)))
    if args.dry_run:
        print(f"Dry run complete: {len(results)} render(s).")
        return 0

    artifact_root.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "restir_baseline_results_v1",
        "suite": str(suite_path.relative_to(PROJECT_ROOT)),
        "git_revision": git_revision(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results": results,
    }
    if args.verify_lock is not None:
        verify_lock(results, load_json(args.verify_lock.resolve()))
        print(f"Verified baseline lock {args.verify_lock}")
    if args.write_lock is not None:
        lock_path = args.write_lock.resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as stream:
            json.dump(build_lock(report), stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"Wrote baseline lock {lock_path}")
    result_path = artifact_root / f"{args.mode}_results.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Wrote {result_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
