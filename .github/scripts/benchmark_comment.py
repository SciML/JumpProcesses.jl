"""Renders the benchmark comment from AirspeedVelocity's results files.

Run by .github/workflows/benchmark-comment.yml from the default branch. The results come from a
workflow that ran the pull request's code, so they are untrusted: only validated numbers, and
benchmark names made of a restricted set of characters, reach the comment. The tables follow
AirspeedVelocity's: the median time ± the interquartile range, the ratio of the medians with its
propagated error, and allocations and memory.

usage: python3 benchmark_comment.py <results dir> <head sha> > comment.md

The results dir holds one directory per Julia version, benchmark-results-<version>, each with
results_JumpProcesses@master.json and results_JumpProcesses@<head sha>.json.
"""

import json
import math
import os
import re
import statistics
import sys

MARKER = "<!-- benchmark-results -->"
PACKAGE = "JumpProcesses"
BASE = "master"
MAX_FILE_BYTES = 50_000_000
MAX_SAMPLES = 100_000
MAX_BENCHMARKS = 500
MAX_COMMENT_CHARS = 60_000
NAME = re.compile(r"[A-Za-z0-9_./-]{1,100}")
VERSION_DIR = re.compile(r"benchmark-results-([A-Za-z0-9.]{1,10})")
SHA = re.compile(r"[0-9a-f]{40}")

TIME_UNITS = (("ns", 1.0), ("μs", 1e-3), ("ms", 1e-6), ("s", 1e-9), ("h", 1e-9 / 3600))
MEMORY_UNITS = (("B", 1.0), ("kB", 1 / 1024), ("MB", 1 / 1024**2), ("GB", 1 / 1024**3))
ALLOCATION_UNITS = (("", 1.0), ("k", 1e-3), ("M", 1e-6), ("G", 1e-9))


def unit(value, units):
    """The unit whose scaled value is closest to 1, as AirspeedVelocity picks it."""
    if value <= 0:
        return units[0]
    return min(units, key=lambda u: abs(math.log10(value * u[1])))


def number(x, integer=False):
    """A finite, non-negative JSON number as a float; raises TypeError or ValueError."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError
    try:
        x = float(x)
    except OverflowError:
        raise ValueError from None
    if not math.isfinite(x) or x < 0 or (integer and not x.is_integer()):
        raise ValueError
    return x


def summary(trial):
    """Median and interquartile range of the times, and the memory and allocations."""
    times = trial.get("times")
    if not isinstance(times, list) or not 0 < len(times) <= MAX_SAMPLES:
        raise ValueError
    times = [number(t) for t in times]
    stats = {"median": statistics.median(times), "memory": None, "allocs": None}
    if "memory" in trial and "allocs" in trial:
        stats["memory"] = number(trial["memory"], integer=True)
        stats["allocs"] = number(trial["allocs"], integer=True)
    if len(times) > 1:
        q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
        stats["iqr"] = max(0.0, q3 - q1)
    return stats


def flatten(group, prefix, out, skipped):
    """Benchmark name => summary, for the trials in a serialized BenchmarkGroup."""
    data = group.get("data") if isinstance(group, dict) else None
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        name = key if not prefix else prefix + "/" + key
        if not NAME.fullmatch(name) or len(out) >= MAX_BENCHMARKS:
            skipped[0] += 1
        elif isinstance(value, dict) and "times" in value:
            try:
                out[name] = summary(value)
            except (ValueError, TypeError):
                skipped[0] += 1
        else:
            flatten(value, name, out, skipped)


def load(path, skipped):
    if not os.path.isfile(path) or os.path.getsize(path) > MAX_FILE_BYTES:
        return None
    try:
        with open(path, encoding="utf-8") as io:
            results = json.load(io)
    except (OSError, ValueError, RecursionError):
        return None
    out = {}
    flatten(results, "", out, skipped)
    return out


def format_time(stats):
    name, scale = unit(stats["median"], TIME_UNITS)
    median = stats["median"] * scale
    if "iqr" in stats:
        return f"{median:.3g} ± {stats['iqr'] * scale:.2g} {name}"
    return f"{median:.3g} {name}"


def format_memory(stats):
    if stats["allocs"] is None:
        return ""
    allocs_name, allocs_scale = unit(stats["allocs"], ALLOCATION_UNITS)
    memory_name, memory_scale = unit(stats["memory"], MEMORY_UNITS)
    allocs = f"{stats['allocs'] * allocs_scale:.3g}" + (
        " " + allocs_name if allocs_name else ""
    )
    return f"{allocs} allocs: {stats['memory'] * memory_scale:.3g} {memory_name}"


def ratio(base, head, key):
    if key == "median":
        a, b = base["median"], head["median"]
    else:
        a, b = base[key], head[key]
        if a is None or b is None:
            return ""
    if b == 0:
        return ""
    r = a / b
    if key == "median" and "iqr" in base and "iqr" in head and a > 0:
        err = abs(r) * math.sqrt((base["iqr"] / a) ** 2 + (head["iqr"] / b) ** 2)
        return f"{r:.3g} ± {err:.2g}"
    return f"{r:.3g}"


def table(base, head, short_sha, key, format_cell):
    names = sorted(set(base) | set(head), key=lambda n: (n == "time_to_load", n))
    lines = [
        f"| | {BASE} | {short_sha} | {BASE} / {short_sha} |",
        "|:---|:---:|:---:|:---:|",
    ]
    for name in names:
        b, h = base.get(name), head.get(name)
        cells = [
            format_cell(b) if b else "",
            format_cell(h) if h else "",
            ratio(b, h, key) if b and h else "",
        ]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    return "\n".join(lines)


def render(results_dir, head_sha):
    if not SHA.fullmatch(head_sha):
        raise SystemExit("not a commit SHA: " + head_sha)
    short_sha = head_sha[:7]
    parts = [MARKER]
    versions = sorted(
        m.group(1)
        for d in os.listdir(results_dir)
        if (m := VERSION_DIR.fullmatch(d))
        and os.path.isdir(os.path.join(results_dir, d))
    )
    for version in versions:
        directory = os.path.join(results_dir, "benchmark-results-" + version)
        skipped = [0]
        base = load(os.path.join(directory, f"results_{PACKAGE}@{BASE}.json"), skipped)
        head = load(
            os.path.join(directory, f"results_{PACKAGE}@{head_sha}.json"), skipped
        )
        parts.append(f"## Benchmark Results (Julia v{version})")
        if base is None or head is None:
            parts.append("No valid results for both revisions.")
            continue
        for title, key, format_cell in (
            ("Time", "median", format_time),
            ("Memory", "memory", format_memory),
        ):
            rows = table(base, head, short_sha, key, format_cell)
            parts.append(
                f"<details><summary>{title} benchmarks</summary>\n\n{rows}\n\n</details>"
            )
        if skipped[0]:
            parts.append(
                f"{skipped[0]} entries with unexpected names or values were left out."
            )
    comment = "\n\n".join(parts) + "\n"
    if len(comment) > MAX_COMMENT_CHARS:
        comment = comment[:MAX_COMMENT_CHARS] + "\n\n(truncated)\n"
    return comment


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.stdout.write(render(sys.argv[1], sys.argv[2]))
