#!/usr/bin/env python3

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: psutil\n"
        "Install it with:\n"
        "  python3 -m pip install psutil"
    ) from exc


TAG_RE = re.compile(
    r"^(?P<framework>.+)_heads(?P<num_heads>\d+)_(?P<objdet>objdet|no_objdet)$"
)
WINDOW_START_RE = re.compile(r"BENCHMARK_WINDOW_START epoch=([\d.]+)")
WINDOW_END_RE = re.compile(r"BENCHMARK_WINDOW_END epoch=([\d.]+)")
BASELINE_START_RE = re.compile(r"Processing with .+:")
RESULTS_START_MARKER = "--- Benchmark Results ---"
VP_ENGINE_START_MARKER = "Engine test successful"


def parse_tag(tag: str) -> dict:
    match = TAG_RE.match(tag)
    if not match:
        raise ValueError(
            f"Tag {tag!r} does not match expected pattern "
            "<framework>_heads<N>_(objdet|no_objdet)"
        )

    return {
        "framework": match.group("framework"),
        "num_heads": int(match.group("num_heads")),
        "obj_det_heads": match.group("objdet") == "objdet",
    }


def monitor_cpu_per_core(stop_event, samples, interval_sec):
    """Collect host CPU utilization per core while a benchmark command runs."""
    psutil.cpu_percent(interval=None, percpu=True)

    while not stop_event.is_set():
        per_core = psutil.cpu_percent(interval=interval_sec, percpu=True)
        ranked_percent = sorted(per_core, reverse=True)
        samples.append(
            {
                "timestamp": time.time(),
                "per_core_percent": per_core,
                "ranked_percent": ranked_percent,
                "total_cpu_percent": sum(per_core),
            }
        )


def update_window_markers(line: str, window_markers: dict) -> None:
    start_match = WINDOW_START_RE.search(line)
    if start_match:
        window_markers["start"] = float(start_match.group(1))
        return

    end_match = WINDOW_END_RE.search(line)
    if end_match:
        window_markers["end"] = float(end_match.group(1))
        return

    # Fallback markers for older benchmark runs without explicit timestamps.
    now = time.time()
    if window_markers["start"] is None and BASELINE_START_RE.search(line):
        window_markers["start"] = now
    if window_markers["start"] is None and VP_ENGINE_START_MARKER in line:
        window_markers["start"] = now
    if window_markers["end"] is None and RESULTS_START_MARKER in line:
        window_markers["end"] = now


def stream_command_output(proc, window_markers, output_lines):
    """Stream subprocess stdout/stderr and record benchmark window markers."""
    assert proc.stdout is not None

    for raw_line in iter(proc.stdout.readline, b""):
        line = raw_line.decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
        update_window_markers(line, window_markers)


def filter_samples_in_window(samples, window_start, window_end):
    if window_start is None or window_end is None:
        return []

    return [
        sample
        for sample in samples
        if window_start <= sample["timestamp"] <= window_end
    ]


def aggregate_window_samples(samples_in_window):
    if not samples_in_window:
        return [], 0.0, 0

    num_ranks = len(samples_in_window[0]["ranked_percent"])
    rank_sums = [0.0] * num_ranks
    total_sum = 0.0

    for sample in samples_in_window:
        for rank_idx, value in enumerate(sample["ranked_percent"]):
            rank_sums[rank_idx] += value
        total_sum += sample["total_cpu_percent"]

    count = len(samples_in_window)
    rank_avg = [round(value / count, 4) for value in rank_sums]
    avg_total_cpu_percent = round(total_sum / count, 4)
    return rank_avg, avg_total_cpu_percent, count


def append_jsonl_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark command, sample CPU utilization at 10 Hz, and "
            "aggregate utilization within the actual benchmark window parsed "
            "from benchmark logs."
        )
    )
    parser.add_argument(
        "--output-file",
        default="results_cpu_utilization.jsonl",
        help="JSONL file where aggregated CPU utilization results are appended.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag encoding framework/heads/objdet, e.g. vp_engine_heads4_objdet.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.1,
        help="CPU sampling interval in seconds (default: 0.1 = 10 Hz).",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Benchmark command to run. Put -- before the command.",
    )

    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        raise SystemExit(
            "No command provided.\n"
            "Example:\n"
            "  python3 run_with_cpu_core_monitoring.py "
            "--tag vp_engine_heads1_no_objdet -- "
            "python3 run_with_gpu_monitoring.py --framework vp_engine --num_heads 1"
        )

    metadata = parse_tag(args.tag)
    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = Path(__file__).resolve().parent / output_file

    samples = []
    stop_event = threading.Event()
    window_markers = {"start": None, "end": None}
    output_lines = []

    monitor_thread = threading.Thread(
        target=monitor_cpu_per_core,
        args=(stop_event, samples, args.interval_sec),
        daemon=True,
    )

    print("=" * 80)
    print(f"CPU monitor tag: {args.tag}")
    print(f"Sampling interval: {args.interval_sec:.3f} sec")
    print(f"Output JSONL: {output_file}")
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    monitor_thread.start()

    return_code = 0
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        stream_command_output(proc, window_markers, output_lines)
        return_code = proc.wait()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating benchmark command...")
        return_code = 130

        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
    finally:
        stop_event.set()
        monitor_thread.join(timeout=args.interval_sec + 2.0)

    samples_in_window = filter_samples_in_window(
        samples,
        window_markers["start"],
        window_markers["end"],
    )
    rank_avg, avg_total_cpu_percent, num_samples_in_window = aggregate_window_samples(
        samples_in_window
    )

    benchmark_window_sec = None
    if window_markers["start"] is not None and window_markers["end"] is not None:
        benchmark_window_sec = round(
            window_markers["end"] - window_markers["start"],
            4,
        )

    record = {
        **metadata,
        "rank_avg_cpu_percent": rank_avg,
        "avg_total_cpu_percent": avg_total_cpu_percent,
        "num_samples_in_window": num_samples_in_window,
        "benchmark_window_sec": benchmark_window_sec,
    }

    append_jsonl_record(output_file, record)

    print("\nBenchmark window markers:")
    print(f"  start: {window_markers['start']}")
    print(f"  end:   {window_markers['end']}")
    print(f"  duration_sec: {benchmark_window_sec}")
    print(f"  samples in window: {num_samples_in_window} / {len(samples)}")

    if rank_avg:
        print("\nAverage CPU utilization by rank (highest-util core first):")
        for rank_idx, value in enumerate(rank_avg, start=1):
            print(f"  rank {rank_idx:02d}: {value:.2f}%")
        print(f"\nAverage total CPU utilization (sum across cores): {avg_total_cpu_percent:.2f}%")
    else:
        print(
            "\nWarning: no CPU samples fell within the parsed benchmark window. "
            "Check that benchmark logs contained recognizable start/end markers."
        )

    print(f"\nResults appended to {output_file}")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
