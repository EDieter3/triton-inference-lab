import csv
import json
import math
import os
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TRITON_URL = "http://localhost:8000/v2/models/simple/infer"
METRICS_URL = "http://localhost:8002/metrics"
ARTIFACT_DIR = os.path.expanduser("~/triton-lab/artifacts")
CSV_PATH = os.path.join(ARTIFACT_DIR, "benchmark_results.csv")


def build_request(batch_size: int) -> bytes:
    input0 = []
    input1 = []

    for row in range(1, batch_size + 1):
        input0.extend([row] * 16)
        input1.extend([row * 10] * 16)

    payload = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [batch_size, 16],
                "datatype": "INT32",
                "data": input0,
            },
            {
                "name": "INPUT1",
                "shape": [batch_size, 16],
                "datatype": "INT32",
                "data": input1,
            },
        ],
        "outputs": [
            {"name": "OUTPUT0"},
            {"name": "OUTPUT1"},
        ],
    }
    return json.dumps(payload).encode("utf-8")


def send_request(batch_size: int):
    data = build_request(batch_size)
    req = urllib.request.Request(
        TRITON_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    latency_ms = (time.perf_counter() - start) * 1000.0

    return json.loads(body), latency_ms


def fetch_metrics_text() -> str:
    with urllib.request.urlopen(METRICS_URL, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_metric(metrics_text: str, metric_name: str, model_name: str = "simple") -> int:
    for line in metrics_text.splitlines():
        if (
            line.startswith(metric_name)
            and f'model="{model_name}"' in line
            and 'version="1"' in line
        ):
            try:
                return int(float(line.strip().split()[-1]))
            except (ValueError, IndexError):
                return 0
    return 0


def get_simple_metrics() -> dict:
    text = fetch_metrics_text()
    return {
        "request_success": parse_metric(text, "nv_inference_request_success"),
        "inference_count": parse_metric(text, "nv_inference_count"),
        "exec_count": parse_metric(text, "nv_inference_exec_count"),
    }


def validate_response(response: dict, batch_size: int) -> None:
    outputs = {o["name"]: o["data"] for o in response["outputs"]}

    expected_output0 = []
    expected_output1 = []

    for row in range(1, batch_size + 1):
        expected_output0.extend([row + row * 10] * 16)
        expected_output1.extend([row - row * 10] * 16)

    if outputs["OUTPUT0"] != expected_output0:
        raise ValueError("OUTPUT0 did not match expected values")
    if outputs["OUTPUT1"] != expected_output1:
        raise ValueError("OUTPUT1 did not match expected values")


def percentile(sorted_values, pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return sorted_values[low]

    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def summarize_latencies(latencies_ms):
    values = sorted(latencies_ms)
    return {
        "avg_ms": statistics.mean(values) if values else 0.0,
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def worker(batch_size: int, requests_per_worker: int):
    completed = 0
    latencies = []

    for _ in range(requests_per_worker):
        response, latency_ms = send_request(batch_size)
        validate_response(response, batch_size)
        completed += 1
        latencies.append(latency_ms)

    return completed, latencies


def ensure_csv():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "batch_size",
                "concurrency",
                "requests_per_worker",
                "total_requests",
                "total_inferences",
                "delta_request_success",
                "delta_inference_count",
                "delta_exec_count",
                "total_seconds",
                "requests_per_sec",
                "inferences_per_sec",
                "avg_ms",
                "min_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
            ])


def append_csv(row: dict):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["batch_size"],
            row["concurrency"],
            row["requests_per_worker"],
            row["total_requests"],
            row["total_inferences"],
            row["delta_request_success"],
            row["delta_inference_count"],
            row["delta_exec_count"],
            f'{row["total_seconds"]:.6f}',
            f'{row["requests_per_sec"]:.2f}',
            f'{row["inferences_per_sec"]:.2f}',
            f'{row["avg_ms"]:.3f}',
            f'{row["min_ms"]:.3f}',
            f'{row["p50_ms"]:.3f}',
            f'{row["p95_ms"]:.3f}',
            f'{row["p99_ms"]:.3f}',
            f'{row["max_ms"]:.3f}',
        ])


def run_benchmark(batch_size: int, concurrency: int, requests_per_worker: int) -> None:
    total_requests = concurrency * requests_per_worker
    expected_inferences = total_requests * batch_size

    print(f"\n=== Batch size {batch_size}, concurrency {concurrency}, {requests_per_worker} requests/worker ===")
    print(f"Total planned requests: {total_requests}")
    print(f"Total planned inferences: {expected_inferences}")

    before = get_simple_metrics()
    print(f"Before metrics: {before}")

    start = time.perf_counter()

    completed_requests = 0
    all_latencies_ms = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker, batch_size, requests_per_worker)
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            completed, latencies = future.result()
            completed_requests += completed
            all_latencies_ms.extend(latencies)

    elapsed = time.perf_counter() - start

    after = get_simple_metrics()
    print(f"After metrics:  {after}")

    delta_requests = after["request_success"] - before["request_success"]
    delta_inferences = after["inference_count"] - before["inference_count"]
    delta_execs = after["exec_count"] - before["exec_count"]

    print("Metric deltas:")
    print(f"  request_success: +{delta_requests}")
    print(f"  inference_count: +{delta_inferences}")
    print(f"  exec_count:      +{delta_execs}")

    reqs_per_sec = completed_requests / elapsed if elapsed > 0 else 0
    infs_per_sec = delta_inferences / elapsed if elapsed > 0 else 0

    print("Timing:")
    print(f"  total_seconds:   {elapsed:.4f}")
    print(f"  requests/sec:    {reqs_per_sec:.2f}")
    print(f"  inferences/sec:  {infs_per_sec:.2f}")

    latency_summary = summarize_latencies(all_latencies_ms)
    print("Latency (per request):")
    print(f"  avg_ms:          {latency_summary['avg_ms']:.3f}")
    print(f"  min_ms:          {latency_summary['min_ms']:.3f}")
    print(f"  p50_ms:          {latency_summary['p50_ms']:.3f}")
    print(f"  p95_ms:          {latency_summary['p95_ms']:.3f}")
    print(f"  p99_ms:          {latency_summary['p99_ms']:.3f}")
    print(f"  max_ms:          {latency_summary['max_ms']:.3f}")

    append_csv({
        "batch_size": batch_size,
        "concurrency": concurrency,
        "requests_per_worker": requests_per_worker,
        "total_requests": total_requests,
        "total_inferences": expected_inferences,
        "delta_request_success": delta_requests,
        "delta_inference_count": delta_inferences,
        "delta_exec_count": delta_execs,
        "total_seconds": elapsed,
        "requests_per_sec": reqs_per_sec,
        "inferences_per_sec": infs_per_sec,
        "avg_ms": latency_summary["avg_ms"],
        "min_ms": latency_summary["min_ms"],
        "p50_ms": latency_summary["p50_ms"],
        "p95_ms": latency_summary["p95_ms"],
        "p99_ms": latency_summary["p99_ms"],
        "max_ms": latency_summary["max_ms"],
    })


def main() -> None:
    print("Checking Triton health...")
    health = subprocess.run(
        ["curl", "-s", "http://localhost:8000/v2/health/ready"],
        capture_output=True,
        text=True,
        check=False,
    )
    if health.returncode != 0:
        raise RuntimeError("Could not reach Triton ready endpoint")
    print("Triton ready endpoint responded.")

    ensure_csv()

    test_matrix = [
        {"batch_size": 1, "concurrency": 1, "requests_per_worker": 25},
        {"batch_size": 1, "concurrency": 4, "requests_per_worker": 25},
        {"batch_size": 1, "concurrency": 8, "requests_per_worker": 25},
        {"batch_size": 4, "concurrency": 1, "requests_per_worker": 25},
        {"batch_size": 4, "concurrency": 4, "requests_per_worker": 25},
        {"batch_size": 8, "concurrency": 4, "requests_per_worker": 25},
    ]

    for test in test_matrix:
        run_benchmark(
            batch_size=test["batch_size"],
            concurrency=test["concurrency"],
            requests_per_worker=test["requests_per_worker"],
        )

    print(f"\nCSV written to: {CSV_PATH}")


if __name__ == "__main__":
    main()
