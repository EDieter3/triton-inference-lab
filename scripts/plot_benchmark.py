import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

ARTIFACT_DIR = os.path.expanduser("~/triton-lab/artifacts")
CSV_PATH = os.path.join(ARTIFACT_DIR, "benchmark_results.csv")

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "batch_size": int(row["batch_size"]),
            "concurrency": int(row["concurrency"]),
            "requests_per_sec": float(row["requests_per_sec"]),
            "inferences_per_sec": float(row["inferences_per_sec"]),
            "p95_ms": float(row["p95_ms"]),
        })

throughput_groups = defaultdict(list)
latency_groups = defaultdict(list)

for row in sorted(rows, key=lambda r: (r["batch_size"], r["concurrency"])):
    throughput_groups[row["batch_size"]].append((row["concurrency"], row["inferences_per_sec"]))
    latency_groups[row["batch_size"]].append((row["concurrency"], row["p95_ms"]))

plt.figure(figsize=(8, 5))
for batch_size, points in sorted(throughput_groups.items()):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker="o", label=f"batch={batch_size}")
plt.xlabel("Concurrency")
plt.ylabel("Inferences/sec")
plt.title("Triton Throughput by Batch Size and Concurrency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "throughput_chart.png"))
plt.close()

plt.figure(figsize=(8, 5))
for batch_size, points in sorted(latency_groups.items()):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker="o", label=f"batch={batch_size}")
plt.xlabel("Concurrency")
plt.ylabel("p95 Latency (ms)")
plt.title("Triton p95 Latency by Batch Size and Concurrency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "latency_chart.png"))
plt.close()

print(f"Charts written to {ARTIFACT_DIR}")
