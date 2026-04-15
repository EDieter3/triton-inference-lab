# Triton Inference Lab

## Overview
This project is a local NVIDIA Triton Inference Server lab built on:
- Windows host
- WSL2 Ubuntu
- Docker Desktop with WSL integration
- NVIDIA GPU acceleration
- Triton Inference Server

The goal of this lab was to:
- stand up a working Triton model-serving environment
- validate inference requests and outputs
- benchmark throughput and latency
- compare the effects of batch size and client concurrency

## Environment
- Host: Windows PC
- Linux environment: WSL2 Ubuntu
- Container runtime: Docker Desktop with WSL integration
- GPU: NVIDIA GeForce RTX 5080
- Triton version: 26.03
- Model used: `simple` example model

## What I Built
- Brought up Triton Inference Server in Docker with GPU access
- Validated health, readiness, model metadata, and metrics endpoints
- Sent inference requests to the `simple` model
- Built Python benchmark scripts to test:
  - batch size = 1, 4, 8
  - concurrency = 1, 4, 8
- Captured benchmark output to text files and CSV
- Generated throughput and p95 latency charts

## Benchmark Matrix
The benchmark tested combinations of:
- Batch size 1, concurrency 1
- Batch size 1, concurrency 4
- Batch size 1, concurrency 8
- Batch size 4, concurrency 1
- Batch size 4, concurrency 4
- Batch size 8, concurrency 4

## Key Findings
1. Increasing batch size significantly improved inference throughput.
   - Larger batches increased the amount of useful work completed per request.
   - Batch size 8 with concurrency 4 produced the highest inference throughput in this lab.

2. Moderate concurrency improved throughput, but higher concurrency did not always help.
   - Moving from concurrency 1 to 4 improved throughput substantially.
   - Moving from concurrency 4 to 8 for batch size 1 increased latency without delivering proportional throughput gains.

3. Tail latency increased as concurrency increased.
   - p95 and p99 latency were noticeably higher at concurrency 8 than at concurrency 4 for batch size 1.
   - This suggests that higher concurrency introduced overhead or contention in this setup.

4. Batch size and concurrency affect different parts of performance.
   - Batch size increased work per request.
   - Concurrency increased the number of requests in flight.
   - The best operating point in this test appeared to be a balance of moderate concurrency and larger batch sizes.

## Best Observed Configurations
- Best small-request latency: batch size 1, concurrency 1
- Best small-request throughput balance: batch size 1, concurrency 4
- Best overall throughput: batch size 8, concurrency 4
- Best overall balance of throughput and latency: batch size 4, concurrency 4

## Artifacts
- `artifacts/benchmark_results.csv`
- `artifacts/throughput_chart.png`
- `artifacts/latency_chart.png`
- `artifacts/benchmark_run_01.txt`
- `artifacts/benchmark_run_02_concurrency.txt`
- `artifacts/benchmark_run_03_latency.txt`
- `artifacts/benchmark_run_04_csv.txt`

## Next Steps
- Add Prometheus and Grafana for live dashboarding
- Test a more realistic ONNX or generative AI model
- Compare Triton behavior with different model types
- Move the lab into Kubernetes with NVIDIA GPU Operator
- Add automated CSV exports and additional benchmark scenarios


## Charts

### Throughput
![Throughput Chart](artifacts/throughput_chart.png)

### p95 Latency
![Latency Chart](artifacts/latency_chart.png)

## Resume-Relevant Takeaways
- Built and benchmarked a local NVIDIA Triton inference lab with GPU acceleration
- Measured request throughput, inference throughput, and latency across batch-size and concurrency scenarios
- Used Triton metrics to analyze request count, inference count, and execution behavior
