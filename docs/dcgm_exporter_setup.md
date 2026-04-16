# DCGM Exporter Setup

## Purpose
DCGM Exporter was added to expose NVIDIA GPU telemetry to Prometheus and Grafana.

## Container
- Image: `nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.6.0-ubuntu22.04`
- Port: `9400`

## Prometheus target
- `host.docker.internal:9400`

## Example Grafana queries
- `DCGM_FI_DEV_GPU_UTIL`
- `DCGM_FI_DEV_FB_USED`
- `DCGM_FI_DEV_POWER_USAGE`
- `DCGM_FI_DEV_GPU_TEMP`
