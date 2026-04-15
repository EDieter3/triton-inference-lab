# Prometheus + Grafana Setup

## Prometheus
- Runs on port 9090
- Scrapes:
  - itself at localhost:9090
  - Triton metrics at host.docker.internal:8002

## Grafana
- Runs on port 3000
- Uses Prometheus at host.docker.internal:9090 as a data source

## Useful URLs
- Prometheus UI: http://localhost:9090
- Prometheus targets: http://localhost:9090/targets
- Grafana UI: http://localhost:3000
