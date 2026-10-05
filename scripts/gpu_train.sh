#!/usr/bin/env bash
# Run on a Linux box with a 24 GB NVIDIA GPU (A10 / A5000 / L40S / A100).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m nimbus_llm dataset --out data/processed
python -m nimbus_llm train --config configs/train_qwen3b.yaml
python -m nimbus_llm evaluate \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/qwen25-3b-nimbus-lora/adapter \
  --test data/processed/test.jsonl \
  --out results/eval.json \
  --max-new-tokens 128
python -m nimbus_llm quantize \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/qwen25-3b-nimbus-lora/adapter \
  --merged-out outputs/merged \
  --quant-out outputs/awq-int4 \
  --calib data/processed/train.jsonl

echo "Merged + AWQ written. Start serving with:"
echo "  docker compose -f docker/docker-compose.yml up --build"
echo "Then:"
echo "  python -m nimbus_llm benchmark --url http://127.0.0.1:8000/v1/chat/completions --concurrency 8 --repeats 32 --out results/benchmark.json"
