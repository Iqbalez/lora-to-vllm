#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m nimbus_llm dataset --out data/processed
python -m pytest tests/test_dataset.py tests/test_api.py
python -m nimbus_llm train --config configs/train_smoke.yaml
python -m nimbus_llm evaluate \
  --base sshleifer/tiny-gpt2 \
  --adapter outputs/smoke-tiny-gpt2/adapter \
  --test data/processed/test.jsonl \
  --out results/eval_smoke.json \
  --limit 8 \
  --max-new-tokens 32

echo "Smoke train/eval finished. Serve with:"
echo "  python -m nimbus_llm serve --backend transformers --model sshleifer/tiny-gpt2 --adapter outputs/smoke-tiny-gpt2/adapter"
