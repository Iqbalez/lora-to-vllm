$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python -m nimbus_llm dataset --out data/processed
python -m pytest tests/test_dataset.py tests/test_api.py
python -m nimbus_llm train --config configs/train_smoke.yaml
python -m nimbus_llm evaluate --base sshleifer/tiny-gpt2 --adapter outputs/smoke-tiny-gpt2/adapter --test data/processed/test.jsonl --out results/eval_smoke.json --limit 8 --max-new-tokens 32
Write-Host "Smoke train/eval finished. Start the API with:"
Write-Host "  python -m nimbus_llm serve --backend transformers --model sshleifer/tiny-gpt2 --adapter outputs/smoke-tiny-gpt2/adapter"
