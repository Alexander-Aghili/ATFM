#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --extra dev --extra serve
uv venv --python 3.12 tmp/venvs/gpu
uv pip sync --python tmp/venvs/gpu/bin/python experiments/gpu-cache/requirements.lock
tmp/venvs/gpu/bin/python -c 'import torch, vllm, lmcache; assert torch.cuda.is_available(); print(torch.__version__, vllm.__version__, lmcache.__version__, torch.cuda.get_device_name())'
printf '%s\n' 'Ready: .venv/bin/python -m atfm_experiments.gpu_cache --output runs/gpu-check'
