#!/usr/bin/env bash
# Prepare a fresh Pod checkout for pod_round.sh: uv, serving venv, AIPerf client.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf
bash experiments/gpu-cache/setup.sh
uv venv --python 3.12 tmp/venvs/aiperf
uv pip sync --python tmp/venvs/aiperf/bin/python experiments/local-cluster/client-lock.txt
tmp/venvs/aiperf/bin/aiperf --version
