#!/usr/bin/env bash
# Run one unattended round of public-trace replays on a rented GPU Pod.
# Usage: pod_round.sh <capacity|repeat-chunk> [watchdog seconds]
# Each step writes runs/round/<plan>/<step>/ plus <step>.tar.gz (L2 KV objects excluded).
# The watchdog stops the Pod as a backstop; the operator still downloads and stops it.
set -uo pipefail
plan=${1:?plan: capacity or repeat-chunk}
cap_s=${2:-6900}
cd "$(dirname "$0")/../.."
out="$PWD/runs/round/$plan"
mkdir -p "$out"
export CUDA_HOME="$PWD/tmp/venvs/gpu/lib/python3.12/site-packages/nvidia/cu13"
export HF_HOME=/root/hf
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  nohup bash -c "sleep $cap_s; runpodctl stop pod $RUNPOD_POD_ID" >"$out/watchdog.log" 2>&1 &
fi

step() {
  local name=$1 rc=0
  shift
  echo "$(date -u +%FT%TZ) start $name" >>"$out/steps.log"
  "$@" >"$out/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >>"$out/steps.log"
  tar -C "$out" --exclude='l2' -czf "$out/$name.tar.gz" "$name" "$name.log" 2>/dev/null
  rm -rf "$out/$name/l2"
}

replay() {
  local name=$1
  shift
  step "$name" .venv/bin/python -m atfm_experiments.gpu_cache.replay \
    --source runs/public-selection --output "$out/$name" "$@"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >"$out/gpu.csv"
free -g >"$out/memory.txt"
case "$plan" in
  capacity)
    replay multi-l1-24 --case multi-branch --l1-gb 24 --case-timeout 2400 --total-timeout 2460
    replay multi-l1-48 --case multi-branch --l1-gb 48 --case-timeout 2400 --total-timeout 2460
    ;;
  repeat-chunk)
    step tools .venv/bin/python -m atfm_experiments.gpu_cache.tool_matrix --output "$out/tools"
    for rep in 1 2 3; do replay "rep$rep" --case short-branch --case sequential; done
    replay chunk-64 --case sequential --chunk-size 64
    replay chunk-256 --case sequential --chunk-size 256
    ;;
  *) echo "unknown plan: $plan" >&2; exit 2 ;;
esac
echo "$(date -u +%FT%TZ) DONE" >>"$out/steps.log"
