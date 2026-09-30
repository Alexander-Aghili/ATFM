#!/usr/bin/env bash
# Run one unattended round of public-trace replays on a rented GPU Pod.
# Usage: pod_round.sh <plan> [watchdog seconds]
# <plan> names experiments/gpu-cache/plans/<plan>.txt: one "<step> <tools|replay|retrieval> [args]" per line.
# Each step writes runs/round/<plan>/<step>/ plus <step>.tar.gz (L2 KV objects excluded).
# The watchdog stops the Pod as a backstop; the operator still downloads and stops it.
set -uo pipefail
plan=${1:?plan name under experiments/gpu-cache/plans}
cap_s=${2:-6900}
cd "$(dirname "$0")/../.."
out="$PWD/runs/round/$plan"
mkdir -p "$out"
export CUDA_HOME="$PWD/tmp/venvs/gpu/lib/python3.12/site-packages/nvidia/cu13"
export HF_HOME=/root/hf
plan_file="experiments/gpu-cache/plans/$plan.txt"
test -f "$plan_file" || { echo "missing $plan_file" >&2; exit 2; }
cp "$plan_file" "$out/plan.txt"
# SSH sessions lack the container's Runpod variables; PID 1 has them.
eval "$(tr '\0' '\n' </proc/1/environ 2>/dev/null | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  nohup bash -c "sleep $cap_s; runpodctl stop pod $RUNPOD_POD_ID" >"$out/watchdog.log" 2>&1 &
fi

step() {
  local name=$1 rc=0
  shift
  echo "$(date -u +%FT%TZ) start $name" >>"$out/steps.log"
  "$@" </dev/null >"$out/$name.log" 2>&1 || rc=$?
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
while read -r name kind args; do
  case "$kind" in
    tools) step "$name" .venv/bin/python -m atfm_experiments.gpu_cache.tool_matrix --output "$out/$name" ;;
    replay) replay "$name" $args ;;
    retrieval) step "$name" .venv/bin/python -m atfm_experiments.gpu_cache.retrieval --output "$out/$name" $args ;;
    *) echo "$(date -u +%FT%TZ) skip $name: unknown kind $kind" >>"$out/steps.log" ;;
  esac
done < <(grep -v '^#' "$plan_file" | grep -v '^$')
echo "$(date -u +%FT%TZ) DONE" >>"$out/steps.log"
