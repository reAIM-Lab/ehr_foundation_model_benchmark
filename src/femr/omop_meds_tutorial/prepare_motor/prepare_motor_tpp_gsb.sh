#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the directory of THIS script (works regardless of how you invoke it)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LOGFILE="${SCRIPT_DIR}/prepare_motor_tpp_output.log"

# Ensure directory exists and is writable
mkdir -p -- "$SCRIPT_DIR"
: > "$LOGFILE" 2>/dev/null || { echo "Cannot write to $LOGFILE"; exit 1; }

# Use process substitution if supported on this host
if (: > >(cat) ) 2>/dev/null; then
  # Make tee line-buffered so you see lines immediately
  exec > >(stdbuf -oL -eL tee -a "$LOGFILE") 2>&1
else
  echo "Process substitution not available; using pipeline fallback" >&2
  USE_PIPELINE_FALLBACK=1
fi

echo "Starting MOTOR preparation script at $(date)"
echo "Log: $LOGFILE"
echo "================================="

# Force unbuffered Python so prints appear in real time
export PYTHONUNBUFFERED=1

if [[ "${USE_PIPELINE_FALLBACK:-0}" -eq 1 ]]; then
  # Portable fallback: pipe stdout+stderr into tee
  python -u prepare_motor_tpp.py \
    --pretraining_data /user/zj2398/cache/deephit_tpp_8k \
    --athena_path " " \
    --num_bins 8 \
    --num_threads 100 \
    --meds_reader /user/zj2398/cache/mimic/meds_v0.6_reader \
  |& tee -a "$LOGFILE"
  exit ${PIPESTATUS[0]}
else
  # Normal path (exec already redirected both stdout/stderr through tee)
  python -u prepare_motor_tpp.py \
    --pretraining_data /user/zj2398/cache/deephit_tpp_8k \
    --athena_path " " \
    --num_bins 8 \
    --num_threads 100 \
    --meds_reader /user/zj2398/cache/mimic/meds_v0.6_reader 
fi

