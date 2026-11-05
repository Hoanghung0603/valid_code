#!/usr/bin/env bash
# usage: bash scripts/run_limited.sh <log> <command...>   (laptop guard: one heavy job at a time)
# CPU: 4 threads for torch / numpy BLAS, below-normal priority (set by the scripts). RAM: stop below 0.6 GB free.
cd "$(dirname "$0")/.."
mkdir -p data/logs
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
LOG=$1; shift
"$@" > "$LOG" 2>&1 &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  FREE=$(powershell -NoProfile -command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)")
  echo "$(date '+%H:%M') $FREE GB free" >> data/logs/ram.log
  if awk "BEGIN{exit !($FREE < 0.6)}"; then
    echo "RAMGUARD: $FREE GB free, stopping" | tee -a "$LOG" >> data/logs/ram.log
    kill $JOB
    break
  fi
  sleep 30
done
wait $JOB; echo "exit $?" >> "$LOG"
