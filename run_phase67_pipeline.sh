#!/bin/zsh
# Resumable Phase-6/7 pipeline. Idempotent: downloaders skip existing files,
# experiments reuse cached per-region-window JSON/NPZ artifacts.
# Safe to rerun from scratch after any interruption:
#   caffeinate -i ./run_phase67_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")"

echo "=== downloads (sequential; CDS queue limits) ==="
(cd clean_experiments && python3 download_b6_cape6h.py  --out-dir ../data/b6cape  --workers 1)
(cd clean_experiments && python3 download_b6_ivt.py     --out-dir ../data/b6ivt   --workers 1)
(cd clean_experiments && python3 download_b7_budget.py  --out-dir ../data/b7budget --workers 1)

n_cape=$(ls data/b6cape/*.nc 2>/dev/null | wc -l | tr -d ' ')
n_ivt=$(ls data/b6ivt/*.nc 2>/dev/null | wc -l | tr -d ' ')
n_bud=$(ls data/b7budget/*.nc 2>/dev/null | wc -l | tr -d ' ')
echo "=== data: cape $n_cape/32, ivt $n_ivt/24, budget $n_bud/24 ==="

if [ "$n_cape" -ge 32 ]; then
  echo "=== 6A source-sink ==="
  python3 clean_experiments/experiment_B6a_source_sink.py
fi
if [ "$n_ivt" -ge 24 ]; then
  echo "=== N2 IVT profiles ==="
  python3 clean_experiments/experiment_B6n2_ivt_profiles.py
fi
if [ "$n_ivt" -ge 24 ] && [ "$n_bud" -ge 24 ]; then
  echo "=== Phase 7 moisture closure ==="
  python3 clean_experiments/experiment_B7_moisture_closure.py
fi
echo "=== pipeline pass complete ==="
