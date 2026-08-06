#!/usr/bin/env bash
set -euo pipefail

ROUTE_ROOT="/home/${USER}/trace_rc_route"
cd "$ROUTE_ROOT"

module load python/3.10.8
python src/verify_herbench_seal.py --route-root . >/tmp/trace_rc_seal_verify.txt

prep_job="$(qsub katana/prepare_herbench_assets.pbs)"
if ! gpu_job="$(qsub -W "depend=afterok:${prep_job}" katana/run_herbench_consolidated.pbs)"; then
  printf 'PREP_JOB_SUBMITTED=%s\nGPU_JOB_SUBMISSION=FAILED\n' "$prep_job"
  exit 1
fi

seal_sha="$(sha256sum HERBENCH_PREINFERENCE_SEAL_20260802.json | awk '{print $1}')"
submitted_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
record="results_fresh/herbench_submission_${prep_job%%.*}_${gpu_job%%.*}.json"
mkdir -p results_fresh
printf '{\n  "submitted_utc": "%s",\n  "preparation_job": "%s",\n  "gpu_job": "%s",\n  "dependency": "afterok:%s",\n  "preinference_seal_sha256": "%s"\n}\n' \
  "$submitted_utc" "$prep_job" "$gpu_job" "$prep_job" "$seal_sha" >"$record"

printf 'PREP_JOB=%s\nGPU_JOB=%s\nRECORD=%s\nSEAL_SHA256=%s\n' \
  "$prep_job" "$gpu_job" "$record" "$seal_sha"
