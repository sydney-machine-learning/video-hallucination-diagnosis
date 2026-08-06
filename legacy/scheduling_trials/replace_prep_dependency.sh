#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OLD_PREPARATION_JOB_ID GPU_JOB_ID" >&2
  exit 2
fi

old_prep="$1"
gpu_job="$2"
route_root="/home/z5611125/trace_rc_route"
cd "$route_root"
module load python/3.10.8
python src/verify_herbench_seal.py --route-root . >/tmp/trace_rc_seal_verify.txt

new_prep="$(qsub katana/prepare_herbench_assets.pbs)"
if ! qalter -W "depend=afterok:${new_prep}" "$gpu_job"; then
  qdel "$new_prep" 2>/dev/null || true
  printf 'NEW_PREP_SUBMITTED=%s\nGPU_DEPENDENCY_UPDATE=FAILED\n' "$new_prep"
  exit 1
fi
qdel "$old_prep"

seal_sha="$(sha256sum HERBENCH_PREINFERENCE_SEAL_20260802.json | awk '{print $1}')"
submitted_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
record="results_fresh/herbench_prep_replacement_${old_prep%%.*}_${new_prep%%.*}.json"
printf '{\n  "changed_utc": "%s",\n  "superseded_preparation_job": "%s",\n  "replacement_preparation_job": "%s",\n  "retained_gpu_job": "%s",\n  "new_dependency": "afterok:%s",\n  "reason": "parallel checksum-pinned acquisition under the 12-hour project limit",\n  "preinference_seal_sha256": "%s"\n}\n' \
  "$submitted_utc" "$old_prep" "$new_prep" "$gpu_job" "$new_prep" "$seal_sha" >"$record"
printf 'OLD_PREP_CANCELLED=%s\nNEW_PREP=%s\nGPU_JOB_RETAINED=%s\nRECORD=%s\nSEAL_SHA256=%s\n' \
  "$old_prep" "$new_prep" "$gpu_job" "$record" "$seal_sha"
