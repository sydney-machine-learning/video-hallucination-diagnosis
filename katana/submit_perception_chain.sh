#!/bin/bash
set -Eeuo pipefail

PROJECT="/home/${USER}/trace_rc_route"
PREP_JOB=$(qsub "${PROJECT}/katana/prepare_perception_valid500.pbs")
GPU_JOB=$(qsub -W "depend=afterok:${PREP_JOB}" "${PROJECT}/katana/run_perception_valid500.pbs")
printf 'PREP_JOB=%s\nGPU_JOB=%s\n' "${PREP_JOB}" "${GPU_JOB}"
