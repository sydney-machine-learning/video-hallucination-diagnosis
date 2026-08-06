#!/bin/bash
set -Eeuo pipefail

module purge
module load python/3.10.8
module load cuda/12.8.0
source /srv/scratch/z5611125/sicr/envs/llava-ov2/bin/activate
export HF_HOME=/srv/scratch/z5611125/sicr/huggingface
export HF_HUB_CACHE=/srv/scratch/z5611125/sicr/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=1 nohup python -u \
  /home/z5611125/trace_rc_route/katana/herbench_tail_half_runner.py \
  --backend llava \
  --manifest /home/z5611125/trace_rc_route/input/herbench_lite_v2_inference.jsonl \
  --output /srv/scratch/z5611125/sicr/trace_rc_herbench/results/llava_tail_predictions.jsonl \
  --state-dir /srv/scratch/z5611125/sicr/trace_rc_herbench/state/llava_tail \
  --output-prefix herbench-llava-tail \
  --resume --checkpoint-every 10 \
  --stop-file /srv/scratch/z5611125/sicr/trace_rc_herbench/state/GRACEFUL_STOP_REQUESTED \
  > /srv/scratch/z5611125/sicr/trace_rc_herbench/logs/llava_tail.8915415.log \
  2>&1 < /dev/null &

echo $!
