# Perception Test valid500 execution log

All times and job states below precede model inference unless stated otherwise.

## Benchmark choice

- LongVideoBench was rejected because its gated media archive was too large
  for the available scratch allocation.
- MLVU was rejected because its official media required gated access and its
  compressed test set exceeded the practical storage budget.
- Google DeepMind Perception Test was selected because it is public, provides
  official validation labels and cut frames, and supports video-level sampling
  without downloading the complete 70.2 GB archive.
- Official repository commit:
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.

## Frozen sample

- Sampling frame: 5,260 validation videos with MCQA annotations.
- Selection: first 500 IDs under a salted SHA-256 order; no question text,
  option, or answer was used.
- Split: a second salted SHA-256 order assigns 250 videos to calibration and
  250 to final testing.
- Inventory: 1,888 questions (977 calibration, 911 final test).
- Final manifest SHA-256:
  `614862e5fd8ec344cf2cdcd76f1b18dbf5192567f23021dea0d9744982fcc389`.
- Pilot-label SHA-256:
  `eb60c3ef1dc0e8162b742d03f04e452b7255e690844ec96dd3fc27bbe2f08e6a`.
- Sealed-test-label SHA-256:
  `f37f1e50e6e0df98be4802073a1884e3c447d1ad48d4179dfd598fbaaeefc952`.

## Media preparation

- The selected members were read from the official range-addressable ZIP on
  `kdm.restech.unsw.edu.au`; the complete 70.2 GB archive was not stored.
- 500 MP4 files total 6,228,470,128 bytes.
- The extraction inventory SHA-256 is
  `8cb0ae8db19b5f721e37fff8ffe7c52b949991938141eaac78328249d27e41ff`.
- A second pass verified the recorded size and SHA-256 of every MP4 before
  writing `PREP_COMPLETE.json`.

## Pre-inference amendments

- `PREINFERENCE_AMENDMENT_20260803_POSIX_PATH.json` records a Windows-to-POSIX
  path serialization correction. Selection, roles, questions, options, and
  labels did not change; no model inference had started.
- `PREINFERENCE_AMENDMENT_20260803_FOUR_GPU.json` records the change from a
  two-GPU serial schedule to four concurrent GPUs after PBS reported a
  dedicated-time conflict. The registered model panel and analysis are
  unchanged.

## PBS history

- `8919080`/`8919081`: superseded CPU-prep dependency chain.
- `8919082`/`8919083`: superseded after moving pure data transfer to KDM.
- `8919087`: held during media preparation; superseded before inference.
- `8919109`: generic four-GPU request; superseded before inference after the
  scheduler continued to report a dedicated-time conflict.
- `8919110`: current single-allocation request, pinned to the unreserved k206
  H200 node; queued with 4 H200 GPUs and 05:58 walltime.

## Post-maintenance scheduling recovery (5 August 2026)

- Katana accepted SSH again at 08:42:50 AEST after an extended maintenance
  window. Job `8919110` remained queued and retained its accumulated eligible
  time; it was not deleted or resubmitted.
- The scheduler cleared its dedicated-time conflict after 09:04. By then all
  eight GPUs on k206 had been assigned to other jobs.
- A transient diagnostic rebind to the physically idle k175 exposed an
  incompatibility between a fixed host and the queue-managed `generalgpu12`
  placement set. The job did not start under that request.
- At 09:28:50 AEST the host constraint was removed from the same job. The final
  request remains 4 H200 GPUs, 32 CPUs, 320 GB, and 05:58 walltime, but PBS may
  now choose any eligible H200 placement set. The scheduler subsequently
  reported only transient GPU availability rather than a configuration that
  could never run.
- No inference had started during these scheduling changes. The frozen data,
  model panel, views, analysis, and success gate were unchanged. Full details
  are recorded in
  `EXECUTION_AMENDMENT_20260805_POSTMAINTENANCE_SCHEDULING.json`.

## Three-GPU earliest-finish alternative (5 August 2026)

- At 12:30 AEST, PBS estimated that the four-GPU job would start at 20:38,
  while one node exposed three idle H200 GPUs.
- A single-allocation three-GPU wrapper was added. Qwen, InternVL, and the
  LLaVA head half start together; the first released GPU immediately processes
  the LLaVA tail half. It still produces all 22,656 registered
  model--question--view records and performs both analysis phases in the same
  allocation.
- The original four-GPU job is retained only until one alternative actually
  starts. The other job is then held before inference, so duplicate inference
  is not allowed.
- PBS assigned candidate job `8921298` an estimated start of 21:08, later than
  the four-GPU job's 20:38 estimate. Because the three-GPU schedule also has a
  longer runtime, it could not improve time to the complete result. It was
  deleted before inference, and `8919110` remains the sole queued job.
- The scheduling-only amendment is recorded in
  `EXECUTION_AMENDMENT_20260805_THREE_GPU_EARLIEST_FINISH.json`.

## Short-backfill prefill (5 August 2026)

- At 16:01 AEST, the main job still had a 20:35 estimated start while three
  H200 GPUs were physically idle but unavailable for its six-hour window.
- A 1:58 short-backfill candidate uses the same frozen runners to precompute
  both LLaVA halves and InternVL. It writes to job-specific temporary files.
- Only successful records are merged atomically into the canonical resumable
  outputs. The main job is held only while prefill inference is actually
  running and is released before the prefill exits, preventing concurrent
  writers without losing its original queue record.
- The protocol is recorded in
  `EXECUTION_AMENDMENT_20260805_SHORT_BACKFILL_PREFILL.json`.
- PBS estimated the three-GPU candidate `8922648` at 21:08, later than the
  unchanged 20:35 main-job estimate, so it was deleted before execution.
- A narrower candidate, `8922656`, requests one H200 for 1:28 and precomputes
  InternVL records. It has no permission to delay the main route: at startup
  it requires a safe window ending 15 minutes before the main estimate and
  skips automatically if fewer than 30 useful minutes remain.
- PBS also estimated this one-GPU candidate at 21:08. It was therefore deleted
  before execution. The physically unassigned GPUs were scheduler-reserved,
  and no backfill request from three down to one H200 could start earlier than
  the retained four-GPU job.
- At 16:19, a guarded two-H200 candidate `8922679` was submitted to exploit
  any transient two- or three-GPU release. It precomputes Qwen and InternVL for
  at most 95 useful minutes and atomically merges successful records. Its
  initial estimate was 21:08, but it remains queued as an opportunity: it can
  run only while the main job is queued and a safe window of at least 30
  minutes exists, ending 15 minutes before the main estimate.

## Completed confirmation (6 August 2026)

- The two-H200 prefill job `8922679` ran from 17:20 to 18:57 AEST and exited
  with status 0. It checkpointed and atomically merged 3,368 Qwen records and
  2,376 InternVL records before releasing the main job.
- The four-H200 job `8919110` ran on `k201` from 21:09 on 5 August to 01:47 on
  6 August (04:38:36 walltime) and exited with status 0. All 22,656 registered
  model--question--view records completed; inference, pilot analysis, final
  analysis, and bundle stages all passed.
- The primary confirmation gate passed. Macro relative AURC reduction was
  18.8680% with a video-grouped 95% bootstrap interval of [15.4289%, 22.1395%].
  Qwen and LLaVA were promoted; InternVL used exact raw-confidence fallback,
  giving a worst-model gain of 0.0%.
- The sealed analysis archive was downloaded to
  `results/perception_confirmation/8919110.kman.restech.unsw.edu.au` and every
  file listed in `archive.sha256` passed local SHA-256 verification.
