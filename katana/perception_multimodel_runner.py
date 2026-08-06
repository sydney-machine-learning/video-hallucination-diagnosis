#!/usr/bin/env python3
"""Three-model, four-view Perception Test valid500 TRACE-RC runner."""

from __future__ import annotations

import herbench_multimodel_runner as runner


runner.base.SCHEMA_VERSION = "trace-rc-perception-test-valid500-fresh-inference-1.0"
runner.base.SEED = 20260803

_base_record = runner.base.base_record


def base_record(*args, **kwargs):
    record = _base_record(*args, **kwargs)
    question = args[0].question
    record["schema_version"] = runner.base.SCHEMA_VERSION
    record["dataset"] = "Perception-Test-valid500"
    record["dataset_repository"] = question["dataset_repository"]
    record["dataset_revision"] = question["dataset_revision"]
    record["split"] = question["split"]
    record["video_id"] = question["video_id"]
    record["question_type"] = question["task"]
    record["reasoning"] = question["reasoning"]
    record.pop("source_dataset", None)
    record["gold_candidate_index"] = None
    return record


runner.base.base_record = base_record


if __name__ == "__main__":
    raise SystemExit(runner.base.main())
