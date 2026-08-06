#!/usr/bin/env python3
"""Run the first half of the sealed Perception Test question list."""

from __future__ import annotations

import perception_multimodel_runner as perception


_load_manifest = perception.runner.base.load_manifest


def load_head(path, roles, views, max_questions):
    if max_questions is not None:
        raise ValueError("half runner does not accept --max-questions")
    items = _load_manifest(path, roles, views, max_questions)
    group_size = len(views)
    if len(items) % group_size:
        raise ValueError("work list is not divisible by view count")
    boundary = (len(items) // group_size // 2) * group_size
    return items[:boundary]


perception.runner.base.load_manifest = load_head


if __name__ == "__main__":
    raise SystemExit(perception.runner.base.main())
