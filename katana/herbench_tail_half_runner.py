#!/usr/bin/env python3
"""Execution-only helper for the second half of the sealed HERBench work list.

This wrapper does not change prompts, views, model revisions, labels, or record
construction.  It partitions the already sealed, label-free work list at a
question boundary so an otherwise idle GPU can checkpoint disjoint LLaVA
records during walltime-limited recovery.
"""

from __future__ import annotations

import herbench_multimodel_runner as her


_load_manifest = her.base.load_manifest


def load_tail_half(path, roles, views, max_questions):
    if max_questions is not None:
        raise ValueError("tail-half recovery does not accept --max-questions")
    items = _load_manifest(path, roles, views, max_questions)
    group_size = len(views)
    if len(items) % group_size:
        raise ValueError("sealed work list is not divisible by the view count")
    question_count = len(items) // group_size
    first_tail_question = question_count // 2
    return items[first_tail_question * group_size :]


her.base.load_manifest = load_tail_half


if __name__ == "__main__":
    raise SystemExit(her.base.main())
