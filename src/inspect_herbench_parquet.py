"""Print an auditable compact inventory for a HERBench parquet release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.parquet)
    report = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "unique_videos": int(frame["video_id"].nunique()),
        "candidate_count_distribution": {
            str(key): int(value)
            for key, value in frame["choices"].map(len).value_counts().sort_index().items()
        },
        "task_counts": {
            str(key): int(value)
            for key, value in frame["task_type"].value_counts().sort_index().items()
        },
        "source_counts": {
            str(key): int(value)
            for key, value in frame["source_dataset"].value_counts().sort_index().items()
        },
        "first_record": frame.head(1).to_dict(orient="records")[0],
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
