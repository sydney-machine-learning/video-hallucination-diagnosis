"""Audit the double review and final adjudication of the VHD question bank.

The script reads the reviewer exports and final adjudication tables without
modifying them.  Agreement is computed only where a label is applicable.  For
example, language-prior and visual-answer labels are assessed only for PDI
questions that both reviewers marked as valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABEL_FIELDS = (
    "question_valid",
    "final_correct_answer",
    "final_A_language_prior",
    "final_A_vision",
    "final_uncertainty_label",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean(value: str | None) -> str:
    return (value or "").strip().lower()


def required_fields(row_a: dict[str, str], row_b: dict[str, str]) -> tuple[str, ...]:
    fields = ["question_valid"]
    if clean(row_a.get("question_valid")) != "yes" or clean(row_b.get("question_valid")) != "yes":
        return tuple(fields)
    fields.append("final_correct_answer")
    task = clean(row_a.get("task"))
    if task == "vhd_i_pdi":
        fields.extend(("final_A_language_prior", "final_A_vision"))
    elif task == "vhd_ii_hce":
        fields.append("final_uncertainty_label")
    return tuple(fields)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    labels = sorted({label for pair in pairs for label in pair})
    expected = sum(
        (sum(a == label for a, _ in pairs) / len(pairs))
        * (sum(b == label for _, b in pairs) / len(pairs))
        for label in labels
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def audit_pair(
    source: str,
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
) -> tuple[dict, dict[str, list[tuple[str, str]]]]:
    by_id_a = {row["question_id"]: row for row in rows_a}
    by_id_b = {row["question_id"]: row for row in rows_b}
    if set(by_id_a) != set(by_id_b):
        missing_a = sorted(set(by_id_b) - set(by_id_a))
        missing_b = sorted(set(by_id_a) - set(by_id_b))
        raise ValueError(
            f"{source}: reviewer inventories differ; "
            f"missing from reviewer 1={missing_a[:5]}, reviewer 2={missing_b[:5]}"
        )

    pairs_by_field: dict[str, list[tuple[str, str]]] = {
        field: [] for field in LABEL_FIELDS
    }
    disagreements = 0
    incomplete = 0
    task_counts: Counter[str] = Counter()
    for question_id in sorted(by_id_a):
        row_a = by_id_a[question_id]
        row_b = by_id_b[question_id]
        if clean(row_a.get("task")) != clean(row_b.get("task")):
            raise ValueError(f"{source}: task mismatch for {question_id}")
        task_counts[clean(row_a.get("task"))] += 1
        row_disagrees = False
        for field in required_fields(row_a, row_b):
            value_a = clean(row_a.get(field))
            value_b = clean(row_b.get(field))
            if not value_a or not value_b:
                incomplete += 1
                continue
            pairs_by_field[field].append((value_a, value_b))
            row_disagrees |= value_a != value_b
        disagreements += int(row_disagrees)

    agreement = {}
    for field, pairs in pairs_by_field.items():
        agreement[field] = {
            "n_pairs": len(pairs),
            "percent_agreement": 100.0 * sum(a == b for a, b in pairs) / len(pairs)
            if pairs
            else None,
            "cohen_kappa": cohen_kappa(pairs),
        }
    return (
        {
            "source": source,
            "double_reviewed_questions": len(by_id_a),
            "questions_with_any_pre_adjudication_disagreement": disagreements,
            "incomplete_applicable_label_pairs": incomplete,
            "task_counts": dict(sorted(task_counts.items())),
            "agreement": agreement,
        },
        pairs_by_field,
    )


def final_inventory(dataset_root: Path) -> dict:
    metadata = dataset_root / "metadata"
    real = read_csv(metadata / "real_question_adjudication.csv")
    generated = read_csv(metadata / "ai_generated_question_adjudication.csv")
    rows = [("real", row) for row in real] + [("ai_generated", row) for row in generated]
    unresolved = [
        row["question_id"]
        for _, row in rows
        if clean(row.get("needs_adjudication")) == "yes"
        or clean(row.get("final_question_valid")) not in {"yes", "no"}
    ]
    valid = [(source, row) for source, row in rows if clean(row.get("final_question_valid")) == "yes"]
    return {
        "candidate_questions": len(rows),
        "accepted_questions": len(valid),
        "rejected_questions": len(rows) - len(valid),
        "retention_percent": 100.0 * len(valid) / len(rows),
        "accepted_by_source": dict(Counter(source for source, _ in valid)),
        "accepted_by_task": dict(sorted(Counter(clean(row.get("task")) for _, row in valid).items())),
        "unresolved_question_ids": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vhd"))
    args = parser.parse_args()

    real_dir = args.dataset_root / "hitl_real_question_review"
    generated_dir = args.dataset_root / "hitl_question_review"
    real_a = [
        row
        for row in read_csv(real_dir / "real_question_reviewer_1.csv")
        if row["question_id"].startswith("real_")
    ]
    real_b = read_csv(real_dir / "real_question_reviewer_2_completed.csv")
    generated_a = read_csv(generated_dir / "question_reviewer_1.csv")
    generated_b = read_csv(generated_dir / "question_reviewer_2_completed.csv")

    source_reports = []
    combined_pairs = {field: [] for field in LABEL_FIELDS}
    for source, first, second in (
        ("real", real_a, real_b),
        ("ai_generated", generated_a, generated_b),
    ):
        report, pairs = audit_pair(source, first, second)
        source_reports.append(report)
        for field in LABEL_FIELDS:
            combined_pairs[field].extend(pairs[field])

    queue = read_csv(generated_dir / "question_adjudication_queue.csv")
    overall_agreement = {}
    for field, pairs in combined_pairs.items():
        overall_agreement[field] = {
            "n_pairs": len(pairs),
            "percent_agreement": 100.0 * sum(a == b for a, b in pairs) / len(pairs)
            if pairs
            else None,
            "cohen_kappa": cohen_kappa(pairs),
        }
    inventory = final_inventory(args.dataset_root)
    report = {
        "schema_version": "vhd-annotation-audit-1.0",
        "review_policy": (
            "Two independent reviews per candidate question; agreement is evaluated only "
            "for task-applicable labels, and disagreements are resolved in the recorded queue."
        ),
        "double_reviewed_questions": sum(item["double_reviewed_questions"] for item in source_reports),
        "adjudication_queue_questions": len(queue),
        "adjudication_rate_percent": 100.0 * len(queue) / inventory["candidate_questions"],
        "overall_agreement": overall_agreement,
        "by_source": source_reports,
        "final_inventory": inventory,
    }
    if report["double_reviewed_questions"] != inventory["candidate_questions"]:
        raise RuntimeError("double-review and final-adjudication inventories differ")
    if inventory["unresolved_question_ids"]:
        raise RuntimeError("final adjudication still contains unresolved questions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "vhd_annotation_audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "vhd_annotation_agreement.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "field", "n_pairs", "percent_agreement", "cohen_kappa"),
        )
        writer.writeheader()
        scopes = [("overall", overall_agreement)] + [
            (item["source"], item["agreement"]) for item in source_reports
        ]
        for scope, values in scopes:
            for field in LABEL_FIELDS:
                writer.writerow({"scope": scope, "field": field, **values[field]})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
