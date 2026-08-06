"""Freeze the Perception Test split, code, model panel, and gates before inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "METHOD_SPEC.md",
    "src/build_perception_test_manifest.py",
    "src/extract_remote_zip_subset.py",
    "src/freeze_perception_protocol.py",
    "src/verify_perception_media.py",
    "src/verify_perception_seal.py",
    "src/evaluate_v2_target_pilot.py",
    "src/analyze_herbench_trace_rc.py",
    "src/analyze_perception_trace_rc.py",
    "katana/herbench_multimodel_runner.py",
    "katana/perception_multimodel_runner.py",
    "katana/perception_head_half_runner.py",
    "katana/perception_tail_half_runner.py",
    "katana/merge_perception_halves.py",
    "katana/prepare_perception_valid500.pbs",
    "katana/run_perception_valid500.pbs",
    "input/perception_test/frozen/perception_test_valid500_inference.jsonl",
    "input/perception_test/frozen/preinference_split_lock.json",
    "input/perception_test/frozen/selected_video_ids.txt",
    "PREINFERENCE_AMENDMENT_20260803_POSIX_PATH.json",
    "PREINFERENCE_AMENDMENT_20260803_FOUR_GPU.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--herbench-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.route_root.resolve()
    prelock_path = root / "input/perception_test/frozen/preinference_split_lock.json"
    prelock = json.loads(prelock_path.read_text(encoding="utf-8"))
    parent_gate_path = args.herbench_gate.resolve()
    parent_gate = json.loads(parent_gate_path.read_text(encoding="utf-8"))
    if not parent_gate["primary_gate_pass"]:
        raise RuntimeError("the fixed TRACE-RC route lacks a positive parent confirmation")
    hashes = {}
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = digest(path)
    seal = {
        "schema_version": "trace-rc-perception-test-preinference-seal-1.0",
        "created_date": "2026-08-03",
        "dataset": "Google DeepMind Perception Test validation, deterministic 500-video subset",
        "dataset_repository": prelock["dataset_repository"],
        "dataset_revision": prelock["dataset_revision"],
        "annotation_sha256": prelock["annotations_sha256"],
        "cut_mapping_sha256": prelock["cut_mapping_sha256"],
        "manifest_sha256": prelock["manifest_sha256"],
        "pilot_labels_sha256": prelock["pilot_labels_sha256"],
        "sealed_test_labels_sha256": prelock["sealed_test_labels_sha256"],
        "selected_video_ids_sha256": prelock["selected_video_ids_sha256"],
        "selection": {
            "frame": prelock["selection_frame"],
            "method": "first 500 video IDs under a salted SHA-256 order",
            "uses_question_text_options_or_answers": False,
            "salt": prelock["selection_salt"],
        },
        "split": {
            "grouping_unit": "source video",
            "method": "salted SHA-256 order",
            "calibration_videos": prelock["calibration_videos"],
            "test_videos": prelock["test_videos"],
            "calibration_questions": prelock["calibration_questions"],
            "test_questions": prelock["test_questions"],
            "salt": prelock["split_salt"],
        },
        "model_panel": {
            "qwen": ["Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"],
            "internvl": ["OpenGVLab/InternVL3_5-14B-Instruct", "72c82460a2c05a3bc2b47be230c44edc88e86ed4"],
            "llava": ["lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", "802f16c8346a062fef7ddf92b2bd64a770a50a1a"],
        },
        "views": {
            "base": 16,
            "invariant_nuisance": 8,
            "dense_acquisition": 24,
            "temporal_order_destruction": 8,
        },
        "promotion": {
            "candidates": ["source_trace_hgb", "target_logistic"],
            "candidate_lower_bound_margins_percent": {
                "source_trace_hgb": 0.5,
                "target_logistic": 0.0,
            },
            "familywise_alpha": 0.05,
            "video_grouped_bootstrap_iterations": 2000,
            "rejection_action": "pointwise raw-confidence fallback",
        },
        "confirmation_gate": {
            "macro_relative_aurc_gain_percent_at_least": 1.0,
            "video_grouped_95ci_lower_above_zero": True,
            "worst_model_gain_percent_at_least": -1.0,
            "at_least_one_environment_promoted": True,
            "final_bootstrap_iterations": 10000,
            "bootstrap_seed": 20260803,
        },
        "route_fixed_before_dataset_inference": True,
        "parent_herbench_confirmation": {
            "path": parent_gate_path.relative_to(root).as_posix(),
            "sha256": digest(parent_gate_path),
            "macro_gain_percent": parent_gate["macro_mean_relative_aurc_reduction_percent"],
        },
        "execution": {
            "single_gpu_pbs_allocation": True,
            "gpus": 4,
            "gpu_model": "H200",
            "walltime": "05:58:00",
            "schedule": "Qwen, InternVL, and two disjoint LLaVA halves run concurrently",
            "analysis_inside_same_allocation": True,
            "checkpointed": True,
        },
        "source_hashes": hashes,
        "test_labels_must_not_be_passed_to_pilot_phase": True,
        "failed_gate_must_be_retained": True,
    }
    args.output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
