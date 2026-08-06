"""Freeze code, model, split, and gate hashes before HERBench inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "METHOD_SPEC.md",
    "src/build_herbench_manifest.py",
    "src/evaluate_v2_target_pilot.py",
    "src/analyze_herbench_trace_rc.py",
    "katana/herbench_multimodel_runner.py",
    "katana/herbench_persistent_worker.py",
    "katana/prepare_herbench_assets.pbs",
    "katana/run_herbench_consolidated.pbs",
    "input/herbench_lite_v2_inference.jsonl",
    "input/preinference_split_lock.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.route_root.resolve()
    prelock = json.loads((root / "input/preinference_split_lock.json").read_text(encoding="utf-8"))
    development_gate = args.development_gate.resolve()
    development = json.loads(development_gate.read_text(encoding="utf-8"))
    if not development["gate_pass"] or development["selected_pilot_fraction"] != 0.5:
        raise RuntimeError("Video-MME-v2 development gate does not authorize freezing")
    hashes = {}
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = digest(path)
    seal = {
        "schema_version": "trace-rc-herbench-preinference-seal-1.0",
        "created_date": "2026-08-02",
        "dataset": "HERBench Lite-v2",
        "dataset_revision": prelock["dataset_revision"],
        "manifest_sha256": prelock["manifest_sha256"],
        "pilot_labels_sha256": prelock["pilot_labels_sha256"],
        "sealed_test_labels_sha256": prelock["sealed_test_labels_sha256"],
        "split": {
            "grouping_unit": "source video",
            "pilot_fraction": 0.5,
            "pilot_videos": 34,
            "test_videos": 34,
            "pilot_questions": prelock["pilot_questions"],
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
            "grouped_bootstrap_iterations": 2000,
            "strictly_positive_adjusted_lower_bound_required": True,
            "rejection_action": "pointwise raw-confidence fallback",
        },
        "confirmation_gate": {
            "macro_relative_aurc_gain_percent_at_least": 1.0,
            "video_grouped_95ci_lower_above_zero": True,
            "worst_model_gain_percent_at_least": -1.0,
            "at_least_one_environment_promoted": True,
            "final_bootstrap_iterations": 10000,
            "bootstrap_seed": 20260802,
        },
        "development_authorization": {
            "path": development_gate.relative_to(root).as_posix(),
            "sha256": digest(development_gate),
            "candidate_library": development["candidate_library"],
            "selected_pilot_fraction": development["selected_pilot_fraction"],
            "mean_test_gain_threshold": development["development_gate_thresholds"]["mean_test_gain_percent_at_least"],
        },
        "execution": {
            "single_gpu_allocation": True,
            "gpus": 3,
            "gpu_model": "H200",
            "walltime": "11:58:00",
            "checkpointed": True,
            "pilot_and_final_analysis_inside_same_allocation": True,
        },
        "source_hashes": hashes,
        "test_labels_must_not_be_passed_to_pilot_phase": True,
        "failed_gate_must_be_retained": True,
    }
    args.output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
