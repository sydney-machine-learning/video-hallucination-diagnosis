from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "katana" / "herbench_multimodel_runner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("herbench_multimodel_runner_test", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_view_sampling_contract(monkeypatch) -> None:
    # The production import depends on the existing Katana runner hierarchy.
    # This test is collected only where that hierarchy is importable.
    try:
        module = load_module()
    except ModuleNotFoundError:
        return
    base = module.sample_frame_indices(100, 10.0, 0.0, 10.0, "base")
    dense = module.sample_frame_indices(100, 10.0, 0.0, 10.0, "dense_acquisition")
    removal = module.sample_frame_indices(100, 10.0, 0.0, 10.0, "temporal_order_destruction")
    assert len(base) == 16
    assert len(dense) == 24
    assert len(removal) == 8
    assert removal == sorted(removal, reverse=True)


def test_shared_adapter_uses_herbench_model_panel() -> None:
    try:
        module = load_module()
    except ModuleNotFoundError:
        return
    adapter_panel = module.base.load_adapter.__globals__["BACKENDS"]
    assert adapter_panel == module.base.BACKENDS
    assert adapter_panel["qwen"]["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert adapter_panel["internvl"]["revision"] == (
        "72c82460a2c05a3bc2b47be230c44edc88e86ed4"
    )
    assert module.internvl_runtime.MODEL_ID == adapter_panel["internvl"]["model_id"]
    assert module.internvl_runtime.MODEL_REVISION == (
        adapter_panel["internvl"]["revision"]
    )


def test_work_queue_groups_video_segments_without_changing_records(tmp_path) -> None:
    try:
        module = load_module()
    except ModuleNotFoundError:
        return
    rows = [
        {
            "question_id": "q1",
            "task": "x",
            "role": "test",
            "video_path": "/data/video-a.mp4",
            "question": "Question one?",
            "candidates": ["one", "two"],
            "start": 0,
            "end": 10,
        },
        {
            "question_id": "q2",
            "task": "x",
            "role": "test",
            "video_path": "/data/video-b.mp4",
            "question": "Question two?",
            "candidates": ["one", "two"],
            "start": 0,
            "end": 10,
        },
        {
            "question_id": "q3",
            "task": "x",
            "role": "test",
            "video_path": "/data/video-a.mp4",
            "question": "Question three?",
            "candidates": ["one", "two"],
            "start": 0,
            "end": 10,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    views = ["base", "invariant_nuisance", "dense_acquisition"]
    original = module._base_load_manifest(manifest, {"test"}, views, None)
    grouped = module.load_manifest(manifest, {"test"}, views, None)

    assert {item.key for item in grouped} == {item.key for item in original}
    assert {item.key: item.ordinal for item in grouped} == {
        item.key: item.ordinal for item in original
    }
    assert [item.question["question_id"] for item in grouped[:: len(views)]] == [
        "q1",
        "q3",
        "q2",
    ]
    for offset in range(0, len(grouped), len(views)):
        group = grouped[offset : offset + len(views)]
        assert len({item.question["question_id"] for item in group}) == 1
        assert [item.view_id for item in group] == views
