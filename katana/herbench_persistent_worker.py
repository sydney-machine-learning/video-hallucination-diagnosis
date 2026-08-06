#!/usr/bin/env python3
"""Keep one HERBench model loaded while processing a resumable command."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from argparse import Namespace
from pathlib import Path

import herbench_multimodel_runner as her


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def command_args(backend: str, payload: dict[str, object], stop_file: Path) -> Namespace:
    locked = her.base.BACKENDS[backend]
    return Namespace(
        backend=backend,
        manifest=Path(str(payload["manifest"])),
        roles=("calibration", "test"),
        views=tuple(her.base.VIEW_SPECS),
        output=Path(str(payload["output"])),
        state_dir=Path(str(payload["state_dir"])),
        output_prefix=str(payload["output_prefix"]),
        model_id=locked["model_id"],
        revision=locked["revision"],
        shard_size=25,
        max_questions=None,
        checkpoint_every=10,
        stop_file=stop_file,
        resume=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(her.base.BACKENDS), required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    root = args.command_root / args.backend
    root.mkdir(parents=True, exist_ok=True)
    locked = her.base.BACKENDS[args.backend]
    print(f"WORKER_MODEL_LOAD_START {args.backend} {locked['model_id']}", flush=True)
    adapter = her.base.load_adapter(args.backend, locked["model_id"], locked["revision"])
    atomic_text(root / "READY", f"pid={os.getpid()}\n")
    print(f"WORKER_READY {args.backend}", flush=True)
    while True:
        if (args.command_root / "SHUTDOWN").exists():
            return 0
        commands = sorted(root.glob("*.command.json"))
        pending = [
            path
            for path in commands
            if not path.with_suffix(".done.json").exists()
            and not path.with_suffix(".failed.json").exists()
        ]
        if not pending:
            if args.stop_file.exists():
                return 75
            time.sleep(args.poll_seconds)
            continue
        command = pending[0]
        payload = json.loads(command.read_text(encoding="utf-8"))
        started = time.time()
        try:
            rc = her.base.run(command_args(args.backend, payload, args.stop_file), adapter)
            if rc != 0:
                raise RuntimeError(f"runner returned {rc}")
            atomic_text(
                command.with_suffix(".done.json"),
                json.dumps(
                    {
                        "backend": args.backend,
                        "status": "ok",
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
            )
        except BaseException as exc:
            atomic_text(
                command.with_suffix(".failed.json"),
                json.dumps(
                    {
                        "backend": args.backend,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        )[-12000:],
                    },
                    indent=2,
                )
                + "\n",
            )
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
