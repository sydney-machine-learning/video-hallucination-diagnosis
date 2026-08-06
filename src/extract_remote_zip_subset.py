"""Extract selected MP4 members from an HTTP range-addressable ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests


class HttpRangeReader(io.RawIOBase):
    """Small seekable reader backed by strict HTTP byte-range requests."""

    def __init__(self, url: str, block_size: int):
        self.url = url
        self.block_size = block_size
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise IOError("remote archive does not advertise byte ranges")
        self.position = 0
        self.cache_start = 0
        self.cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _fetch(self, minimum: int) -> None:
        if self.position >= self.size:
            self.cache_start = self.size
            self.cache = b""
            return
        count = max(self.block_size, minimum)
        end = min(self.size - 1, self.position + count - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
            timeout=180,
        )
        if response.status_code != 206:
            raise IOError(f"strict range request returned HTTP {response.status_code}")
        expected = end - self.position + 1
        if len(response.content) != expected:
            raise IOError(f"short range response: expected {expected}, got {len(response.content)}")
        self.cache_start = self.position
        self.cache = response.content

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        remaining = min(size, self.size - self.position)
        chunks = []
        while remaining:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(remaining)
                cache_end = self.cache_start + len(self.cache)
            offset = self.position - self.cache_start
            take = min(remaining, cache_end - self.position)
            chunks.append(self.cache[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--block-size-mib", type=int, default=8)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    selected = {
        line.strip() for line in args.selected_ids.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if len(selected) != 500:
        raise ValueError(f"expected 500 selected video IDs, got {len(selected)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    remote = HttpRangeReader(args.url, args.block_size_mib * 1024 * 1024)
    with remote, zipfile.ZipFile(remote) as archive:
        matched: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                continue
            video_id = Path(info.filename).stem
            if video_id in selected:
                if video_id in matched:
                    raise ValueError(f"duplicate archive member for {video_id}")
                matched[video_id] = info
        missing = sorted(selected.difference(matched))
        if missing:
            raise ValueError(f"selected archive members missing: {missing[:10]}")
        if args.list_only:
            print(
                "PERCEPTION_TEST_REMOTE_ZIP_INDEX_PASS "
                f"videos={len(matched)} "
                f"compressed_bytes={sum(info.compress_size for info in matched.values())} "
                f"uncompressed_bytes={sum(info.file_size for info in matched.values())}"
            )
            return 0

        records = []
        ordered = sorted(matched.items(), key=lambda item: item[1].header_offset)
        for index, (video_id, info) in enumerate(ordered, 1):
            output = args.output_dir / f"{video_id}.mp4"
            if output.is_file() and output.stat().st_size == info.file_size:
                status = "reused"
            else:
                temporary = output.with_suffix(".mp4.partial")
                temporary.unlink(missing_ok=True)
                with archive.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"short extraction for {video_id}")
                temporary.replace(output)
                status = "extracted"
            records.append(
                {
                    "video_id": video_id,
                    "archive_member": info.filename,
                    "archive_header_offset": info.header_offset,
                    "compressed_size": info.compress_size,
                    "size": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            )
            print(
                f"[{index}/500] {video_id} {status} bytes={output.stat().st_size}",
                flush=True,
            )
    inventory = {
        "schema_version": "trace-rc-perception-test-valid500-media-inventory-1.0",
        "archive_url": args.url,
        "selected_ids_sha256": sha256_file(args.selected_ids),
        "videos": len(records),
        "total_bytes": sum(row["size"] for row in records),
        "elapsed_seconds": time.time() - started,
        "records": sorted(records, key=lambda row: row["video_id"]),
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PERCEPTION_TEST_MEDIA_PASS videos={len(records)} bytes={inventory['total_bytes']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
