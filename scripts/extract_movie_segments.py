import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "missing_files_v1-2_test" / "movie_segments"

SEGMENTS = [
    {
        "source": ROOT / "data" / "\u5f02\u5f62.mp4",
        "start": "01:39:00",
        "duration": "00:02:00",
        "output": "alien_013900_014100.mp4",
    },
    {
        "source": ROOT / "data" / "\u666e\u7f57\u7c73\u4fee\u65af.mp4",
        "start": "00:01:15",
        "duration": "00:02:00",
        "output": "prometheus_000115_000315.mp4",
    },
    {
        "source": ROOT / "data" / "\u53f2\u5bc6\u65af\u592b\u5987.mp4",
        "start": "00:46:55",
        "duration": "00:00:07",
        "output": "mr_mrs_smith_004655_004702.mp4",
    },
    {
        "source": ROOT / "data" / "\u661f\u9645\u8ff7\u822a\u6697\u9ed1\u65e0\u754c.mp4",
        "start": "01:54:48",
        "duration": "00:00:12",
        "output": "star_trek_into_darkness_015448_015500.mp4",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for segment in SEGMENTS:
        output_path = OUT_DIR / segment["output"]
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            segment["start"],
            "-i",
            str(segment["source"]),
            "-t",
            segment["duration"],
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
