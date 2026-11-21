"""Stream the pinned HERBench Lite archive into job-local storage, keeping only this shard's videos."""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REV = '30b42e97180f2c7c0fdc3f2aed412ae04f0366ca'
BASE = f'https://huggingface.co/datasets/DanBenAmi/HERBench/resolve/{REV}/videos/'
PARTS = [f'videos.tar.part.{i:02d}' for i in range(4)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--shard-index', type=int, required=True)
    a = p.parse_args()
    dest = Path(os.environ['VIDEO_ROOT']); dest.mkdir(parents=True, exist_ok=True)
    assert dest.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve())
    plan = json.loads((ROOT / f'SHARD_PLAN_{a.model}.json').read_text())
    vids = set(plan['shards'][a.shard_index]['video_ids'])
    rows = [json.loads(s) for s in (ROOT / 'full_manifest.jsonl').read_text().splitlines()]
    members = sorted({r['video_path'] for r in rows if r['video_id'] in vids})
    if all((dest / m).is_file() and (dest / m).stat().st_size > 0 for m in members):
        print('VIDEOS_ALREADY_PRESENT', len(members), flush=True); return
    with urllib.request.urlopen(BASE + 'videos.tar.checksums.txt', timeout=120) as r:
        sums = {line.split()[1]: line.split()[0] for line in r.read().decode().splitlines() if line.strip()}
    assert all(m.startswith('videos/') for m in members); (dest / 'videos').mkdir(exist_ok=True)
    # Archive members are stored without the videos/ prefix used in the manifest.
    listfile = dest / 'members.txt'; listfile.write_text(''.join(m[len('videos/'):] + '\n' for m in members))
    tar = subprocess.Popen(['tar', '-x', '-f', '-', '-C', str(dest / 'videos'), '-T', str(listfile)], stdin=subprocess.PIPE)
    started = time.time(); got = {}
    for part in PARTS:
        h = hashlib.sha256(); n = 0
        for attempt in range(6):
            try:
                req = urllib.request.Request(BASE + part + '?download=true', headers={'Range': f'bytes={n}-'} if n else {})
                with urllib.request.urlopen(req, timeout=180) as r:
                    assert n == 0 or r.status == 206
                    while True:
                        block = r.read(16 * 1024 * 1024)
                        if not block: break
                        h.update(block); n += len(block); tar.stdin.write(block)
                break
            except (OSError, AssertionError) as e:
                if attempt == 5: raise
                print('RETRY', part, n, repr(e), flush=True); time.sleep(10 * (attempt + 1))
        got[part] = h.hexdigest()
        assert got[part] == sums[part], (part, got[part], sums[part])
        print('PART_VERIFIED', part, n, round(time.time() - started), flush=True)
    tar.stdin.close(); assert tar.wait() == 0
    missing = [m for m in members if not (dest / m).is_file() or (dest / m).stat().st_size == 0]
    assert not missing, missing[:5]
    receipt = {'shard_model': a.model, 'shard_index': a.shard_index, 'videos': len(members), 'parts_sha256': got,
               'seconds': round(time.time() - started), 'job': os.environ.get('PBS_JOBID')}
    (dest / 'FETCH_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('VIDEOS_READY', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
