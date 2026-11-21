"""Read-only Katana inventory and byte-exact copies; no jobs or remote writes."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SSH = ["ssh.exe", "-i", str(Path.home()/".ssh/codex_katana_ed25519"),
       "-o", "HostKeyAlias=katana.restech.unsw.edu.au", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=15", "z5611125@129.94.146.61"]
REMOTE = r'''
import os,json,hashlib,platform
from pathlib import Path
roots=[Path('/home/z5611125'),Path('/srv/scratch/z5611125')]
found=[]; errors=[]
for root in roots:
 for current,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
  dirs[:]=[d for d in dirs if not d.startswith('.') and d not in
           {'envs','venv','site-packages','node_modules','huggingface','wandb','__pycache__'}]
  for name in files:
   p=Path(current)/name
   if ('predictions' in name and p.suffix in {'.jsonl','.csv'}) or \
      ('features' in name and p.suffix=='.csv') or \
      (any(w in name.lower() for w in ('herbench','perception','mvbench')) and
       any(name.endswith(e) for e in ('.zip','.tar','.tar.gz','.tgz','.7z'))):
    found.append({'path':str(p),'bytes':p.stat().st_size})
print(json.dumps({'roots':[str(p) for p in roots],'pruned_directories':
 ['hidden directories','envs','venv','site-packages','node_modules','huggingface','wandb','__pycache__'],
 'files':found,'errors':errors,'platform':platform.platform(),'python':platform.python_version()},indent=2))
'''

def main():
    assert not (OUT/"remote_inventory.json").exists(), "Do not overwrite an audit run"
    run = subprocess.run(SSH+["python3 -"], input=REMOTE.encode(), capture_output=True, timeout=240)
    evidence = {"utc":datetime.now(timezone.utc).isoformat(),"returncode":run.returncode,
                "stderr":run.stderr.decode(errors="replace"),"command":SSH+["python3 -"]}
    if run.returncode == 0:
        evidence.update(json.loads(run.stdout))
    else:
        evidence["stdout"] = run.stdout.decode(errors="replace")
    (OUT/"remote_inventory.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    print("Remote inventory:",run.returncode, len(evidence.get("files",[])),flush=True)
    transfers=[]
    sources = [(f"/home/z5611125/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/{m}_predictions.jsonl",
                f"inputs/mvbench/{m}_predictions.jsonl") for m in ("qwen","internvl","llava")]
    sources += [("/home/z5611125/sicr_route/katana/nextgqa_llava_ov2_runner.py", "inputs/remote_code/nextgqa_llava_ov2_runner.py")]
    # Copy target archives only if a matching file is actually discovered.
    for rec in evidence.get("files",[]):
        p=rec["path"]
        if any(t in p.lower() for t in ("herbench","perception")) and p.endswith(".jsonl") and "predictions" in p:
            sources.append((p,"inputs/recovered/"+p.lstrip("/")))
    for remote,relative in sources:
        local=OUT/relative
        local.parent.mkdir(parents=True,exist_ok=True)
        result=subprocess.run(SSH+["cat -- '"+remote+"'"],capture_output=True,timeout=240)
        item={"remote":remote,"local":relative,"returncode":result.returncode,
              "stderr":result.stderr.decode(errors="replace")}
        if result.returncode==0:
            with local.open("xb") as stream: stream.write(result.stdout)
            item.update(bytes=len(result.stdout),sha256=hashlib.sha256(result.stdout).hexdigest())
        transfers.append(item)
        print("Copy:",relative,item["returncode"],item.get("bytes",0),flush=True)
    (OUT/"remote_transfers.json").write_text(json.dumps(transfers,indent=2),encoding="utf-8")

if __name__=="__main__": main()
