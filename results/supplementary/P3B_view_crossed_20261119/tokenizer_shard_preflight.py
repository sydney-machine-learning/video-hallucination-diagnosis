import argparse,json,os
from pathlib import Path
from transformers import AutoProcessor,AutoTokenizer
from tokenizer_guard import validate
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
directory=ROOT/'models'/args.model
if not directory.exists():directory=ROOT.parent/'P3B_original_panel_20261119/models'/args.model
if args.model=='internvl14':tokenizer=AutoTokenizer.from_pretrained(directory,trust_remote_code=True,local_files_only=True,use_fast=False)
else:tokenizer=AutoProcessor.from_pretrained(directory,trust_remote_code=True,local_files_only=True).tokenizer
rows=[json.loads(s) for s in (ROOT/'full_manifest.jsonl').read_text().splitlines()]
receipt=validate(tokenizer,directory,[r['question']+'\n'+'\n'.join(r['candidates']) for r in rows])
Path(os.environ['TOKENIZER_PREFLIGHT_PATH']).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
