"""Pinned, resumable downloads adapted from the successful pilot preparer."""
import argparse,hashlib,json,shutil,time,urllib.request,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS={'qwen8':('Qwen/Qwen3-VL-8B-Instruct','0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),'internvl14':('OpenGVLab/InternVL3_5-14B-Instruct','72c82460a2c05a3bc2b47be230c44edc88e86ed4'),
        'llava8':('lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct','802f16c8346a062fef7ddf92b2bd64a770a50a1a')}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=MODELS);parser.add_argument('--configs-only',action='store_true');parser.add_argument('--destination',type=Path);args=parser.parse_args()
    model,revision=MODELS[args.model];dest=args.destination or ROOT/'models'/args.model
    if args.destination:assert dest.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve())
    dest.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true',timeout=90) as response: info=json.load(response)
    assert info['sha']==revision
    (ROOT/(args.model+'_metadata.json')).write_text(json.dumps(info,indent=2)+'\n')
    chosen=[r for r in info['siblings'] if '/' not in r['rfilename'] and (r['rfilename'].endswith(('.json','.safetensors','.jinja','.py','.model','.txt')) or r['rfilename']=='merges.txt')]
    if args.configs_only:chosen=[r for r in chosen if not r['rfilename'].endswith('.safetensors')]
    needed=sum(r['size'] for r in chosen if not (dest/r['rfilename']).exists())
    assert shutil.disk_usage(dest).free>needed+1024**3,'Insufficient scratch headroom'
    def fetch(r):
        target=dest/r['rfilename'];part=target.with_name(target.name+'.partial');expected=r.get('lfs',{}).get('sha256')
        if not target.exists():
            for attempt in range(5):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    request=urllib.request.Request(f'https://huggingface.co/{model}/resolve/{revision}/{r["rfilename"]}',headers={'Range':f'bytes={offset}-'} if offset else {})
                    with urllib.request.urlopen(request,timeout=120) as response,part.open('ab' if offset and response.status==206 else 'wb') as out:shutil.copyfileobj(response,out,8*1024*1024)
                    assert part.stat().st_size==r['size']
                    digest=sha(part);assert not expected or digest==expected
                    part.rename(target);break
                except Exception:
                    if attempt==4:raise
                    time.sleep(5*(attempt+1))
        digest=sha(target);assert target.stat().st_size==r['size'] and (not expected or digest==expected)
        print('VERIFIED',target.name,target.stat().st_size,flush=True)
        return {'name':target.name,'bytes':target.stat().st_size,'sha256':digest,'lfs_sha256':expected}
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,chosen))
    name=args.model+('_CONFIGS_READY.json' if args.configs_only else '_MODEL_READY.json')
    (ROOT/name).write_text(json.dumps({'model_id':model,'revision':revision,'records':records,'model_directory':str(dest),'ephemeral':bool(args.destination)},indent=2)+'\n')
    print(name,sum(r['bytes'] for r in records),flush=True)
if __name__=='__main__':main()
