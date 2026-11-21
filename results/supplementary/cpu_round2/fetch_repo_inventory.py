from pathlib import Path
import urllib.request,json,concurrent.futures
root=Path(__file__).resolve().parent/'D_sources'
repos=['QwenLM/Qwen3-VL','OpenGVLab/InternVL','EvolvingLMMs-Lab/LLaVA-OneVision-2']
def fetch(repo):
 url=f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1'
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-document-audit'}),timeout=40) as r:data=r.read()
 (root/(repo.split('/')[-1]+'_tree.json')).write_bytes(data)
 tree=json.loads(data)
 paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob' and any(k in x['path'].lower() for k in ['dataset','data_config','data.json','3_5','3.5','training','train_config']) and x['path'].endswith(('.json','.yaml','.yml','.md','.sh'))]
 return dict(repo=repo,sha=tree.get('sha'),truncated=tree.get('truncated'),paths=paths)
with concurrent.futures.ThreadPoolExecutor(3) as p:out=list(p.map(fetch,repos))
(root/'repo_inventory.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
