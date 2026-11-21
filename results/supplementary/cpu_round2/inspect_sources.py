from pathlib import Path
import json,re,hashlib,csv
from collections import defaultdict
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parent
P5=ROOT.parent/'P5_vhd_transfer_20261119'
hits=[]
for p in (ROOT/'D_sources').glob('*.html'):
 soup=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
 for e in soup(['script','style']):e.decompose()
 lines=[re.sub(r'\s+',' ',x.get_text(' ',strip=True)) for x in soup.select('h1,h2,h3,h4,p,table')]
 p.with_suffix('.txt').write_text('\n'.join(lines),encoding='utf-8')
for p in sorted((ROOT/'D_sources').glob('*')):
 if p.suffix not in ['.txt','.md']:continue
 lines=p.read_text(encoding='utf-8').splitlines()
 matches=[dict(line=i+1,text=s) for i,s in enumerate(lines) if re.search(r'perception[\s_-]*test',s,re.I)]
 hits.append(dict(file=p.name,matches=matches))
(ROOT/'D_document_search.json').write_text(json.dumps(hits,indent=2,ensure_ascii=False),encoding='utf-8')
sources=json.loads((P5/'source_manifest.json').read_text())
rows=[]
for item in sources:
 p=Path(item['path'])
 if p.suffix!='.jsonl':continue
 counts=defaultdict(int);first={}
 with p.open(encoding='utf-8') as f:
  for n,line in enumerate(f,1):
   r=json.loads(line);key=(r.get('dataset',''),r.get('model_id'),r.get('view_id'),r.get('frame_budget'),r.get('generator',r.get('view_generator')))
   counts[key]+=1;first.setdefault(key,n)
 for key,count in counts.items():
  rows.append(dict(dataset=key[0],model_id=key[1],view=key[2],frame_budget=key[3],generator=key[4],rows=count,path=str(p),first_line=first[key]))
with (ROOT/'C_raw_view_audit.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Perception Test matches:',json.dumps(hits,ensure_ascii=False)[:12000])
print('Raw view combinations',len(rows))
