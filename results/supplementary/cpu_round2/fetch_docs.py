from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parent/'D_sources'
URLS={
 'qwen_report':'https://arxiv.org/html/2511.21631v1',
 'internvl_report':'https://arxiv.org/html/2508.18265v2',
 'llava_report':'https://arxiv.org/html/2605.25979v1',
 'encoder_report':'https://arxiv.org/html/2602.08683v1',
 'qwen_card':'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/README.md',
 'internvl_card':'https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct/raw/main/README.md',
 'llava_card':'https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct/raw/main/README.md',
 'qwen_repo':'https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md',
 'internvl_repo':'https://raw.githubusercontent.com/OpenGVLab/InternVL/main/README.md',
 'llava_repo':'https://raw.githubusercontent.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/main/README.md',
 'perception_repo':'https://raw.githubusercontent.com/google-deepmind/perception_test/main/README.md',
}
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=50) as r:data=r.read();final=r.url
  p=ROOT/(name+('.html' if 'report' in name else '.md'));p.write_bytes(data)
  return dict(name=name,url=url,final_url=final,path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(name=name,url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(4) as pool:records=list(pool.map(fetch,URLS.items()))
(ROOT/'SOURCES.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in x.items() if k in ['name','bytes','error']} for x in records]))
