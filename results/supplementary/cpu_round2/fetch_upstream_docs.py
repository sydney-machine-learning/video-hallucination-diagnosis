import json,urllib.request,concurrent.futures,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent/'D_sources'
urls={
'internvl3_report.html':'https://arxiv.org/html/2504.10479v1',
'videochat_training_inventory.json':'https://huggingface.co/api/datasets/OpenGVLab/VideoChat-Flash-Training-Data',
'internvl_data_card.md':'https://huggingface.co/datasets/OpenGVLab/InternVL-Data/raw/main/README.md',
'internvl_debug_sft.json':'https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat_gpt_oss/shell/data/debug_sft.json',
'internvl_debug_mpo.json':'https://raw.githubusercontent.com/OpenGVLab/InternVL/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat_gpt_oss/shell/data/debug_mpo.json',
}
def fetch(item):
 name,url=item
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-document-audit'}),timeout=40) as r:data=r.read()
  (root/name).write_bytes(data)
  return dict(file=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(file=name,url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(4) as p:out=list(p.map(fetch,urls.items()))
(root/'UPSTREAM_SOURCES.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
