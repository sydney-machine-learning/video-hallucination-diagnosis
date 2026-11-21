import json,urllib.request,concurrent.futures,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent/'D_sources';requests=[]
for repo,paths in {
'QwenLM/Qwen3-VL':['qwen-vl-finetune/README.md','qwen-vl-finetune/qwenvl/data/__init__.py'],
'OpenGVLab/InternVL':['internvl_chat_gpt_oss/shell/internvl3_5_qwen3/internvl3_5_14b_sft.sh','internvl_chat_gpt_oss/shell/internvl3_5_qwen3/internvl3_5_14b_mpo.sh'],
'EvolvingLMMs-Lab/LLaVA-OneVision-2':['configs/README.md','configs/sft_dataset_config.json','aiak_megatron/examples/multimodal/sft_dataset.yaml','aiak_megatron/examples/multimodal/pretrain_dataset.yaml','examples/llava_onevision2/qwen3_8b/mid_training.sh'],
}.items():
 sha=json.loads((root/(repo.split('/')[-1]+'_tree.json')).read_text())['sha']
 for p in paths:requests.append((repo.split('/')[-1]+'__'+p.replace('/','__'),f'https://raw.githubusercontent.com/{repo}/{sha}/{p}',f'https://github.com/{repo}/blob/{sha}/{p}'))
pre=json.loads((root.parent/'PREFLIGHT.json').read_text())
for m in pre['models']:
 mid=m['model_ids'][0];rev=m['revisions'][0]
 requests.append((m['model']+'_archived_card.md',f'https://huggingface.co/{mid}/raw/{rev}/README.md',f'https://huggingface.co/{mid}/blob/{rev}/README.md'))
def fetch(item):
 name,url,source=item
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-document-audit'}),timeout=40) as r:data=r.read()
  (root/name).write_bytes(data)
  return dict(file=name,url=url,source=source,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(file=name,url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(4) as p:out=list(p.map(fetch,requests))
(root/'CONFIG_SOURCES.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
