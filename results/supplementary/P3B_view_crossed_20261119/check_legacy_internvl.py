"""CPU checkpoint round-trip for the legacy InternVL loader; no experiment data."""
import copy,json,shutil,tempfile
from pathlib import Path
import torch,transformers
from transformers import AutoConfig,AutoModel,AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'P3B_original_panel_20261119/models/internvl14'
config=AutoConfig.from_pretrained(source,trust_remote_code=True,local_files_only=True)
cls=get_class_from_dynamic_module(config.auto_map['AutoModel'],str(source),local_files_only=True)
config.force_image_size=28
v=config.vision_config;v.image_size=28;v.hidden_size=32;v.intermediate_size=64;v.num_hidden_layers=1;v.num_attention_heads=4
l=config.llm_config;l.hidden_size=32;l.intermediate_size=64;l.num_hidden_layers=1;l.num_attention_heads=4;l.num_key_value_heads=2;l.head_dim=8;l.vocab_size=64;l.layer_types=["full_attention"];l.eos_token_id=2;l.bos_token_id=1;l.pad_token_id=0
model=cls(config,use_flash_attn=False).eval()
fixture=Path(tempfile.mkdtemp(prefix='internvl_loader_fixture_',dir=ROOT))
model.save_pretrained(fixture)
for f in source.glob('*.py'):shutil.copyfile(f,fixture/f.name)
loaded,info=AutoModel.from_pretrained(fixture,local_files_only=True,trust_remote_code=True,dtype=torch.float32,use_flash_attn=False,output_loading_info=True)
assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
for k,v in model.state_dict().items():torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
loaded.img_context_token_id=5
with torch.inference_mode():
    generated=loaded.generate(pixel_values=torch.zeros(1,3,28,28),input_ids=torch.tensor([[1,5,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_logits=True)
assert len(generated.logits)==1 and generated.logits[0].shape[-1]==64
tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=True,use_fast=False)
result={'passed':True,'fixture_is_not_an_experiment_model':True,'fixture_path':str(fixture),'transformers':transformers.__version__,'tokenizer_class':type(tokenizer).__name__,'torch':torch.__version__,'loading_info':info}
(ROOT/'LEGACY_INTERNVL_COMPATIBILITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
