"""Check pinned tokenizer assets, including checkpoints without tokenizer.json."""
import hashlib,json
from pathlib import Path
def validate(tokenizer,directory,samples):
    directory=Path(directory)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    backend=json.loads(tokenizer.backend_tokenizer.to_str())
    if (directory/'tokenizer.json').exists():
        from tokenizers import Tokenizer
        reference=Tokenizer.from_file(str(directory/'tokenizer.json'))
        same_structure=backend['pre_tokenizer']==json.loads((directory/'tokenizer.json').read_text())['pre_tokenizer']
        actual=tokenizer(samples,add_special_tokens=False)['input_ids'];expected=[x.ids for x in reference.encode_batch(samples,add_special_tokens=False)]
        mismatches=sum(a!=b for a,b in zip(actual,expected));assert mismatches==0,('pinned_token_id_mismatches',mismatches,len(samples))
        receipt={'basis':'pinned tokenizer.json token-ID agreement','pretokenizer_serialization_identical':same_structure,'questions_checked':len(samples),'assets':{'tokenizer.json':sha(directory/'tokenizer.json')}}
    else:
        pinned=json.loads((directory/'vocab.json').read_text())
        assert backend['model']['vocab']==pinned
        expected=[line.split() for line in (directory/'merges.txt').read_text().splitlines() if line.strip() and not line.startswith('#version:')]
        actual=[x.split() if isinstance(x,str) else x for x in backend['model']['merges']]
        assert actual==expected
        receipt={'basis':'pinned vocab.json and merges.txt; runtime Qwen2 pretokenizer','assets':{n:sha(directory/n) for n in ['vocab.json','merges.txt','tokenizer_config.json']},'runtime_pretokenizer':backend['pre_tokenizer'],'vocabulary_size':len(pinned)}
    receipt['passed']=True
    return receipt
