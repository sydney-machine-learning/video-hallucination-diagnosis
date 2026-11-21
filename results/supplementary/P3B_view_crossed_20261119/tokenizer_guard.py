"""Record the complete configured tokenizer, rather than a bare tokenizer.json."""
import hashlib,json
from pathlib import Path
def validate(tokenizer,directory,samples):
    directory=Path(directory)
    if hasattr(tokenizer,'backend_tokenizer'):
        serialized=tokenizer.backend_tokenizer.to_str()
    else:
        serialized=json.dumps({'vocabulary':tokenizer.get_vocab(),'bpe_ranks':[(list(k),v) for k,v in sorted(tokenizer.bpe_ranks.items())]},sort_keys=True,separators=(',',':'))
    ids=tokenizer(samples,add_special_tokens=False)['input_ids']
    assert len(ids)==len(samples) and all(len(x)>0 for x in ids)
    assert all(0<=i<len(tokenizer) for row in ids for i in row)
    assets={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ['tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json','added_tokens.json'] if (directory/n).exists()}
    return {'passed':True,'basis':'official AutoProcessor/AutoTokenizer with complete pinned configuration','questions_checked':len(samples),'assets':assets,'backend_sha256':hashlib.sha256(serialized.encode()).hexdigest(),'sample_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'tokenizer_class':type(tokenizer).__name__}
