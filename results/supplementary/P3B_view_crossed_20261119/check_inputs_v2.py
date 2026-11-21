"""CPU-only checks of exact archived sampling and multi-frame processing."""
import ast,json,math
from pathlib import Path
import av,numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,AutoTokenizer
from tokenizer_guard import validate as validate_tokenizer
from native_sampling import indices,FRAME_COUNTS
from run_peer import SYSTEM,digest
import archived_internvl_adapter as archived
ROOT=Path(__file__).resolve().parent

def main():
    row=json.loads((ROOT/'full_manifest.jsonl').read_text().splitlines()[0])
    with av.open(row['video_path']) as container:
        stream=container.streams.video[0];total=int(stream.frames);fps=float(stream.average_rate)
        duration=float(stream.duration*stream.time_base)
        views={p:indices(row,total,duration,p) for p in FRAME_COUNTS}
        wanted={i for v in views.values() for i in v};decoded={}
        for i,frame in enumerate(container.decode(stream)):
            if i in wanted:decoded[i]=frame.to_ndarray(format='rgb24')
            if i>=max(wanted):break
    assert set(decoded)==wanted
    result={'passed':True,'labels_read':False,'video_sha256':digest(row['video_path']),'views':{},'models':{}}
    for p,selected in views.items():
        assert len(selected)==FRAME_COUNTS[p]
        assert selected==sorted(selected,reverse=p==6)
        result['views'][p]=selected
    for model in ['qwen8','llava8','internvl14']:
        directory=(ROOT/'models'/model if model=='qwen8' else ROOT.parent/'P3B_original_panel_20261119/models'/model)
        processor=AutoProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=True) if model!='internvl14' else None
        tokenizer=processor.tokenizer if processor else AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=True,use_fast=False)
        result.setdefault("tokenizer_validation",{})[model]=validate_tokenizer(tokenizer,directory,[row["question"]])
        shapes={}
        for p,selected in views.items():
            images=[Image.fromarray(decoded[i]) for i in selected]
            if model=='internvl14':
                inputs={'pixel_values':archived.preprocess_frames(images,torch.bfloat16,'cpu')}
                assert list(inputs['pixel_values'].shape)==[len(selected),3,448,448]
            else:
                content=([{'type':'video'}] if model=='qwen8' else [{'type':'image'} for _ in images])+[{'type':'text','text':'Which option is correct? A. Moves. B. Stops.'}]
                rendered=processor.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
                if model=='qwen8':
                    inputs=processor(text=[rendered],videos=[np.stack([decoded[i] for i in selected])],video_metadata=[{'total_num_frames':total,'fps':fps,'frames_indices':selected}],do_sample_frames=False,cap_pixels_per_frame=False,return_tensors='pt')
                    assert int(inputs['video_grid_thw'][0,0])*2==len(selected)
                else:
                    inputs=processor(text=[rendered],images=images,padding=True,return_tensors='pt')
                    assert list(inputs['image_grid_thw'].shape)==[len(selected),3]
            shapes[p]={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
            del inputs
        result['models'][model]=shapes
    (ROOT/'CPU_INPUT_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
