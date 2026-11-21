"""Label-blind shared evidence inference, adapted from successful reference runner."""
import argparse, hashlib, json, math, os, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SYSTEM='You answer multiple-choice questions about chronologically ordered frames sampled from a real video. Use the visual evidence and physical evolution across frames. Return exactly one option letter and no other text.'
PHASES=(0,2,4,6)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

from native_sampling import indices,VIEW_IDS,FRAME_COUNTS

def parse_answer(text,labels):
    # A reply such as 'B. A woman ...' starts with the option label; the article 'A' must not count.
    lead=re.match(r'\s*\(?(['+''.join(labels)+r'])(?![A-Za-z])',text)
    if lead:return lead.group(1)
    found=set(re.findall(r'(?<![A-Za-z])['+''.join(labels)+r'](?![A-Za-z])',text.strip()))
    return next(iter(found)) if len(found)==1 else None

def append(path,record):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())

def trim_cache(cache,prefix_length):
    current=cache.get_seq_length()
    assert current>=prefix_length
    if current>prefix_length:
        cache.crop(-(current-prefix_length))
    assert cache.get_seq_length()==prefix_length

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-videos',type=int);parser.add_argument('--shard-index',type=int,required=True);parser.add_argument('--model',choices=['qwen8','internvl14','llava8'],required=True);parser.add_argument('--reference-only',action='store_true'); args=parser.parse_args()
    assert args.reference_only, 'Full proposal permits only the validated reference inference path'
    import av,torch,transformers
    from transformers import AutoProcessor,AutoModelForImageTextToText,AutoTokenizer,AutoModel,Qwen3VLForConditionalGeneration
    from PIL import Image
    import archived_internvl_adapter as archived_internvl
    torch.manual_seed(1234);np.random.seed(1234)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 24*1024**3, 'Allocation is too small (possible MIG slice); refusing model load'
    protocol=json.loads((ROOT/'SHARD_RUN_PROTOCOL.json').read_text())
    assert digest(__file__)==protocol['runner_sha256']
    model_dir=Path(os.environ['LOCAL_MODEL_DIR'])
    assert model_dir.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve())
    model_receipt=Path(os.environ['MODEL_RECEIPT_PATH'])
    model_info=json.loads(model_receipt.read_text())
    assert digest(ROOT/'full_manifest.jsonl')==protocol['manifest_sha256']
    assert model_receipt.exists()
    rows=[json.loads(s) for s in (ROOT/'full_manifest.jsonl').read_text().splitlines()]
    assert all(not ({'answer','answer_id','gold','gold_candidate_index'}&r.keys()) for r in rows)
    by_video=defaultdict(list)
    for r in rows:by_video[r['video_id']].append(r)
    plan=json.loads((ROOT/('SHARD_PLAN_'+args.model+'.json')).read_text())
    video_ids=plan['shards'][args.shard_index]['video_ids']
    assert len(video_ids)==len(set(video_ids)) and set(video_ids)<=set(by_video)
    if args.max_videos:
        pilot_ids=json.loads((ROOT/'pilot_video_ids.json').read_text())
        assert args.max_videos==len(pilot_ids) and set(pilot_ids)<=set(video_ids)
        video_ids=sorted(pilot_ids)
    outdir=ROOT/'shards'/args.model/f'shard_{args.shard_index:02d}';outdir.mkdir(parents=True,exist_ok=True)
    records=outdir/'responses.jsonl'
    old=[json.loads(s) for s in records.read_text().splitlines()] if records.exists() else []
    done={(r['question_id'],r['phase']) for r in old}
    assert len(done)==len(old)
    job=os.environ.get('PBS_JOBID','manual')
    (outdir/f'model_manifest.{job}.json').write_bytes(model_receipt.read_bytes())
    started=time.perf_counter()
    if args.model=='internvl14':
        processor=None
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True,use_fast=False)
        model,loading=AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,use_flash_attn=False,output_loading_info=True)
        model.system_message=SYSTEM
    elif args.model=='qwen8':
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        tokenizer=processor.tokenizer
        model,loading=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    else:
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True)
        tokenizer=processor.tokenizer
        model,loading=AutoModelForImageTextToText.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    (outdir/f'weight_loading.{job}.json').write_text(json.dumps(loading,indent=2,default=str)+'\n')
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
    model=model.to('cuda').eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    from tokenizer_guard import validate as validate_tokenizer
    tokenizer_samples=[r['question']+'\n'+'\n'.join(r['candidates']) for r in rows]
    token_check=validate_tokenizer(tokenizer,model_dir,tokenizer_samples)
    expected_tokenizer=json.loads(Path(os.environ['TOKENIZER_PREFLIGHT_PATH']).read_text())
    assert token_check==expected_tokenizer,'Configured tokenizer differs from CPU preflight'
    (outdir/f'tokenizer_validation.{job}.json').write_text(json.dumps(token_check,indent=2)+'\n')
    # Exhaustive single-token spellings, rather than a guessed A/space-A subset.
    token_groups={c:[] for c in 'ABCDE'}
    for token_id in range(len(tokenizer)):
        text=tokenizer.decode([token_id],skip_special_tokens=False)
        if text.strip() in token_groups:token_groups[text.strip()].append(token_id)
    assert all(token_groups.values())
    (outdir/f'environment.{job}.json').write_text(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'av':av.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'gpu_memory':torch.cuda.get_device_properties(0).total_memory,'model_load_seconds':load_seconds,'token_groups':token_groups,'processor':{'class':type(processor).__name__ if processor else 'archived InternVL frame adapter','adapter_sha256':digest(ROOT/'archived_internvl_adapter.py')},'protocol_sha256':digest(ROOT/'SHARD_RUN_PROTOCOL.json'),'runner_sha256':digest(__file__),'model_manifest_sha256':digest(model_receipt)},indent=2,default=str)+'\n')
    def sync():torch.cuda.synchronize()
    def probs(logits,labels):
        values=torch.stack([torch.logsumexp(logits[token_groups[c]].float(),dim=0) for c in labels])
        return torch.softmax(values,dim=0).cpu().numpy()
    with torch.inference_mode():
        for video_id in video_ids:
            questions=by_video[video_id]
            if all((r['question_id'],p) in done for r in questions for p in PHASES):continue
            decode_start=time.perf_counter()
            path=str(Path(os.environ['VIDEO_ROOT'])/questions[0]['video_path'])
            with av.open(path) as container:
                stream=container.streams.video[0]
                # Same fallbacks as the original HERBench probe_video: fps 24 if absent, duration end+1 if absent,
                # frame count round(duration*fps) if the container does not store it.
                fps=float(stream.average_rate) if stream.average_rate else 24.0
                duration=float(stream.duration*stream.time_base) if stream.duration is not None and stream.time_base is not None else max(float(questions[0]['end'])+1.0,1.0)
                stored=int(stream.frames or 0);total=stored if stored>0 else max(1,round(duration*fps))
                assert total>0 and duration>0
                requested={(r['question_id'],p):indices(r,total,duration,p) for r in questions for p in PHASES}
                wanted={i for v in requested.values() for i in v};decoded={}
                stream.thread_type='AUTO'
                for i,frame in enumerate(container.decode(stream)):
                    if i in wanted:decoded[i]=frame.to_ndarray(format='rgb24')
                    if i>=max(wanted):break
                    last=frame
            # An estimated frame count can exceed the decodable frames; such indices take the final frame.
            clamped=sorted(wanted-set(decoded))
            assert stored==0 or not clamped
            for i in clamped:decoded[i]=last.to_ndarray(format='rgb24')
            assert set(decoded)==wanted
            decode_seconds=time.perf_counter()-decode_start
            append(outdir/'decode.jsonl',{'video_id':video_id,'seconds':decode_seconds,'unique_frames':len(wanted),'frame_count':total,'duration':duration,'fps':fps,'stored_frame_count':stored,'clamped_indices':clamped,'sha256':digest(path),'job':job})
            caches={}
            for row in questions:
                labels=list('ABCDE'[:len(row['candidates'])])
                for phase in PHASES:
                    key=(row['question_id'],phase)
                    if key in done:continue
                    call_start=time.perf_counter()
                    frame_indices=requested[key]
                    video=np.stack([decoded[i] for i in frame_indices])
                    text=row['question']+'\n'+'\n'.join(c+'. '+s for c,s in zip(labels,row['candidates']))+'\nAnswer with exactly one letter.'
                    images=[Image.fromarray(decoded[i]) for i in frame_indices]
                    if args.model=='internvl14':
                        pixels=archived_internvl.preprocess_frames(images,torch.bfloat16,'cuda')
                        rendered,ids,(mask,flags)=archived_internvl.build_query(model,tokenizer,text,[1]*len(images))
                        inp={'pixel_values':pixels,'input_ids':ids,'attention_mask':mask}
                        assert list(pixels.shape)==[len(images),3,448,448]
                    elif args.model=='qwen8':
                        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}]
                        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                        cpu_inputs=processor(text=[rendered],videos=[video],video_metadata=[{'total_num_frames':total,'fps':fps,'frames_indices':frame_indices}],do_sample_frames=False,cap_pixels_per_frame=False,return_tensors='pt')
                        inp={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in cpu_inputs.items()}
                        assert inp['video_grid_thw'][0,0].item()*2==len(images)
                    else:
                        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':text}]}]
                        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                        cpu_inputs=processor(text=[rendered],images=images,padding=True,return_tensors='pt')
                        inp={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in cpu_inputs.items()}
                        inp['pixel_values']=inp['pixel_values'].to(torch.bfloat16)
                        assert list(inp['image_grid_thw'].shape)==[len(images),3]
                    sync();preprocess_seconds=time.perf_counter()-call_start
                    torch.cuda.reset_peak_memory_stats()
                    # Reference: publication's greedy generation and first-step logits.
                    t=time.perf_counter()
                    generated=model.generate(**inp,max_new_tokens=16,do_sample=False,return_dict_in_generate=True,output_logits=True)
                    sync();reference_seconds=time.perf_counter()-t
                    reference_logits=generated.logits[0][0]
                    reference_probs=probs(reference_logits,labels)
                    output_text=tokenizer.decode(generated.sequences[0,-len(generated.logits):],skip_special_tokens=True)
                    reference_answer=parse_answer(output_text,labels)
                    # The generated answer is only an audit of the probability argmax; keep going if it cannot be parsed.
                    if reference_answer is None:print('PARSE_FAILURE',row['question_id'],phase,repr(output_text[:80]),flush=True)
                    del generated
                    if args.reference_only:
                        record={'question_id':row['question_id'],'video_id':video_id,'role':row['role'],'phase':phase,'view_id':VIEW_IDS[phase],'frame_budget':FRAME_COUNTS[phase],'frame_indices':frame_indices,'candidate_probabilities':reference_probs.tolist(),'answer':reference_answer,'generated_text':output_text,'preprocess_seconds':preprocess_seconds,'reference_seconds':reference_seconds,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'input_tokens':int(inp['input_ids'].shape[1]),'video_grid':inp['video_grid_thw'].tolist() if 'video_grid_thw' in inp else None,'input_shapes':{k:list(v.shape) for k,v in inp.items() if isinstance(v,torch.Tensor)},'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'job':job,'model_revision':model_info['revision'],'inference_path':'uncached_greedy_reference','optimized_cache_used':False,'model_id':model_info['model_id']}
                        append(records,record);done.add(key)
                        print('REFERENCE_PHASE_DONE',len(done),row['question_id'],phase,flush=True)
                        del inp,reference_logits
                        continue
            caches.clear()
    all_records=[json.loads(s) for s in records.read_text().splitlines()]
    expected={(r['question_id'],p) for v in video_ids for r in by_video[v] for p in PHASES}
    assert expected<=done
    if not args.max_videos:assert done==expected
    if args.reference_only:
        summary={'complete':True,'mode':'uncached_greedy_reference','videos':len(video_ids),'questions':len(expected)//4,'calls':len(expected),'gpu':torch.cuda.get_device_name(),'model_load_seconds':load_seconds,'session_wall_seconds':time.perf_counter()-started,'responses_sha256':digest(records),'test_labels_read':False,'optimized_cache_used':False,'model_id':model_info['model_id'],'peak_memory_bytes':max(r['peak_memory_bytes'] for r in all_records)}
        for field in ['reference_seconds','preprocess_seconds']:
            a=np.array([r[field] for r in all_records]);summary[field]={'sum':float(a.sum()),'median':float(np.median(a)),'p90':float(np.percentile(a,90))}
        (outdir/('PILOT_COMPLETE.json' if args.max_videos else 'COMPLETE.json')).write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        return
if __name__=='__main__':main()
