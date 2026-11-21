"""One preplanned native-view shard in a bounded single-GPU allocation."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    model=os.environ['MODEL_KEY'];index=int(os.environ['PBS_ARRAY_INDEX']);job=os.environ['PBS_JOBID']
    plan=json.loads((ROOT/('SHARD_PLAN_'+model+'.json')).read_text());assert 0<=index<len(plan['shards'])
    if model=='internvl14':
        legacy=ROOT.parent/'P3B_view_crossed_20261119'
        assert json.loads((legacy/'LEGACY_INTERNVL_COMPATIBILITY.json').read_text())['passed']
        os.environ['PYTHONPATH']=str(legacy/'internvl_legacy_python')+os.pathsep+os.environ.get('PYTHONPATH','')
    stage=ROOT/'staging'/model/job;stage.mkdir(parents=True,exist_ok=False)
    os.environ['MODEL_RECEIPT_PATH']=str(stage/'MODEL_READY.json')
    os.environ['TOKENIZER_PREFLIGHT_PATH']=str(stage/'TOKENIZER_PREFLIGHT.json')
    os.environ['LOCAL_MODEL_DIR']=str(Path(os.environ['TMPDIR'])/('weights_'+model))
    def run(*args):subprocess.run([sys.executable,'-B',*args],cwd=ROOT,check=True)
    import torch;props=torch.cuda.get_device_properties(0)
    gpu={'name':props.name,'total_gib':round(props.total_memory/2**30,1),'job':job}
    print('GPU_GUARD',json.dumps(gpu),flush=True)
    assert 'H200' in props.name and props.total_memory>=100*2**30,('full H200 required, not a MIG slice',gpu)
    (stage/'GPU.json').write_text(json.dumps(gpu))
    os.environ['VIDEO_ROOT']=str(Path(os.environ['TMPDIR'])/'herbench')
    run('fetch_videos.py','--model',model,'--shard-index',str(index))
    run('preflight_assets.py','--protocol','SHARD_RUN_PROTOCOL.json','--runner','run_shard.py')
    run('tokenizer_shard_preflight.py',model)
    run('prepare_shard_model.py',model,'--destination',os.environ['LOCAL_MODEL_DIR'])
    run('run_shard.py','--model',model,'--reference-only','--shard-index',str(index))
    print('SHARD_COMPLETE',model,index,flush=True)
if __name__=='__main__':main()
