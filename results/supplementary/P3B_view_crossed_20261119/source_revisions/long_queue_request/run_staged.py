import json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    model=os.environ['MODEL_KEY'];assert model in ['qwen8','internvl14','llava8']
    start=time.monotonic();cap=17100
    assert json.loads((ROOT/'CPU_INPUT_VALIDATION.json').read_text())['passed']
    destination=Path(os.environ['TMPDIR'])/('weights_'+model)
    os.environ['LOCAL_MODEL_DIR']=str(destination)
    def run(*args):subprocess.run([sys.executable,'-B',*args],cwd=ROOT,check=True)
    run('preflight_assets.py')
    run('tokenizer_preflight.py',model)
    run('prepare_models.py',model,'--destination',str(destination))
    out=ROOT/('responses_'+model)
    if not (ROOT/(model+'_PILOT_VALIDATION.json')).exists():
        run('run_peer.py','--model',model,'--reference-only','--max-videos','50')
        run('validate_native.py',model,'--pilot')
        shutil.copytree(out,ROOT/('pilot_snapshot_'+model),dirs_exist_ok=False)
    gate=json.loads((ROOT/(model+'_PILOT_VALIDATION.json')).read_text());assert gate['passed']
    available=cap-(time.monotonic()-start)-180
    decision={'model':model,'elapsed_seconds':time.monotonic()-start,'available_seconds':available,'estimated_remaining_seconds':gate['remaining_estimate_seconds'],'safety_multiplier':1.5,'labels_used':False}
    decision['continue']=1.5*gate['remaining_estimate_seconds']<available
    (ROOT/(model+'_BUDGET_GATE.'+os.environ.get('PBS_JOBID','manual')+'.json')).write_text(json.dumps(decision,indent=2)+'\n')
    if not decision['continue']:raise SystemExit('TIMING_GATE_REQUIRES_NEW_RESOURCE_ESTIMATE')
    run('run_peer.py','--model',model,'--reference-only')
    run('validate_native.py',model)
if __name__=='__main__':main()
