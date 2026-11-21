"""Save selected PBS accounting fields, excluding environment variables."""
import json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'accounting';out.mkdir(exist_ok=True)
    for job in sys.argv[1:]:
        assert re.fullmatch(r'[0-9]+(?:\[[0-9]*\])?',job.split('.')[0])
        result=subprocess.run(['qstat','-xf',job],text=True,capture_output=True)
        attributes={};key=None
        for line in result.stdout.splitlines():
            match=re.match(r'^    (\S+) = (.*)$',line)
            if match:key,value=match.groups();attributes[key]=value
            elif key and line.startswith(('\t','        ')):attributes[key]+=line.strip()
        chosen=['resources_used.walltime','resources_used.cput','resources_used.mem','resources_used.ncpus',
            'resources_used.GPU_duration','resources_used.GPU_maxGpuMemoryUsed','Exit_status','job_state',
            'stime','etime','obittime','exec_host','Resource_List.select','Resource_List.walltime']
        fields={key:attributes[key] for key in chosen if key in attributes}
        data={'job':job,'utc':datetime.now(timezone.utc).isoformat(),'qstat_exit_code':result.returncode,
            'fields':fields,'error':result.stderr.strip() if result.returncode else None}
        (out/(job+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({'job':job,'walltime':fields.get('resources_used.walltime'),'exit':fields.get('Exit_status')}),flush=True)
if __name__=='__main__':main()
