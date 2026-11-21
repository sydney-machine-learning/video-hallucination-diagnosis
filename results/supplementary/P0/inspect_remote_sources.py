"""Copy remote source/seal evidence without changing either archive."""
import difflib
import json
import subprocess
from pathlib import Path
from collect_remote import SSH, OUT

ROOT=OUT.parents[2]
sources={
 "original_herbench_seal.json":("/home/z5611125/trace_rc_route/results_fresh/herbench_confirmation/8918264.kman.restech.unsw.edu.au/HERBENCH_PREINFERENCE_SEAL_20260802.json", "results/herbench_confirmation/8918264.kman.restech.unsw.edu.au/HERBENCH_PREINFERENCE_SEAL_20260802.json"),
 "remote_herbench_persistent_worker.py":("/home/z5611125/trace_rc_route/katana/herbench_persistent_worker.py","katana/herbench_persistent_worker.py"),
 "remote_analyze_herbench_trace_rc.py":("/home/z5611125/trace_rc_route/src/analyze_herbench_trace_rc.py","src/analyze_herbench_trace_rc.py"),
}
records=[]
for name,(source,local) in sources.items():
    result=subprocess.run(SSH+["cat -- '"+source+"'"],capture_output=True,timeout=60)
    record={"remote":source,"repository_file":local,"returncode":result.returncode,"stderr":result.stderr.decode(errors="replace")}
    if result.returncode==0:
        path=OUT/"inputs/remote_code"/name
        with path.open("xb") as f:f.write(result.stdout)
        record["copy"]=str(path.relative_to(OUT))
        diff="".join(difflib.unified_diff(result.stdout.decode().splitlines(True),
                    (ROOT/local).read_text(encoding="utf-8").splitlines(True),
                    fromfile="remote/"+name,tofile=local))
        with (OUT/(name+".diff")).open("x",encoding="utf-8") as f:f.write(diff)
    records.append(record)
with (OUT/"remote_source_transfers.json").open("x",encoding="utf-8") as f:json.dump(records,f,indent=2)
print(json.dumps(records,indent=2))
