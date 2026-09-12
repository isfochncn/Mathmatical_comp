import hashlib, json, shutil, sys, time, traceback
from datetime import datetime
from pathlib import Path
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from microgrid.absolute_run import RunConfig, run, save_run
from microgrid.cli import cmd_export
p=sys.argv[1]
status_file=ROOT/'运行记录'/f'{p}_status.json'
started=time.time()
def status(state, **extra):
    status_file.write_text(json.dumps(dict(problem=p,state=state,started=datetime.fromtimestamp(started).isoformat(),updated=datetime.now().isoformat(),elapsed_seconds=time.time()-started,**extra),ensure_ascii=False,indent=2),encoding='utf-8')
status('running')
try:
    config=RunConfig(problem=p,out_dir=ROOT/'运行记录',progress_every_days=1)
    result=run(config)
    status('saving',summary=result.summary)
    folder=save_run(result)
    status('exporting',summary=result.summary)
    rc=cmd_export(Namespace(problem=p,out=str(ROOT/'运行记录'),experiment='main'))
    if rc: raise RuntimeError(f'Export failed: {rc}')
    name=p.replace('problem','result')+'.xlsx'
    src=folder/'result'/name
    dest=ROOT/'result'/name
    if dest.exists(): raise FileExistsError(dest)
    shutil.copy2(src,dest)
    hashes=json.loads((ROOT/'核验'/'original_template_hashes.json').read_text(encoding='utf-8'))
    for path, expected in hashes.items():
        if hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()!=expected: raise RuntimeError(f'Original template changed: {path}')
    status('complete',summary=result.summary,output=str(dest),output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    print('COMPLETE',p,str(dest),flush=True)
except BaseException as exc:
    status('failed',error=str(exc),traceback=traceback.format_exc())
    traceback.print_exc()
    raise
