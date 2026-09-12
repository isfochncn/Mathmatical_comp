"""Run one authorized branch continuously from January 1 and archive its ledger."""
from pathlib import Path
from datetime import date, datetime
import argparse
import contextlib
import hashlib
import json
import time
import traceback

from microgrid.absolute_run import RunConfig, run, save_run, describe_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--problem',required=True,choices=('problem2','problem3','problem4-2','problem4-3'))
    ap.add_argument('--out')
    ap.add_argument('--describe',action='store_true',help='Resolve the model without creating files or running the solver')
    ap.add_argument('--run-from',default='2025-02-01')
    ap.add_argument('--run-to',default='2025-12-31')
    ap.add_argument('--pv-method',default='auto',choices=('auto','pooled','report_blend'))
    args=ap.parse_args()
    if not args.out and not args.describe:
        ap.error('--out is required unless --describe is used')
    out=Path(args.out or '.').resolve()
    config=RunConfig(problem=args.problem,pv_method=args.pv_method,out_dir=out/'运行记录',
        run_from=date.fromisoformat(args.run_from),run_to=date.fromisoformat(args.run_to),progress_every_days=1)
    contract=describe_model(config)
    print(json.dumps(contract,ensure_ascii=True,indent=2),flush=True)
    if args.describe:
        return
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/'status.json'
    if status_path.exists():
        raise ValueError('Use a fresh output directory; previous run state must be preserved')
    files=list(Path('src/microgrid').glob('*.py'))+list(Path('data/附件5').glob('result*.xlsx'))
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'source_and_template_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'run_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    status={'state':'running','problem':args.problem,'report_from':args.run_from,'report_to':args.run_to,
        'started_at':datetime.now().isoformat(),'out':str(out),'pv_method':config.effective_pv_method()}
    write=lambda:status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    write();start=time.perf_counter()
    try:
        with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log):
            result=run(config)
            path=save_run(result)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        status.update(state='simulation_complete',run_dir=str(path),summary=result.summary,
            elapsed_seconds=time.perf_counter()-start,completed_at=datetime.now().isoformat())
        write();print(json.dumps(status,ensure_ascii=True,indent=2))
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        write();raise


if __name__=='__main__':
    main()
