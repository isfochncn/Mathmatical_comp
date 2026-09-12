"""Run the current main model continuously from Jan 1, retaining one reporting month."""
from pathlib import Path
from datetime import date, datetime
import argparse
import calendar
import contextlib
import hashlib
import json
import time
import traceback

from microgrid.absolute_run import RunConfig, run, save_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='2025-07')
    ap.add_argument('--out', required=True)
    ap.add_argument('--problem', choices=('problem2','problem3'), default='problem2')
    args = ap.parse_args()
    year, month = map(int, args.month.split('-'))
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in list(Path('src/microgrid').glob('*.py'))+list(Path('data/附件5').glob('result*.xlsx'))}
    (out/'source_and_template_hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf-8')
    status = {'state':'running', 'problem':args.problem, 'month':args.month,
              'started_at':datetime.now().isoformat(), 'out':str(out)}
    status_path = out/'status.json'
    def write_status():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    write_status()
    start = time.perf_counter()
    try:
        with (out/'run.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log):
            result = run(RunConfig(problem=args.problem, out_dir=out/'运行记录',
                         run_from=first, run_to=last, progress_every_days=1))
            path = save_run(result)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
        status.update(state='simulation_complete', run_dir=str(path), summary=result.summary,
                      elapsed_seconds=time.perf_counter()-start, completed_at=datetime.now().isoformat())
        write_status()
        print(json.dumps(status, ensure_ascii=True, indent=2))
    except BaseException:
        status.update(state='failed', error=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
        write_status()
        raise


if __name__ == '__main__':
    main()
