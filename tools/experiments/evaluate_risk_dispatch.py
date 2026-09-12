"""Paired, explicitly restarted two-day dispatch tests; not annual savings estimates."""
from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import numpy as np
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline
from microgrid.forecast import Forecaster
from microgrid.adaptive_forecast import AdaptiveForecaster
from microgrid.absolute_run import POLICIES
from microgrid.simulation import RunOptions, run_absolute
from microgrid.validation import validate_absolute_run


def evaluate(job):
    root, problem, start, variant, out = job
    bundle = load_all()
    tl = build_timeline(bundle)
    begin = (date.fromisoformat(start)-date(2025, 1, 1)).days*1440
    z = dict(np.load(Path(root)/'运行记录'/problem/'trajectory.npz'))
    ix = int(np.flatnonzero(z['abs_minute'] == begin)[0])
    soc = float(z['soc_boundary_kwh'][ix])
    forecaster = (Forecaster(bundle, tl) if variant == 'baseline' else
                  AdaptiveForecaster(bundle, tl, risk_quantile=.90 if variant == 'reserve90' else 0))
    result = run_absolute(timeline=tl, forecaster=forecaster, policy=POLICIES[problem],
        options=RunOptions(experiment='paired-validation', evaluation_restart=True),
        abs_from=begin, abs_to=begin+2880, soc_start_kwh=soc)
    run = result.run
    arr = {key:np.array([getattr(step, key) for step in run.steps]) for key in
           ('grid_kwh', 'emergency_kwh', 'charge_kwh', 'discharge_kwh', 'curtail_kwh', 'surplus_kwh')}
    idx = run.abs_minutes//10
    validate_absolute_run(abs_minutes=run.abs_minutes, **arr, soc_boundary_kwh=run.soc_boundary_kwh,
        demand_kwh=tl.demand_kwh.values[idx], pv_kwh=tl.pv_kw.values[idx]/6,
        soc_start_kwh=soc, allow_spill=True)
    costs = sum(e.cost_yuan for e in run.events)
    costs += sum(p.penalty_yuan for p in run.penalties)
    record = dict(problem=problem, start=start, days=2, variant=variant, cost_yuan=costs,
        grid_kwh=float(arr['grid_kwh'].sum()), emergency_kwh=float(arr['emergency_kwh'].sum()),
        spill_kwh=float(arr['surplus_kwh'].sum()), soc_start=soc, soc_end=float(run.soc_boundary_kwh[-1]),
        risk_shortfall_max=max(a['reserve_shortfall_kwh'] for a in result.forecast_audits),
        n_risk_shortfall=sum(a['reserve_shortfall_kwh'] > 1e-5 for a in result.forecast_audits))
    dest = Path(out)/'paired_runs'/f'{problem}_{start}_{variant}'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'trajectory.npz', abs_minute=run.abs_minutes,
        soc_boundary_kwh=run.soc_boundary_kwh, **arr)
    (dest/'summary.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (dest/'forecast_audit.json').write_text(json.dumps(result.forecast_audits), encoding='utf-8')
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/预测与风险优化_20260912')
    ap.add_argument('--baseline-dir',required=True,help='Explicit archived annual baseline directory')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--variants', nargs='+', default=['baseline', 'adaptive', 'reserve90'],
                    choices=['baseline', 'adaptive', 'reserve90'])
    args = ap.parse_args()
    root = args.baseline_dir
    jobs = [(root, p, d, v, args.out) for p in ('problem2','problem3','problem4-2','problem4-3')
            for d in ('2025-02-06','2025-12-08') for v in args.variants]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            r = future.result()
            records.append(r)
            print(json.dumps(r), flush=True)
    records = [json.loads(p.read_text(encoding='utf-8'))
               for p in (Path(args.out)/'paired_runs').glob('*/summary.json')]
    result = {'scope':'Paired 2-day restarts from the same prior baseline SOC; February and December only. '
                       'Terminal SOC is free and reported; costs are not annual or SOC-neutral savings.',
              'runs':sorted(records, key=lambda r:(r['problem'],r['start'],r['variant']))}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out)/'dispatch_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
