"""Reproducible chronological forecast/reserve evaluation; no model fitting on test targets."""
from pathlib import Path
import argparse
import json
import csv
import numpy as np
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline
from microgrid.forecast import Forecaster
from microgrid.adaptive_forecast import AdaptiveForecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/forecast_risk_validation')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_all()
    timeline = build_timeline(bundle)
    base = Forecaster(bundle, timeline)
    improved = AdaptiveForecaster(bundle, timeline)
    records, rows = {}, []
    for day in range(31, 365):
        for published, adjustable in [(False, False), (True, False), (True, True)]:
            branch = 'published-adjustable' if adjustable else ('published-frozen' if published else 'historical-frozen')
            for hour in ((0, 6, 12, 18) if adjustable else (0,)):
                now = day*1440+hour*60
                end = (day+2)*1440
                size = 36 if adjustable else 144
                f0 = base.window_forecast(now, now, end, use_published_pv=published,
                                         current_price=None, price_mode='repeated')
                f1 = improved.window_forecast(now, now, end, use_published_pv=published,
                                             current_price=None, price_mode='repeated')
                risk = improved.risk_requirements(now, f1, published=published, adjustable=adjustable)
                idx = np.arange(now//10, now//10+size)
                actual_d = timeline.load_kw.values[idx]/6
                actual_p = timeline.pv_kw.values[idx]/6
                error = actual_d-actual_p-(f1.demand_kwh[:size]-f1.pv_kwh[:size])
                peak = max(0., np.cumsum(error).max())
                for label, forecast in [('baseline', f0), ('adaptive', f1)]:
                    key = branch+'/'+label
                    d = actual_d-forecast.demand_kwh[:size]
                    p = actual_p-forecast.pv_kwh[:size]
                    records.setdefault(key, []).append(np.column_stack((d, p, d-p, actual_d, actual_p)))
                rows.append(dict(day=day, hour=hour, branch=branch, n=size,
                    load_mae_base=float(np.mean(np.abs(actual_d-f0.demand_kwh[:size]))),
                    load_mae_adaptive=float(np.mean(np.abs(actual_d-f1.demand_kwh[:size]))),
                    pv_mae_base=float(np.mean(np.abs(actual_p-f0.pv_kwh[:size]))),
                    pv_mae_adaptive=float(np.mean(np.abs(actual_p-f1.pv_kwh[:size]))),
                    energy_buffer_kwh=float(risk['reserve_energy_kwh'][0]),
                    actual_peak_error_kwh=float(peak), energy_covered=int(peak <= risk['reserve_energy_kwh'][0]+1e-6),
                    interval_covered=int(np.sum(actual_d-actual_p <= risk['net_upper_kwh'][:size]+1e-6)),
                    calibration_paths=risk['risk_sample_count']))
        if day%30 == 0:
            print(f'forecast validation: day {day}', flush=True)
    summary = {}
    for key, blocks in records.items():
        a = np.concatenate(blocks)
        summary[key] = {name: {'mae_kwh':float(np.mean(np.abs(a[:, i]))),
                               'rmse_kwh':float(np.sqrt(np.mean(a[:, i]**2))),
                               'bias_actual_minus_forecast_kwh':float(np.mean(a[:, i]))}
                        for i, name in enumerate(('load', 'pv', 'net'))}
        summary[key]['n'] = len(a)
    coverage = {}
    for branch in sorted({r['branch'] for r in rows}):
        r = [x for x in rows if x['branch'] == branch]
        coverage[branch] = dict(blocks=len(r), energy_coverage=float(np.mean([x['energy_covered'] for x in r])),
            interval_coverage=sum(x['interval_covered'] for x in r)/sum(x['n'] for x in r),
            mean_buffer_kwh=float(np.mean([x['energy_buffer_kwh'] for x in r])),
            max_buffer_kwh=max(x['energy_buffer_kwh'] for x in r))
    result = {'period':'2025-02-01 00:00 to 2026-01-01 00:00',
              'scope':'rolling historical evaluation; model design informed by this dataset, not an untouched prospective holdout',
              'forecasts':summary, 'risk_coverage':coverage}
    (out/'forecast_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'forecast_daily.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
