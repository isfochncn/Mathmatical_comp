"""Reproduce one saved 06:00 decision for explanation, without changing its run."""
from pathlib import Path
import json
import argparse
import numpy as np
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline
from microgrid.adaptive_forecast import AdaptiveForecaster
from microgrid.planning import solve_window,FeeMode

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',required=True,help='Archived problem3 directory containing trajectory.npz')
args=parser.parse_args()
folder=Path(args.run_dir)
z=dict(np.load(folder/'trajectory.npz'))
audits={a['formed_at']:a for a in json.loads((folder/'forecast_audit.json').read_text(encoding='utf-8'))}
bundle=load_all(); timeline=build_timeline(bundle); forecaster=AdaptiveForecaster(bundle,timeline)
at=(31+26)*1440+360;midnight=(at//1440+1)*1440
price=float(np.roll(bundle.attachment1.price_yuan_per_kwh,1)[36])
f=forecaster.window_forecast(at,at,midnight+1440,use_published_pv=True,current_price=price,price_mode='repeated')
r=forecaster.risk_requirements(at,f,published=True,adjustable=True)
f.reserve_energy_kwh=r['reserve_energy_kwh'];f.net_upper_kwh=r['net_upper_kwh']
np.testing.assert_allclose(f.pv_kwh,audits[at]['pv_forecast_path_kwh'],atol=1e-7,rtol=0)
index={int(m):i for i,m in enumerate(z['abs_minute'])}
today=f.abs_minutes<midnight;o=np.zeros(f.n)
o[today]=[z['plan_initial_kwh'][index[int(m)]] for m in f.abs_minutes[today]]
soc=float(z['soc_boundary_kwh'][index[at]])
result=solve_window(forecast=f,soc_start_kwh=soc,fee_mode=FeeMode.ADJUSTABLE,o_kwh=o,a_kwh=np.zeros(f.n),
    price_now_yuan_per_kwh=price,adjustable_mask=today,allow_spill=True,problem_name='diagnostic-replay').require_ok()
assert abs(result.objective_yuan-audits[at]['forecast_cost'])<1e-4
rows=[]
for offset in [0,6,12,18,19,24,30,35]:
    m=int(f.abs_minutes[offset]);idx=index[m]
    rows.append({'time':f'{m%1440//60:02d}:{m%60:02d}',
        'soc_forecast_at_06':float(result.soc_boundary_kwh[offset]),'soc_actual':float(z['soc_boundary_kwh'][idx]),
        'grid_decided_at_06':float(result.grid_kwh[offset]),'grid_executed':float(z['grid_kwh'][idx]),
        'emergency_forecast':float(result.emergency_kwh[offset]),'emergency_actual':float(z['emergency_kwh'][idx]),
        'net_forecast':float(f.demand_kwh[offset]-f.pv_kwh[offset]),'net_actual':float(z['demand_kwh'][idx]-z['pv_kwh'][idx])})
data={'date':'2025-02-27','node':'06:00','forecast_objective_matches_saved':True,
      'soc_start':soc,'forecast_emergency_until_12':float(result.emergency_kwh[:36].sum()),
      'reserve_stock_slack':result.reserve_stock_shortfall_kwh,'rows':rows}
Path('outputs/问题三费用诊断_20260912/node_replay.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data,indent=2))
