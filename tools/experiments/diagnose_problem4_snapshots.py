"""Replay archived decision windows for diagnosis; leave source/runs unchanged."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from microgrid.absolute_run import RunConfig,make_forecaster
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline
from microgrid.planning import solve_window,FeeMode,WindowForecast
from microgrid.report_forecast import ReportAwareForecaster

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/问题四_1至2月检验_20260912_165702/紧急购电诊断'
bundle=load_all();timeline=build_timeline(bundle)
origin=datetime(2025,1,1)
results=[]
for problem,when in [('problem4-3','2025-02-12 06:00'),('problem4-3','2025-02-27 06:00'),('problem4-2','2025-02-27 00:00'),('problem4-2','2025-01-05 00:00')]:
 p=OUT.parent/problem/'运行记录'/problem;z=np.load(p/'trajectory.npz')
 audits=json.loads((p/'forecast_audit.json').read_text(encoding='utf-8'))
 now=int((datetime.fromisoformat(when)-origin).total_seconds()/60);idx=(now-10)//10
 end=(now//1440+2)*1440;dayend=end-1440;stop=min(dayend,now+360)
 fc=make_forecaster(RunConfig(problem=problem,pv_method="pooled"),bundle,timeline)
 record={'problem':problem,'when':when,'soc_start':float(z['soc_boundary_kwh'][idx]),'variants':{}}
 for label,forecaster in [('archived',fc),('optimized_pv',ReportAwareForecaster(bundle,timeline,history_days=28,risk_quantile=.9,variant='blend',nowcast=True))]:
  f=forecaster.window_forecast(now,now,end,use_published_pv=True,current_price=None,price_mode='historical')
  risk=forecaster.risk_requirements(now,f,published=True,adjustable=problem=='problem4-3')
  f.reserve_energy_kwh=risk['reserve_energy_kwh'];f.net_upper_kwh=risk['net_upper_kwh']
  today=f.abs_minutes<dayend;o=np.zeros(f.n);o[today]=z['plan_initial_kwh'][idx:idx+int(today.sum())]
  mode=FeeMode.ADJUSTABLE if problem=='problem4-3' else FeeMode.FIRST_PLAN
  r=solve_window(forecast=f,soc_start_kwh=record['soc_start'],fee_mode=mode,o_kwh=o,a_kwh=np.zeros(f.n),
    price_now_yuan_per_kwh=float(f.price_yuan_per_kwh[0]),adjustable_mask=today if mode==FeeMode.ADJUSTABLE else None,allow_spill=True).require_ok()
  n=(stop-now)//10
  if label=='archived':
   assert np.max(np.abs(f.pv_kwh-np.array(audits[idx]['pv_forecast_path_kwh'])))<1e-7
   assert np.max(np.abs(r.grid_kwh[:n]-z['grid_kwh'][idx:idx+n]))<1e-4
  actualnet=z['demand_kwh'][idx:idx+n]-z['pv_kwh'][idx:idx+n]
  # Ex-post lower bound: exact minimum urgent with this fixed ordinary plan,
  # actual six-hour supply/demand, and fully flexible physical battery dispatch.
  truth=WindowForecast(f.abs_minutes[:n],z['demand_kwh'][idx:idx+n],z['pv_kwh'][idx:idx+n],
    np.ones(n),np.full(n,'ex-post diagnostic only',dtype=object))
  oracle=solve_window(forecast=truth,soc_start_kwh=record['soc_start'],fee_mode=FeeMode.FROZEN,
    o_kwh=r.grid_kwh[:n],a_kwh=np.zeros(n),committed_mask=np.ones(n,dtype=bool),
    committed_grid_kwh=r.grid_kwh[:n],allow_spill=True).require_ok()
  error=actualnet-(f.demand_kwh[:n]-f.pv_kwh[:n]);trace=[]
  for j in range(n):
   if j%6==0 or z['emergency_kwh'][idx+j]>100:
    trace.append({'minute':int(f.abs_minutes[j]%1440),'load_hat':float(f.demand_kwh[j]),'pv_hat':float(f.pv_kwh[j]),
      'load':float(z['demand_kwh'][idx+j]),'pv':float(z['pv_kwh'][idx+j]),'grid':float(r.grid_kwh[j]),
      'soc_hat':float(r.soc_boundary_kwh[j]),'soc_actual':float(z['soc_boundary_kwh'][idx+j]),
      'soc_end_hat':float(r.soc_boundary_kwh[j+1]),'charge_hat':float(r.charge_kwh[j]),'discharge_hat':float(r.discharge_kwh[j]),
      'urgent_actual':float(z['emergency_kwh'][idx+j]),'reserve':float(f.reserve_energy_kwh[j]),'price_hat':float(f.price_yuan_per_kwh[j]),
      'price_actual':float(z['price_actual'][idx+j]),'cum_net_error':float(error[:j+1].sum())})
  record['variants'][label]={'grid_first6h':float(r.grid_kwh[:n].sum()),'pv_first6h':float(f.pv_kwh[:n].sum()),
   'actual_pv_first6h':float(z['pv_kwh'][idx:idx+n].sum()),'reserve_at_start':float(f.reserve_energy_kwh[0]),
   'reserve_stock_slack':float(r.reserve_stock_shortfall_kwh),'risk_penalty':float(r.risk_penalty_yuan),
   'max_cumulative_net_error':float(np.maximum(0,np.cumsum(error)).max()),
   'expost_min_urgent_given_candidate_grid':float(oracle.emergency_kwh.sum()),
   'actual_urgent_first6h':float(z['emergency_kwh'][idx:idx+n].sum()),'trace':trace}
 if problem=='problem4-3':
  original=z['plan_initial_kwh'][idx:idx+n]
  unchanged=solve_window(forecast=truth,soc_start_kwh=record['soc_start'],fee_mode=FeeMode.FROZEN,
    o_kwh=original,a_kwh=np.zeros(n),committed_mask=np.ones(n,dtype=bool),
    committed_grid_kwh=original,allow_spill=True).require_ok()
  record['no_6am_reduction_expost_min_urgent']=float(unchanged.emergency_kwh.sum())
  record['original_grid_first6h']=float(original.sum())
 results.append(record)
 print(json.dumps({k:v for k,v in record.items() if k!='variants'}),flush=True)
 for k,v in record['variants'].items():print(k,json.dumps({key:value for key,value in v.items() if key!='trace'}),flush=True)
(OUT/'snapshot_replays.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
