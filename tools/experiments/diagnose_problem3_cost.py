"""Compare saved February simulations without changing models or rerunning them."""
from pathlib import Path
from datetime import datetime,timedelta
import json
import argparse
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--problem2-out',required=True)
parser.add_argument('--problem3-out',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
root=Path(args.out);root.mkdir(exist_ok=True)
reports={}
for problem,saved in [('problem2',args.problem2_out),('problem3',args.problem3_out)]:
    folder=Path(saved)
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    run=Path(status['run_dir']); z=dict(np.load(run/'trajectory.npz'))
    audit={a['formed_at']:a for a in json.loads((run/'forecast_audit.json').read_text(encoding='utf-8'))}
    payload=json.loads((folder/'month_payload.json').read_text(encoding='utf-8'))
    mask=(z['abs_minute']>=44650)&(z['abs_minute']<84970);i=np.flatnonzero(mask);t=z['abs_minute'][i]
    d,pv,g,u,pr=[z[k][i] for k in ['demand_kwh','pv_kwh','grid_kwh','emergency_kwh','price_actual']]
    soc=z['soc_boundary_kwh'][i];dis=z['discharge_kwh'][i]
    pred=np.array([[audit[int(m)]['demand_forecast_kwh'],audit[int(m)]['pv_forecast_kwh']] for m in t])
    decision=np.array([int(m//(360 if problem=='problem3' else 1440))*(360 if problem=='problem3' else 1440) for m in t])
    legal=np.array([[audit[int(a)]['demand_forecast_path_kwh'][(m-a)//10],audit[int(a)]['pv_forecast_path_kwh'][(m-a)//10]] for m,a in zip(t,decision)])
    err=(d-pv)-(pred[:,0]-pred[:,1]);legalerr=(d-pv)-(legal[:,0]-legal[:,1]);urgent=u>1e-6
    theoretical=np.maximum(0,d-pv-g-np.minimum(750,.9*(soc-1200)))
    power=np.maximum(0,d-pv-g-750)
    metrics={}
    for name,prediction in [('rolling',pred),('last_legal_node',legal)]:
        erd=d-prediction[:,0];erp=pv-prediction[:,1];ern=erd-erp
        metrics[name]={'load_mae':float(np.mean(abs(erd))),'pv_mae':float(np.mean(abs(erp))),
            'pv_bias_actual_minus_pred':float(erp.mean()),'net_mae':float(abs(ern).mean()),
            'net_underforecast_sum':float(np.maximum(ern,0).sum()),
            'urgent_load_error_mean':float(erd[urgent].mean()),'urgent_pv_error_mean':float(erp[urgent].mean()),
            'urgent_net_error_mean':float(ern[urgent].mean())}
    hours=[]
    for h in range(24):
        k=t%1440//60==h
        hours.append({'hour':h,'emergency_kwh':float(u[k].sum()),'cost':float((5*u[k]*pr[k]).sum()),'n':int(urgent[k].sum())})
    examples=[]
    for j in np.argsort(u)[-8:][::-1]:
        at=int(t[j]);a=int(decision[j]);idx=int(i[j]);offset=(at-a)//10
        examples.append({'at':(datetime(2025,1,1)+timedelta(minutes=at)).isoformat(),
            'decision_at':(datetime(2025,1,1)+timedelta(minutes=a)).isoformat(),
            'demand':float(d[j]),'pv':float(pv[j]),'grid':float(g[j]),'emergency':float(u[j]),
            'soc_start':float(soc[j]),'soc_end':float(z['soc_boundary_kwh'][idx+1]),'discharge':float(dis[j]),
            'initial_plan':float(z['plan_initial_kwh'][idx]),'original_exec':float(z['plan_exec_kwh'][idx]),'add_exec':float(z['add_exec_kwh'][idx]),
            'node_demand_pred':float(legal[j,0]),'node_pv_pred':float(legal[j,1]),'rolling_demand_pred':float(pred[j,0]),'rolling_pv_pred':float(pred[j,1]),
            'node_net_upper':float(audit[a]['net_upper_path_kwh'][offset]),
            'node_reserve_power_slack':float(audit[a]['reserve_power_shortfall_path_kwh'][offset]),
            'reserve_current':float(audit[at]['reserve_required_kwh'])})
    reports[problem]={'totals':payload['current'],'forecast':metrics,'hourly_emergency':hours,'largest_emergencies':examples,
        'physical_emergency_formula_max_error':float(abs(u-theoretical).max()),
        'power_only_shortage_kwh':float(power.sum()),'additional_stock_shortage_kwh':float((theoretical-power).sum()),
        'urgent_discharge_at_750_count':int((urgent & (dis>=750-1e-5)).sum()),
        'urgent_soc_end_at_min_count':int((urgent & (z['soc_boundary_kwh'][i+1]<=1200+1e-5)).sum()),
        'soc_mean':float(soc.mean()),'reserve_mean':float(np.mean([audit[int(m)]['reserve_required_kwh'] for m in t])),
        'decision_node_urgent_count':int((urgent & (t%360==0)).sum()),'decision_node_urgent_kwh':float(u[t%360==0].sum())}
    np.savez_compressed(root/f'{problem}_comparison.npz',t=t,d=d,pv=pv,g=g,u=u,price=pr,soc=soc,pred=pred,legal=legal)
diff=reports['problem3']['totals']['total_cost']-reports['problem2']['totals']['total_cost']
reports['cost_decomposition']={'total_difference':diff,
    'ordinary_execution_difference':reports['problem3']['totals']['plan_cost']+reports['problem3']['totals']['add_cost']-reports['problem2']['totals']['ordinary_cost'],
    'emergency_difference':reports['problem3']['totals']['emergency_cost']-reports['problem2']['totals']['emergency_cost'],
    'penalty_difference':reports['problem3']['totals']['penalty_cost']}
(root/'diagnosis.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(reports,ensure_ascii=True,indent=2))
