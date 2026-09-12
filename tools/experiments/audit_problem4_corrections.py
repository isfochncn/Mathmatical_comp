"""Audit v3/v4/v5 correction activation and continuous warmup/report periods."""
from pathlib import Path
from datetime import date
import argparse
import json
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    status=read(out/'status.json');assert status['state'] in ('simulation_complete','complete')
    run=Path(status['run_dir']);config=read(run/'config.json');summary=read(run/'summary.json')['summary']
    z=dict(np.load(run/'trajectory.npz'));audits=read(run/'forecast_audit.json')
    ledger=read(run/'settlement_events.json');contract=read(out/'run_contract.json')
    version=summary['forecast_policy_version']
    assert version in ('report-cumulative-guard-v3','report-economic-reserve-v4','report-economic-reserve-v5')
    reserve_policy={'report-economic-reserve-v5':'economic-cumulative-v3','report-economic-reserve-v4':'economic-cumulative-v2'}.get(version,'cumulative-stress-v1')
    assert contract['forecast_class']=='CumulativeReportForecaster' and contract['price_mode']=='historical'
    minutes=z['abs_minute'];assert len(audits)==len(minutes)
    guards=[];paths=0
    for i,a in enumerate(audits):
        now=int(minutes[i]);assert a['formed_at']==now and a['observed_end']==now
        assert a['reserve_policy']==reserve_policy and 'report-v2=blend' in a['source']
        if version=='report-economic-reserve-v4':assert a['stress_weight']==.2
        if version=='report-economic-reserve-v5':
            assert a['stress_weight']==(.5 if config['problem']=='problem4-2' and now<7*1440+10 else .2)
        legal=config['problem']=='problem4-3' and now%1440 in (360,720,1080)
        g=a['reduction_guard'];assert bool(g)==legal
        if g:
            assert g['until_abs']==(now//360+1)*360
            if g['checked']:
                if g['blocked']:
                    assert g['candidate_stress_emergency_kwh']>g['original_stress_emergency_kwh']+1e-5
                    assert g['restored_stress_emergency_kwh']<=g['original_stress_emergency_kwh']+1e-4
                    assert g['restored_kwh']>0
                else:
                    assert g['candidate_stress_emergency_kwh']<=g['original_stress_emergency_kwh']+1e-5
            guards.append(dict(at_abs=now,**g))
        if 'stress_soc_path_kwh' in a:
            paths+=1;soc=np.array(a['stress_soc_path_kwh']);d=np.array(a['demand_forecast_path_kwh'])
            assert len(soc)==len(d)+1 and min(soc)>=1200-1e-6 and max(soc)<=10800+1e-6
            assert abs(soc[0]-z['soc_boundary_kwh'][i])<1e-6
            assert np.all(np.array(a['stress_demand_path_kwh'])>=d-1e-6)
    fees=z['price_actual']*(z['plan_exec_kwh']+1.5*z['add_exec_kwh']+5*z['emergency_kwh'])
    for p in ledger['penalties']:fees[(p['at_abs']-10)//10]+=.5*p['price_at_yuan_per_kwh']*sum(p['reduced_kwh'])
    start=(date.fromisoformat(config['run_from'])-date(2025,1,1)).days*1440+10
    sections={}
    for label,mask in [('warmup',minutes<start),('report',minutes>=start)]:
        idx=np.flatnonzero(mask);assert len(idx)>0
        aa=[audits[i] for i in idx];gg=[a['reduction_guard'] for a in aa if a['reduction_guard']]
        demand_err=np.array([a['demand_forecast_kwh'] for a in aa])-z['demand_kwh'][mask]
        pv_err=np.array([a['pv_forecast_kwh'] for a in aa])-z['pv_kwh'][mask]
        section=dict(n_intervals=int(mask.sum()),start_abs=int(minutes[idx[0]]),end_abs=int(minutes[idx[-1]]+10),
            initial_plan_kwh=float(z['plan_initial_kwh'][mask].sum()),ordinary_kwh=float(z['grid_kwh'][mask].sum()),
            added_kwh=float(z['add_exec_kwh'][mask].sum()),emergency_kwh=float(z['emergency_kwh'][mask].sum()),
            emergency_intervals=int(np.sum(z['emergency_kwh'][mask]>1e-5)),
            emergency_cost_yuan=float(np.dot(5*z['price_actual'][mask],z['emergency_kwh'][mask])),
            total_cost_yuan=float(fees[mask].sum()),spill_kwh=float(z['surplus_kwh'][mask].sum()),
            soc_start_kwh=float(z['soc_boundary_kwh'][idx[0]]),soc_end_kwh=float(z['soc_boundary_kwh'][idx[-1]+1]),
            soc_min_kwh=float(z['soc_boundary_kwh'][idx[0]:idx[-1]+2].min()),soc_max_kwh=float(z['soc_boundary_kwh'][idx[0]:idx[-1]+2].max()),
            guard_nodes=len(gg),guard_checks=sum(g['checked'] for g in gg),guard_blocks=sum(g['blocked'] for g in gg),
            guard_restored_kwh=sum(g['restored_kwh'] for g in gg),cold_start_windows=sum(a['cold_start_margin'] for a in aa),
            load_mae_kwh=float(abs(demand_err).mean()),pv_mae_kwh=float(abs(pv_err).mean()),
            load_bias_kwh_per_interval=float(demand_err.mean()),pv_bias_kwh_per_interval=float(pv_err.mean()))
        sections[label]=section
    assert abs(sections['warmup']['soc_end_kwh']-sections['report']['soc_start_kwh'])<1e-7
    assert abs(sections['report']['total_cost_yuan']-summary['total_cost_yuan'])<1e-4
    result=dict(problem=config['problem'],version=summary['forecast_policy_version'],all_correction_audits_passed=True,
        stress_paths_checked=paths,continuous_soc_verified=True,sections=sections,guard_events=guards,
        notes='Guard guarantees no worsening only under the current deterministic stress before the next legal node. Risk penalties are excluded from bills.')
    dest=out/'核验/correction_checks.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='guard_events'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
