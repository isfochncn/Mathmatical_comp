"""Reconcile old/new realized bills and energy under identical source periods."""
from pathlib import Path
import argparse
import json
import numpy as np


def read_run(folder,problem):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    state=read(folder/problem/'status.json');assert state['state'] in ('simulation_complete','complete')
    run=Path(state['run_dir']);z=dict(np.load(run/'trajectory.npz'));ledger=read(run/'settlement_events.json')
    penalty=np.zeros(len(z['abs_minute']))
    for p in ledger['penalties']:penalty[(p['at_abs']-10)//10]+=.5*p['price_at_yuan_per_kwh']*sum(p['reduced_kwh'])
    z['penalty_fee']=penalty
    return z,state['summary'],read(run/'config.json')


def metrics(z,mask):
    i=np.flatnonzero(mask);soc=z['soc_boundary_kwh'];p=z['price_actual'][mask]
    base=float(np.dot(p,z['grid_kwh'][mask]));premium=float(np.dot(.5*p,z['add_exec_kwh'][mask]))
    urgent_fee=float(np.dot(5*p,z['emergency_kwh'][mask]));penalty=float(z['penalty_fee'][mask].sum())
    charge=z['charge_kwh'][mask].sum();dis=z['discharge_kwh'][mask].sum()
    loss=float(charge/.9-charge+dis/.9-dis);stock=float(soc[i[-1]+1]-soc[i[0]])
    purchased=float((z['grid_kwh'][mask]+z['emergency_kwh'][mask]).sum())
    load=float(z['demand_kwh'][mask].sum());pv=float(z['pv_kwh'][mask].sum())
    curtail=float(z['curtail_kwh'][mask].sum());spill=float(z['surplus_kwh'][mask].sum())
    assert abs(purchased-(load-pv+curtail+loss+stock+spill))<1e-4
    return dict(intervals=len(i),total_cost_yuan=base+premium+urgent_fee+penalty,
        base_ordinary_cost_yuan=base,adjustment_premium_yuan=premium,emergency_cost_yuan=urgent_fee,penalty_yuan=penalty,
        initial_plan_kwh=float(z['plan_initial_kwh'][mask].sum()),ordinary_kwh=float(z['grid_kwh'][mask].sum()),
        adjustment_kwh=float(z['add_exec_kwh'][mask].sum()),emergency_kwh=float(z['emergency_kwh'][mask].sum()),
        emergency_intervals=int(np.sum(z['emergency_kwh'][mask]>1e-5)),purchased_kwh=purchased,
        spill_kwh=spill,pv_curtail_kwh=curtail,pv_use_rate=(pv-curtail)/pv if pv else None,
        loss_kwh=loss,soc_start_kwh=float(soc[i[0]]),soc_end_kwh=float(soc[i[-1]+1]),stock_change_kwh=stock)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--candidate',required=True)
    args=ap.parse_args();oldroot=Path(args.baseline).resolve();newroot=Path(args.candidate).resolve();result={}
    for problem in ('problem4-2','problem4-3'):
        old,os,oc=read_run(oldroot,problem);new,ns,nc=read_run(newroot,problem)
        for k in ('abs_minute','demand_kwh','pv_kwh','price_actual'):np.testing.assert_array_equal(old[k],new[k])
        assert old['soc_boundary_kwh'][0]==new['soc_boundary_kwh'][0]==6000
        assert oc['run_from']==nc['run_from'] and oc['run_to']==nc['run_to']
        nreport=ns['n_intervals'];reportmask=np.arange(len(new['abs_minute']))>=len(new['abs_minute'])-nreport
        sections={}
        for label,mask in [('whole_run',np.ones(len(reportmask),dtype=bool)),('report',reportmask),('warmup',~reportmask)]:
            a,b=metrics(old,mask),metrics(new,mask)
            if label=='report':
                assert abs(a['total_cost_yuan']-os['total_cost_yuan'])<1e-4
                assert abs(b['total_cost_yuan']-ns['total_cost_yuan'])<1e-4
            keys=('base_ordinary_cost_yuan','adjustment_premium_yuan','emergency_cost_yuan','penalty_yuan')
            delta={k:b[k]-a[k] for k in keys};cost_change=b['total_cost_yuan']-a['total_cost_yuan']
            assert abs(sum(delta.values())-cost_change)<1e-5
            energy_delta={k:b[k]-a[k] for k in ('pv_curtail_kwh','spill_kwh','loss_kwh','stock_change_kwh')}
            assert abs(sum(energy_delta.values())-(b['purchased_kwh']-a['purchased_kwh']))<1e-4
            sections[label]=dict(baseline=a,candidate=b,saving_yuan=-cost_change,saving_fraction=-cost_change/a['total_cost_yuan'],
                                 fee_change_components=delta,purchase_change_components_kwh=energy_delta)
        result[problem]=sections
    payload=dict(baseline_directory=str(oldroot),candidate_directory=str(newroot),identical_source_arrays_verified=True,
        comparisons=result,notes='Development-period realized-cost comparison. Report starting stocks differ; whole runs share initial SOC. Risk calibration is not independently validated.')
    (newroot/'经济性对比.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({p:{s:{k:v for k,v in x.items() if k in ('saving_yuan','saving_fraction')} for s,x in r.items()} for p,r in result.items()},indent=2))


if __name__=='__main__':main()
