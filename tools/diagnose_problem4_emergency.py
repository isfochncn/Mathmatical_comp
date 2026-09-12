"""Read archived two-month runs; decompose urgent purchases without rerunning policy."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/问题四_1至2月检验_20260912_165702/紧急购电诊断'
OUT.mkdir(exist_ok=True)
ORIGIN = datetime(2025, 1, 1)
def stamp(t): return (ORIGIN+timedelta(minutes=int(t))).isoformat(sep=' ')
def total(a,m): return float(a[m].sum())

def analyze(problem):
    p=OUT.parent/problem/'运行记录'/problem
    z=np.load(p/'trajectory.npz')
    a=json.loads((p/'forecast_audit.json').read_text(encoding='utf-8'))
    raw=json.loads((p/'summary.json').read_text(encoding='utf-8'))
    t=z['abs_minute'];u=z['emergency_kwh'];n=len(t)
    assert len(a)==n and np.array_equal(t,[r['formed_at'] for r in a])
    soc=z['soc_boundary_kwh'][:-1];net=z['demand_kwh']-z['pv_kwh'];g=z['grid_kwh']
    # Exact minimum urgent purchase with fixed purchases and interval-start SOC.
    power=np.maximum(net-g-750,0)
    need=np.maximum(net-g-np.minimum(750,.9*(soc-1200)),0)
    assert np.max(np.abs(need-u))<1e-6
    energy=np.maximum(need-power,0)
    decision_load=np.zeros(n);decision_pv=np.zeros(n);upper=np.zeros(n);samples=np.zeros(n)
    reserve=np.zeros(n);shortfall=np.zeros(n);formed=np.zeros(n,dtype=int)
    for i,r in enumerate(a):
        if 'demand_forecast_path_kwh' in r:
            last=r
        j=(int(t[i])-last['formed_at'])//10
        decision_load[i]=last['demand_forecast_path_kwh'][j]
        decision_pv[i]=last['pv_forecast_path_kwh'][j]
        upper[i]=last['net_upper_path_kwh'][j]
        reserve[i]=last['reserve_path_kwh'][j]
        shortfall[i]=last['reserve_power_shortfall_path_kwh'][j]
        samples[i]=last['risk_sample_count'];formed[i]=last['formed_at']
    loaderr=z['demand_kwh']-decision_load;pverr=decision_pv-z['pv_kwh']
    neterr=loaderr+pverr;curload=np.array([r['demand_forecast_kwh'] for r in a]);curpv=np.array([r['pv_forecast_kwh'] for r in a])
    def detail(i):
        return dict(time=stamp(t[i]),formed_at=stamp(formed[i]),urgent=float(u[i]),grid=float(g[i]),initial_plan=float(z['plan_initial_kwh'][i]),
            load=float(z['demand_kwh'][i]),pv=float(z['pv_kwh'][i]),soc_start=float(soc[i]),soc_end=float(z['soc_boundary_kwh'][i+1]),
            discharge=float(z['discharge_kwh'][i]),power_floor=float(power[i]),energy_extra=float(energy[i]),
            decision_load=float(decision_load[i]),decision_pv=float(decision_pv[i]),load_error=float(loaderr[i]),pv_error=float(pverr[i]),
            current_load=float(curload[i]),current_pv=float(curpv[i]),net_upper=float(upper[i]),reserve_energy=float(reserve[i]),
            decision_power_slack=float(shortfall[i]),risk_samples=int(samples[i]),price=float(z['price_actual'][i]))
    out={'problem':problem,'initial_soc':float(soc[0]),'source_max_urgent_reconstruction_error':float(np.max(np.abs(need-u))),'periods':{}}
    for name,m in [('January',(t>=10)&(t<44650)),('February',t>=44650),('Both',t>=10)]:
        hits=m&(u>1e-6);morning=m&(t%1440>=540)&(t%1440<660)
        # Keep the result-row convention used in the delivered workbooks.
        rows=[]
        for d in np.unique((t[m]-10)//1440):
            dm=m&((t-10)//1440==d)
            rows.append({'date':stamp(d*1440)[:10],'urgent_kwh':total(u,dm),'power_floor_kwh':total(power,dm),'energy_extra_kwh':total(energy,dm)})
        out['periods'][name]=dict(intervals=int(m.sum()),urgent_kwh=total(u,m),urgent_intervals=int(hits.sum()),
            urgent_fee_yuan=total(5*u*z['price_actual'],m),ordinary_kwh=total(g,m),
            power_floor_kwh=total(power,m),energy_extra_kwh=total(energy,m),
            morning_9_11_kwh=total(u,morning),morning_load_error_kwh=total(loaderr,morning),morning_pv_error_kwh=total(pverr,morning),
            morning_decision_pv_kwh=total(decision_pv,morning),morning_actual_pv_kwh=total(z['pv_kwh'],morning),
            morning_decision_load_kwh=total(decision_load,morning),morning_actual_load_kwh=total(z['demand_kwh'],morning),
            urgent_with_soc_above_5000=total(u,hits&(soc>5000)),urgent_with_discharge_750=total(u,hits&(z['discharge_kwh']>749.999)),
            urgent_with_zero_grid=total(u,hits&(g<1e-6)),urgent_with_net_above_upper=total(u,hits&(net>upper+1e-6)),
            urgent_with_decision_power_slack=total(u,hits&(shortfall>1e-5)),urgent_with_less7_samples=total(u,hits&(samples<7)),
            hours=[{'hour':h,'urgent':total(u,m&(t//60%24==h))} for h in range(24)],
            top_days=sorted(rows,key=lambda r:-r['urgent_kwh'])[:8],
            top_intervals=[detail(i) for i in np.flatnonzero(m)[np.argsort(u[m])[-5:][::-1]]])
    # A February morning with repeated large urgent purchases.
    chosen=(t>=((datetime(2025,2,12)-ORIGIN).days*1440+480))&(t<((datetime(2025,2,12)-ORIGIN).days*1440+720))
    out['feb12_8_to_12']=[detail(i) for i in np.flatnonzero(chosen)]
    # Quantify current forecasts, not just the old decision forecast, at urgent events.
    out['urgent_current_forecast_errors']={k:float(v[t>=44650].sum()) for k,v in {
        'load_underprediction':np.where(u>1e-6,z['demand_kwh']-curload,0),
        'pv_overprediction':np.where(u>1e-6,curpv-z['pv_kwh'],0)}.items()}
    out['cold_start']={'first_seven_result_days_urgent_kwh':float(u[t<10090].sum()),'examples':[]}
    for day in (1,5):
        start=max(10,(day-1)*1440);i=(start-10)//10;r=a[i];length=(1440-start%1440)//10
        out['cold_start']['examples'].append({'day':day,'samples':r['risk_sample_count'],
            'load_forecast':sum(r['demand_forecast_path_kwh'][:length]),'load_actual':float(z['demand_kwh'][i:i+length].sum()),
            'pv_forecast':sum(r['pv_forecast_path_kwh'][:length]),'pv_actual':float(z['pv_kwh'][i:i+length].sum())})
    return out

if __name__=='__main__':
    results=[analyze(p) for p in ('problem4-2','problem4-3')]
    (OUT/'diagnosis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:
        print(r['problem'])
        for name,m in r['periods'].items():print(name,json.dumps({k:v for k,v in m.items() if k not in ('hours','top_intervals','top_days')},ensure_ascii=True))
        print('February top',json.dumps(r['periods']['February']['top_intervals'][:2],ensure_ascii=True))
        print('Jan days',json.dumps(r['periods']['January']['top_days'],ensure_ascii=True))
