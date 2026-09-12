"""Build February reporting rows from the continuous problem-three dispatch ledger."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import csv
import hashlib
import json
import numpy as np


ORIGIN = datetime(2025, 1, 1)


def stamp(minute):
    return (ORIGIN + timedelta(minutes=int(minute))).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state'] in ('simulation_complete', 'complete')
    run_dir = Path(status['run_dir'])
    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    assert config['problem'] == 'problem3' and config['load_method'] == 'adaptive'
    assert config['plan_refresh_intervals'] == 1 and config['risk_quantile'] == .9
    z = dict(np.load(run_dir/'trajectory.npz'))
    first, last = (datetime.fromisoformat(config[k]) for k in ('run_from','run_to'))
    lo = int((first-ORIGIN).total_seconds()/60)+10
    hi = int((last+timedelta(days=1)-ORIGIN).total_seconds()/60)+10
    mask = (z['abs_minute'] >= lo) & (z['abs_minute'] < hi)
    indices = np.flatnonzero(mask)
    np.testing.assert_array_equal(z['abs_minute'][mask], np.arange(lo, hi, 10))
    np.testing.assert_array_equal(z['abs_minute'], np.arange(10, hi, 10))
    assert z['soc_boundary_kwh'][0] == 6000
    np.testing.assert_allclose(z['grid_kwh'], z['plan_exec_kwh']+z['add_exec_kwh'], atol=1e-7, rtol=0)
    ledger = json.loads((run_dir/'settlement_events.json').read_text(encoding='utf-8'))
    execution = {e['at_abs']:e for e in ledger['execution']}
    penalties = {}
    penalty_rows, penalty_ranges = [], {}
    for event in sorted(ledger['penalties'], key=lambda e:e['at_abs']):
        at = int(event['at_abs'])
        assert at % 1440 in (360, 720, 1080)
        reduced = np.asarray(event['reduced_kwh'], dtype=float)
        targets = np.asarray(event['reduced_abs'], dtype=int)
        assert np.all(targets[reduced > 0] >= at)
        cost = float(.5*event['price_at_yuan_per_kwh']*reduced.sum())
        qty, fee = penalties.get(at, (0., 0.))
        penalties[at] = (qty+float(reduced.sum()), fee+cost)
        if lo <= at < hi:
            begin_row = len(penalty_rows)+5
            for target, amount in zip(targets, reduced):
                if amount > 0:
                    price = float(event['price_at_yuan_per_kwh'])
                    penalty_rows.append([stamp(at),stamp(target),float(amount),price,.5*price,.5*price*float(amount)])
            if len(penalty_rows)+4 >= begin_row:
                if at in penalty_ranges:
                    penalty_ranges[at][1] = len(penalty_rows)+4
                else:
                    penalty_ranges[at] = [begin_row,len(penalty_rows)+4]
    audits = {int(a['formed_at']):a for a in json.loads((run_dir/'forecast_audit.json').read_text(encoding='utf-8'))}
    rows, checks = [], []
    for local,i in enumerate(indices):
        minute = int(z['abs_minute'][i])
        value = lambda key:float(z[key][i])
        plan, original, add, grid, urgent, price = (value(k) for k in (
            'plan_initial_kwh','plan_exec_kwh','add_exec_kwh','grid_kwh','emergency_kwh','price_actual'))
        event = execution[minute]
        np.testing.assert_allclose([original,add,urgent,price], [event[k] for k in (
            'o_exec_kwh','a_exec_kwh','emergency_kwh','price_actual_yuan_per_kwh')], atol=1e-7, rtol=0)
        cut, penalty = penalties.get(minute,(0.,0.))
        fee = price*(original+1.5*add+5*urgent)+penalty
        rows.append([stamp(minute),stamp(minute+10),(ORIGIN+timedelta(minutes=minute-10)).date().isoformat(),
            local%144+1,plan,original,add,grid,grid-plan,urgent,grid+urgent,price,
            original*price,1.5*price,1.5*add*price,5*price,5*urgent*price,cut,penalty,fee,f'{fee:.2f}元',
            value('surplus_kwh'),grid+urgent-value('surplus_kwh')])
        plan_at = minute//1440*1440
        a, initial = audits[minute],audits[plan_at]
        offset = (minute-plan_at)//10
        planned_d = initial['demand_forecast_path_kwh'][offset]
        planned_p = initial['pv_forecast_path_kwh'][offset]
        bus = grid+urgent+value('pv_kwh')-value('curtail_kwh')+value('discharge_kwh')-value('demand_kwh')-value('charge_kwh')/.9-value('surplus_kwh')
        stock = float(z['soc_boundary_kwh'][i+1]-z['soc_boundary_kwh'][i])-value('charge_kwh')+value('discharge_kwh')/.9
        checks.append([stamp(minute),stamp(plan_at),planned_d,planned_p,value('demand_kwh'),value('pv_kwh'),
            value('demand_kwh')-value('pv_kwh'),value('demand_kwh')-value('pv_kwh')-planned_d+planned_p,
            a['demand_forecast_kwh'],a['pv_forecast_kwh'],
            value('demand_kwh')-value('pv_kwh')-a['demand_forecast_kwh']+a['pv_forecast_kwh'],
            float(z['soc_boundary_kwh'][i]),float(z['soc_boundary_kwh'][i+1]),value('charge_kwh'),value('discharge_kwh'),
            value('curtail_kwh'),value('surplus_kwh'),a['reserve_required_kwh'],a['reserve_shortfall_kwh'],
            a['reserve_power_shortfall_now_kwh'],bus,stock])
    def total(col):
        return float(sum(r[col] for r in rows))
    current = {key:total(col) for key,col in {
        'plan_kwh':4,'retained_kwh':5,'add_kwh':6,'ordinary_kwh':7,'net_adjustment_kwh':8,
        'emergency_kwh':9,'total_kwh':10,'plan_cost':12,'add_cost':14,'emergency_cost':16,
        'reduced_kwh':17,'penalty_cost':18,'total_cost':19,'spill_kwh':21}.items()}
    current.update(soc_start=checks[0][11],soc_end=checks[-1][12],
        emergency_intervals=sum(r[9]>1e-6 for r in rows),revised_intervals=sum(abs(r[8])>1e-6 for r in rows),
        penalty_nodes=len(penalty_ranges),spill_intervals=sum(r[21]>1e-6 for r in rows))
    summary = json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))['summary']
    mapping = {'retained_kwh':'plan_kwh','add_kwh':'add_kwh','emergency_kwh':'emergency_kwh',
        'total_kwh':'total_purchased_kwh','plan_cost':'plan_cost_yuan','add_cost':'add_cost_yuan',
        'emergency_cost':'emergency_cost_yuan','penalty_cost':'reduce_cost_yuan','total_cost':'total_cost_yuan'}
    for key,source in mapping.items():
        assert abs(current[key]-summary[source]) < 1e-4,(key,current[key],summary[source])
    assert max(abs(r[-2]) for r in checks) < 1e-5
    assert max(abs(r[-1]) for r in checks) < 1e-5
    assert abs(sum(r[5] for r in penalty_rows)-current['penalty_cost']) < 1e-5
    warm_indices = np.flatnonzero(z['abs_minute'] < lo)
    warm_cost = sum(e['price_actual_yuan_per_kwh']*(e['o_exec_kwh']+1.5*e['a_exec_kwh']+5*e['emergency_kwh'])
                    for e in ledger['execution'] if e['at_abs'] < lo)
    warm_cost += sum(cost for at,(_,cost) in penalties.items() if at < lo)
    warmup = {'from':stamp(10),'to':stamp(lo),'n':len(warm_indices),'soc_start':6000.,
              'soc_end':float(z['soc_boundary_kwh'][indices[0]]),'total_cost':warm_cost}
    purchase_headers = ['开始时间','结束时间','归属日期','日内时段','日初计划购电_kWh','保留原计划执行_kWh',
        '调整增购执行_kWh','调整后普通购电_kWh','相对日初净调整_kWh','紧急购电_kWh','实际外网总购电_kWh',
        '普通电价_元每kWh','原计划执行费_元','调整增购单价_元每kWh','调整增购费_元',
        '紧急单价_元每kWh','紧急购电费_元','本时刻调减总量_kWh','本时刻违约费_元',
        '本段总费用_元','费用标签','弃购电_kWh','购电利用量_kWh']
    check_headers = ['开始时间','日计划制定时间','计划负荷预测_kWh','计划光伏预测_kWh','实际负荷_kWh',
        '实际光伏_kWh','实际净负荷_kWh','计划净负荷误差_kWh','执行时负荷预测_kWh','执行时光伏预测_kWh',
        '执行时净负荷误差_kWh','段初SOC_kWh','段末SOC_kWh','实际存入_kWh','实际送达_kWh',
        '弃光_kWh','弃购电_kWh','当段累计备用需求_送达kWh','窗口最大规划备用缺口_kWh',
        '当段规划供电缺口_kWh','母线平衡残差_kWh','SOC递推残差_kWh']
    penalty_headers = ['调减发生时间','被调减的交付段起点','调减电量_kWh','发生时电价_元每kWh','违约单价_元每kWh','违约费_元']
    payload = {'problem':'problem3','month':status['month'],'from':stamp(lo),'to':stamp(hi),'n':len(rows),
        'execution_start':stamp(10),'rows':rows,'check_rows':checks,'penalty_rows':penalty_rows,
        'penalty_ranges':{stamp(k):v for k,v in penalty_ranges.items()},'current':current,'warmup':warmup,
        'purchase_headers':purchase_headers,'check_headers':check_headers,'penalty_headers':penalty_headers,
        'rules':{'eta_charge':.9,'eta_discharge':.9,'add_multiplier':1.5,'reduce_multiplier':.5,
            'emergency_multiplier':5,'risk_quantile':.9,'interval_minutes':10,'soc_min':1200,'soc_max':10800,'initial_soc':6000}}
    for name,headers,table in [('逐时购电与费用.csv',purchase_headers,rows),('供电与预测核验.csv',check_headers,checks),
                               ('调减违约明细.csv',penalty_headers,penalty_rows)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(headers);writer.writerows(table)
    (out/'month_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    excluded = ('rows','check_rows','penalty_rows','penalty_ranges')
    (out/'month_summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in excluded},ensure_ascii=False,indent=2),encoding='utf-8')
    hashes=json.loads((out/'source_and_template_hashes.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({'current':current,'warmup':warmup,'n':len(rows),'penalty_rows':len(penalty_rows)},indent=2))


if __name__ == '__main__':
    main()
