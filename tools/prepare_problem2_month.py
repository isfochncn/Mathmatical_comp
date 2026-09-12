"""Prepare auditable month rows after a continuous main-model run completes."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import csv
import json
import hashlib
import numpy as np


def month_totals(z, mask):
    grid, urgent, price = (z[k][mask] for k in ('grid_kwh','emergency_kwh','price_actual'))
    indices = np.flatnonzero(mask)
    return {'plan_kwh':float(z['plan_initial_kwh'][mask].sum()),
            'ordinary_kwh':float(grid.sum()), 'emergency_kwh':float(urgent.sum()),
            'total_kwh':float((grid+urgent).sum()), 'ordinary_cost':float(np.dot(grid,price)),
            'emergency_cost':float(5*np.dot(urgent,price)), 'total_cost':float(np.dot(grid+5*urgent,price)),
            'spill_kwh':float(z['surplus_kwh'][mask].sum()),
            'soc_start':float(z['soc_boundary_kwh'][indices[0]]),
            'soc_end':float(z['soc_boundary_kwh'][indices[-1]+1]),
            'emergency_intervals':int(np.sum(urgent > 1e-6)),
            'revised_intervals':int(np.sum(np.abs(grid-z['plan_initial_kwh'][mask]) > 1e-6))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    status = json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert status['state'] in ('simulation_complete','complete')
    run_dir = Path(status['run_dir'])
    z = dict(np.load(run_dir/'trajectory.npz'))
    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    assert config['problem'] == 'problem2' and config['load_method'] == 'adaptive'
    origin = datetime(2025,1,1)
    first = datetime.fromisoformat(config['run_from'])
    last = datetime.fromisoformat(config['run_to'])
    lo = int((first-origin).total_seconds()/60)+10
    hi = int((last+timedelta(days=1)-origin).total_seconds()/60)+10
    mask = (z['abs_minute'] >= lo) & (z['abs_minute'] < hi)
    indices = np.flatnonzero(mask)
    assert len(indices) == (last-first).days*144+144
    np.testing.assert_array_equal(z['abs_minute'][mask], np.arange(lo,hi,10))
    assert z['abs_minute'][0] == 10 and z['soc_boundary_kwh'][0] == 6000
    np.testing.assert_allclose(z['grid_kwh'][mask], z['plan_initial_kwh'][mask], atol=1e-7, rtol=0)
    assert not np.any(z['add_exec_kwh'][mask])
    audits = {int(a['formed_at']):a for a in json.loads((run_dir/'forecast_audit.json').read_text(encoding='utf-8'))}
    current = month_totals(z, mask)
    summary = json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))['summary']
    assert abs(current['total_cost']-summary['total_cost_yuan']) < 1e-5
    legacy_root = Path(Path('out/current_annual_output.txt').read_text(encoding='utf-8-sig').strip())
    old = dict(np.load(legacy_root/'运行记录/problem2/trajectory.npz'))
    old_mask = (old['abs_minute'] >= lo) & (old['abs_minute'] < hi)
    np.testing.assert_array_equal(old['abs_minute'][old_mask], z['abs_minute'][mask])
    legacy = month_totals(old, old_mask)
    rows, checks = [], []
    for local, i in enumerate(indices):
        minute = int(z['abs_minute'][i])
        begin = origin+timedelta(minutes=minute)
        finish = begin+timedelta(minutes=10)
        row_day = (begin-timedelta(minutes=10)).date().isoformat()
        plan_at = minute//1440*1440
        a, initial = audits[minute], audits[plan_at]
        offset = (minute-plan_at)//10
        planned_d = initial['demand_forecast_path_kwh'][offset]
        planned_p = initial['pv_forecast_path_kwh'][offset]
        value = lambda key:float(z[key][i])
        normal, urgent, price = value('grid_kwh'),value('emergency_kwh'),value('price_actual')
        fee = price*(normal+5*urgent)
        rows.append([begin.isoformat(),finish.isoformat(),row_day,local%144+1,
            value('plan_initial_kwh'),normal,normal-value('plan_initial_kwh'),urgent,normal+urgent,price,
            normal*price,5*price,urgent*5*price,fee,f'{fee:.2f}元',value('surplus_kwh'),normal+urgent-value('surplus_kwh')])
        bus = normal+urgent+value('pv_kwh')-value('curtail_kwh')+value('discharge_kwh')-value('demand_kwh')-value('charge_kwh')/.9-value('surplus_kwh')
        stock = float(z['soc_boundary_kwh'][i+1]-z['soc_boundary_kwh'][i])-value('charge_kwh')+value('discharge_kwh')/.9
        checks.append([begin.isoformat(),(origin+timedelta(minutes=plan_at)).isoformat(),
            planned_d,planned_p,value('demand_kwh'),value('pv_kwh'),value('demand_kwh')-value('pv_kwh'),
            value('demand_kwh')-value('pv_kwh')-planned_d+planned_p,
            a['demand_forecast_kwh'],a['pv_forecast_kwh'],
            value('demand_kwh')-value('pv_kwh')-a['demand_forecast_kwh']+a['pv_forecast_kwh'],
            float(z['soc_boundary_kwh'][i]),float(z['soc_boundary_kwh'][i+1]),
            value('charge_kwh'),value('discharge_kwh'),value('curtail_kwh'),value('surplus_kwh'),
            a['reserve_required_kwh'],a['reserve_shortfall_kwh'],a['reserve_power_shortfall_now_kwh'],bus,stock])
    assert max(abs(r[-2]) for r in checks) < 1e-5
    assert max(abs(r[-1]) for r in checks) < 1e-5
    purchase_headers = ['开始时间','结束时间','归属日期','日内时段','计划购电_kWh','实际普通购电_kWh',
        '普通调整差额_kWh','紧急购电_kWh','实际外网总购电_kWh','普通电价_元每kWh','普通电费_元',
        '紧急单价_元每kWh','紧急电费_元','本段总电费_元','电费标签','弃购电_kWh','购电利用量_kWh']
    check_headers = ['开始时间','日计划制定时间','计划负荷预测_kWh','计划光伏预测_kWh','实际负荷_kWh',
        '实际光伏_kWh','实际净负荷_kWh','计划净负荷误差_kWh','执行时负荷预测_kWh','执行时光伏预测_kWh',
        '执行时净负荷误差_kWh','段初SOC_kWh','段末SOC_kWh','实际存入_kWh','实际送达_kWh','弃光_kWh',
        '弃购电_kWh','当段累计备用需求_送达kWh','窗口最大规划备用缺口_kWh','当段规划供电缺口_kWh',
        '母线平衡残差_kWh','SOC递推残差_kWh']
    for name, headers, table in [('逐时购电与费用.csv',purchase_headers,rows),('供电与预测核验.csv',check_headers,checks)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(headers); writer.writerows(table)
    payload = {'month':status['month'],'from':rows[0][0],'to':rows[-1][1], 'n':len(rows),
        'execution_start':(origin+timedelta(minutes=10)).isoformat(),
        'rows':rows,'check_rows':checks,'current':current,'legacy':legacy,
        'purchase_headers':purchase_headers,'check_headers':check_headers,
        'rules':{'eta_charge':.9,'eta_discharge':.9,'emergency_multiplier':5,'risk_quantile':config['risk_quantile'],
                 'interval_minutes':10,'soc_min':1200,'soc_max':10800,'discharge_max_kwh':750,'initial_soc':6000}}
    (out/'month_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    (out/'month_summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ('rows','check_rows')},ensure_ascii=False,indent=2),encoding='utf-8')
    for path, digest in json.loads((out/'source_and_template_hashes.json').read_text(encoding='utf-8')).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    print(json.dumps({'current':current,'legacy':legacy,'rows':len(rows)},indent=2))


if __name__ == '__main__':
    main()
