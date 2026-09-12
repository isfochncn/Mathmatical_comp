"""Compare like-for-like February ledgers, preserving original baseline summaries."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--old-problem3', default='outputs/问题三_1至2月检验_20260912_120602')
    ap.add_argument('--problem2', default='outputs/问题二_2月检验_20260912_111238')
    ap.add_argument('--prediction-reference', default='outputs/光伏预报优化_1至2月_20260912_130151')
    args = ap.parse_args()
    out = Path(args.out)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    current = read(out/'month_summary.json')
    old_path = Path(args.old_problem3)/'month_summary.json'
    p2_path = Path(args.problem2)/'month_summary.json'
    old, p2 = read(old_path), read(p2_path)
    for baseline in (old, p2):
        for key in ('month', 'from', 'to', 'n', 'execution_start'):
            assert baseline[key] == current[key], (key, baseline[key], current[key])
    status = read(out/'status.json')
    run_summary = read(Path(status['run_dir'])/'summary.json')['summary']
    assert run_summary['forecast_policy_version'] == 'report-hourly-blend-risk-v2'
    assert run_summary['pv_forecast_method'] == 'report_blend'
    assert current['warmup']['soc_start'] == old['warmup']['soc_start'] == 6000
    assert current['warmup']['n']+current['n'] == 8496
    reference_path = Path(args.prediction_reference)/'forecast_arrays.npz'
    reference = np.load(reference_path)
    run_dir = Path(status['run_dir'])
    audit = read(run_dir/'forecast_audit.json')
    audit = sorted(audit,key=lambda row:int(row['formed_at']))
    np.testing.assert_array_equal([row['formed_at'] for row in audit],reference['minutes'])
    for field,key in (('pv_forecast_kwh','new_rolling'),('demand_forecast_kwh','load_rolling')):
        np.testing.assert_allclose([row[field] for row in audit],reference[key],atol=1e-8,rtol=0)
    trajectory = np.load(run_dir/'trajectory.npz')
    np.testing.assert_array_equal(trajectory['abs_minute'],reference['minutes'])
    for field,key in (('pv_kwh','actual_pv'),('demand_kwh','actual_load')):
        np.testing.assert_allclose(trajectory[field],reference[key],atol=1e-8,rtol=0)
    first = (np.datetime64(current['from'])-np.datetime64('2025-01-01T00:00:00'))/np.timedelta64(1,'m')
    mask = reference['minutes'] >= first
    prediction_checks = {'all_intervals':len(audit),'dispatch_forecasts_match_selected_test':True,
        'same_actual_load_and_pv':True,
        'february_rolling_pv_mae_kwh':float(np.mean(abs(reference['new_rolling'][mask]-reference['actual_pv'][mask]))),
        'february_rolling_load_mae_kwh':float(np.mean(abs(reference['load_rolling'][mask]-reference['actual_load'][mask])))}
    # No adjustment transactions exist in problem two. Its ordinary fee is all
    # retained original-plan execution, so these mappings preserve definitions.
    p2['current'].update(retained_kwh=p2['current']['ordinary_kwh'], add_kwh=0.,
        net_adjustment_kwh=0., plan_cost=p2['current']['ordinary_cost'], add_cost=0.,
        reduced_kwh=0., penalty_cost=0., penalty_nodes=0)
    changes = {key:{'old':old['current'][key], 'new':value,
                   'difference':value-old['current'][key],
                   'relative_change':(value-old['current'][key])/abs(old['current'][key])
                                      if old['current'][key] else None}
               for key,value in current['current'].items()}
    components = ('plan_cost', 'add_cost', 'emergency_cost', 'penalty_cost')
    assert abs(sum(changes[k]['difference'] for k in components)-changes['total_cost']['difference']) < 1e-5
    old_two = old['warmup']['total_cost']+old['current']['total_cost']
    new_two = current['warmup']['total_cost']+current['current']['total_cost']
    result = {'old_problem3':old, 'problem2':p2, 'new_problem3':current,
        'changes':changes, 'two_month_cost':{'old':old_two, 'new':new_two,
            'difference':new_two-old_two, 'relative_change':new_two/old_two-1},
        'versus_problem2_cost':{'difference':current['current']['total_cost']-p2['current']['total_cost'],
            'relative_change':current['current']['total_cost']/p2['current']['total_cost']-1},
        'prediction_checks':prediction_checks,
        'source_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,p2_path,reference_path)},
        'caveats':['各策略从年初6000 kWh连续运行，二月起末库存不同，未作等库存折算。',
                   '二月属于开发回测，不能作为未参与设计的独立验证或外推全年。']}
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text = ['# 优化光伏后的问题三两月测试', '',
        '从2025年1月1日00:10、6000 kWh连续执行8496个十分钟时段。一月前置，二月检验，二月共4032段，止于3月1日00:10（不含）。', '',
        '| 二月指标 | 旧问题三 | 优化问题三 | 差额 |', '|---|---:|---:|---:|']
    labels = {'plan_kwh':'日初计划购电 / kWh', 'ordinary_kwh':'调整后普通购电 / kWh',
        'add_kwh':'调整增购执行 / kWh', 'emergency_kwh':'紧急购电 / kWh',
        'plan_cost':'原计划执行费 / 元','add_cost':'调整增购费 / 元',
        'emergency_cost':'紧急购电费 / 元','penalty_cost':'调减违约费 / 元',
        'total_cost':'总费用 / 元','spill_kwh':'弃购电 / kWh','soc_start':'月初SOC / kWh','soc_end':'月末SOC / kWh'}
    for key,label in labels.items():
        c = changes[key]
        text.append(f"| {label} | {c['old']:,.2f} | {c['new']:,.2f} | {c['difference']:+,.2f} |")
    text += ['', f'两月合计费用：旧问题三{old_two:,.2f}元，本次{new_two:,.2f}元，变化{new_two/old_two-1:+.2%}。',
        f"本次二月相对问题二的费用变化：{result['versus_problem2_cost']['difference']:+,.2f}元（{result['versus_problem2_cost']['relative_change']:+.2%}）。", '',
        '费用按照保留原计划、调整增购、紧急购电和实际发生的调减违约四部分汇总。历史误差由新预测器重新校准，风险惩罚不进入账单。', '',
        *result['caveats']]
    (out/'两月测试说明.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'changes':changes, 'two_month_cost':result['two_month_cost'],
                      'versus_problem2_cost':result['versus_problem2_cost'],
                      'prediction_checks':prediction_checks},indent=2))


if __name__ == '__main__':
    main()
