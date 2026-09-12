"""Build the review note and source receipt from completed validation artifacts."""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET

out = Path('outputs/预测与风险优化_20260912')
forecast = json.loads((out/'forecast_metrics.json').read_text(encoding='utf-8'))
dispatch = json.loads((out/'dispatch_metrics.json').read_text(encoding='utf-8'))
assert len(dispatch['runs']) == 24
aggregate = []
for p in ('problem2', 'problem3', 'problem4-2', 'problem4-3'):
    for variant in ('baseline', 'adaptive', 'reserve90'):
        runs = [r for r in dispatch['runs'] if r['problem'] == p and r['variant'] == variant]
        assert len(runs) == 2
        aggregate.append({'problem':p, 'variant':variant,
            **{key:sum(r[key] for r in runs) for key in ('cost_yuan','emergency_kwh','spill_kwh')},
            'terminal_soc_sum_kwh':sum(r['soc_end'] for r in runs)})
(out/'dispatch_aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
tree = ET.parse(out/'tests.xml')
assert all(int(s.get('failures',0)) == 0 and int(s.get('errors',0)) == 0 for s in tree.iter('testsuite'))
tests = sum(int(s.get('tests',0)) for s in tree.iter('testsuite'))
hashes = json.loads((out/'source_hashes.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())

lines = ['# 预测优化与风险备用验证结果', '',
    '代码已升级；原result模板和此前全年结果未覆盖。以下预测指标是2—12月全年滚动历史检验，调度指标是指定短期成对对照，不是新版全年电费。', '',
    '## 预测误差', '', '|信息与计划分支|指标|旧预测MAE（kWh/段）|新预测MAE（kWh/段）|下降|',
    '|---|---|---:|---:|---:|']
metric_rows = []
for branch in ('historical-frozen','published-frozen','published-adjustable'):
    for field in ('load','pv','net'):
        a = forecast['forecasts'][branch+'/baseline'][field]['mae_kwh']
        b = forecast['forecasts'][branch+'/adaptive'][field]['mae_kwh']
        metric_rows.append({'branch':branch,'metric':field,'old_mae':a,'new_mae':b,'decrease_pct':100*(1-b/a)})
        lines.append(f'|{branch}|{field}|{a:.2f}|{b:.2f}|{100*(1-b/a):.1f}%|')
lines += ['', 'historical-frozen对应问题2；published-frozen对应4-2；published-adjustable对应3与4-3。日初冻结预测检验随后24小时，可调分支在0/6/12/18分别检验随后6小时，均覆盖48096个时段。设计已查看本数据，因此不将因果滚动检验称为未接触的独立留出集。', '',
    '## 备用误差覆盖', '', '|分支|累计误差路径覆盖|逐段净负荷上界覆盖|平均累计备用（送达侧kWh）|', '|---|---:|---:|---:|']
for branch, r in forecast['risk_coverage'].items():
    lines.append(f"|{branch}|{100*r['energy_coverage']:.2f}%|{100*r['interval_coverage']:.2f}%|{r['mean_buffer_kwh']:.2f}|")
lines += ['', '目标分位为90%。这里检验的是估计的误差上界，不是电池实际达到备用要求的比率，更不是未来绝不紧急购电的保证。容量和冻结购电限制导致的备用缺口另有审计记录。', '',
    '## 调度对照', '', '每个分支分别比较2025年2月6—7日、12月8—9日。每一对照窗口使用该分支原轨迹中的同一个起始SOC，连续执行两天。下表合计该分支的两段窗口（共4日）；不同题目是互斥方案，不代表同时发生的支出。', '',
    '|分支|版本|实付总费用（元）|紧急购电（kWh）|弃购电（kWh）|两窗口末端SOC合计（kWh）|',
    '|---|---|---:|---:|---:|---:|']
for r in aggregate:
    lines.append(f"|{r['problem']}|{r['variant']}|{r['cost_yuan']:.2f}|{r['emergency_kwh']:.2f}|{r['spill_kwh']:.2f}|{r['terminal_soc_sum_kwh']:.2f}|")
lines += ['', 'baseline是旧预测；adaptive只改预测；reserve90是在改进预测基础上加入90%经验备用。', '',
    '各版本末端SOC不同，短期费用差不能直接解释为相同库存下的纯节省。备用可能增加弃电和部分日期的支出；以误差覆盖、紧急购电、实付费用、弃电和期末储电量共同判断。风险代理惩罚没有进入实际费用。', '',
    '## 核验与使用', '', f'- {tests}项自动化测试通过；包括未来实测和未发布光伏隔离、跨日星期对应、冻结交易权限、真实充电消纳、满电禁止循环消耗备用购电、不可实现备用缺口及费用分离。',
    '- 原模板SHA256核验全部一致。全部24段调度对照通过物理验证；最终备用版本8段按最终代码重新执行。',
    '- 全年预测检验与风险覆盖已执行；新版全年连续调度尚未执行，旧全年Excel仍表示旧模型。',
    '- 模型规则和CLI命令见Pr/预测优化与风险备用说明.md，假设更新见备忘录第4.1节。',
    '- forecast_metrics.json、forecast_daily.csv为全年预测证据；dispatch_metrics.json及paired_runs为调度证据；tests.xml和source_hashes.json为代码核验记录。', '']
(out/'验证结论.md').write_text('\n'.join(lines), encoding='utf-8')
payload = {'schemaVersion':1, 'items':[
    {'id':'prediction','title':'负荷与光伏预测改进及误差覆盖','queries':[
        {'id':'forecast','source':{'label':'全年滚动预测验证','files':[{'label':'forecast_metrics.json'},{'label':'forecast_daily.csv'},{'label':'evaluate_forecast_risk.py'}],
          'metricDefinitions':[{'label':'预测误差','definition':'平均绝对误差按每个十分钟时段的kWh计算；每个信息分支覆盖48096个时段。'}],
          'caveats':['模型设计已查看本数据；每次预测只使用当时成熟信息，但这不是未接触的独立留出集。'],
          'evidenceFlow':[{'kind':'validation','title':'因果性测试','detail':'未来实测与未发布光伏预报被改写后，当前预测及备用计算保持不变。'}]},
         'reportingPeriod':forecast['period'], 'columns':list(metric_rows[0]),'rows':metric_rows,
         'methods':[{'language':'python','code':'error = actual_kwh - forecast_kwh\nmae = np.mean(np.abs(error))\ndecrease_pct = 100 * (1 - new_mae / old_mae)'}]},
        {'id':'coverage','source':{'label':'联合误差备用校准','files':[{'label':'adaptive_forecast.py'},{'label':'forecast_daily.csv'}],
          'caveats':['经验误差上界覆盖不等于实际储能充足率或未来供电保证。']},
         'columns':['branch','energy_coverage','interval_coverage','mean_buffer_kwh'],
         'rows':[{'branch':b,**{k:r[k] for k in ('energy_coverage','interval_coverage','mean_buffer_kwh')}} for b,r in forecast['risk_coverage'].items()]}]},
    {'id':'dispatch','title':'改进预测与备用的短期调度对照','queries':[
        {'id':'paired','source':{'label':'四分支成对调度验证','files':[{'label':'dispatch_metrics.json'},{'label':'paired_runs/trajectory.npz'},{'label':'tests.xml'}],
          'metricDefinitions':[{'label':'实付总费用','definition':'总费用只包含原计划、增购、紧急购电和调减违约费用，排除规划风险惩罚。'}],
          'caveats':['每个分支共两个两日窗口；相同起点SOC、不同末端SOC，不能外推为全年或相同库存下的节省。'],
          'evidenceFlow':[{'kind':'validation','title':'核验结果','detail':f'{tests}项测试通过，24段对照通过物理校验；原result模板哈希保持一致。'}]},
         'reportingPeriod':'2025年2月6—7日、12月8—9日；每个方案共4日',
         'columns':list(aggregate[0]),'rows':aggregate,
         'methods':[{'language':'calculation','code':'实际费用 = Σ 实际电价 × (原计划执行量 + 1.5×增购执行量 + 5×紧急购电量) + 调减违约费用'}]}]}
]}
(out/'sources.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
print(json.dumps(aggregate, indent=2))
