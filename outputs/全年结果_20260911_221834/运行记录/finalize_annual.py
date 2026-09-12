import csv, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
problems=('problem1','problem2','problem3','problem4-2','problem4-3')
state=ROOT/'核验'/'delivery_status.json'
def write_state(status, **extra):
    state.write_text(json.dumps(dict(state=status,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra),ensure_ascii=False,indent=2),encoding='utf-8')
write_state('waiting_for_runs')
while True:
    statuses={p:json.loads((ROOT/'运行记录'/f'{p}_status.json').read_text(encoding='utf-8')) for p in problems}
    failed=[p for p,s in statuses.items() if s['state']=='failed']
    if failed:
        write_state('run_failed',failed=failed)
        sys.exit(1)
    if all(s['state']=='complete' for s in statuses.values()): break
    time.sleep(30)
write_state('verifying')
checks=[]
commands=[[sys.executable,str(ROOT/'核验'/'verify_outputs.py')]]
commands.extend([sys.executable,str(PROJECT/'tools'/'verify_timeaxis.py'),str(ROOT/'运行记录'/p)] for p in problems[1:])
for i,command in enumerate(commands):
    r=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    logfile=ROOT/'核验'/('workbooks.txt' if i==0 else f'{problems[i]}_physics_timeaxis.txt')
    logfile.write_text(r.stdout+r.stderr,encoding='utf-8')
    checks.append(dict(log=logfile.name,returncode=r.returncode))
if any(c['returncode'] for c in checks):
    write_state('verification_failed',checks=checks)
    sys.exit(2)
with (ROOT/'核验'/'result_hashes.json').open('w',encoding='utf-8') as f:
    json.dump({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'result').glob('*.xlsx')},f,indent=2)
# Full-year and template reporting windows have separate quantity and cost totals.
import numpy as np
rows=[]
for p in problems[1:]:
    folder=ROOT/'运行记录'/p
    a=dict(np.load(folder/'trajectory.npz',allow_pickle=False))
    payload=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    summary=payload['summary']
    ledger=json.loads((folder/'settlement_events.json').read_text(encoding='utf-8'))
    assert len(a['abs_minute'])==52560 and a['abs_minute'][0]==10 and a['abs_minute'][-1]==525600
    assert abs(a['soc_boundary_kwh'][0]-6000)<1e-6
    events=list(csv.DictReader((folder/'spill_events.csv').open(encoding='utf-8-sig',newline='')))
    assert abs(sum(float(e['spill_kwh']) for e in events)-float(a['surplus_kwh'].sum()))<1e-3
    assert all(e['capacity_limited']=='True' or e['power_limited']=='True' for e in events)
    full_cost=sum(e['price_actual_yuan_per_kwh']*(e['o_exec_kwh']+1.5*e['a_exec_kwh']+5*e['emergency_kwh']) for e in ledger['execution'])+sum(0.5*e['price_at_yuan_per_kwh']*sum(e['reduced_kwh']) for e in ledger['penalties'])
    assert abs(full_cost-summary['warmup_total_cost_yuan']-summary['report_start_midnight_cost_yuan']-summary['total_cost_yuan'])<1e-3
    rows.append(dict(problem=p,full_year_intervals=len(a['abs_minute']),full_year_cost_yuan=full_cost,full_year_purchase_kwh=float((a['grid_kwh']+a['emergency_kwh']).sum()),full_year_paid_disposal_kwh=float(a['surplus_kwh'].sum()),template_days=summary['output_days'],template_cost_yuan=summary['total_cost_yuan'],template_paid_disposal_kwh=summary['surplus_disposed_kwh'],final_soc_kwh=float(a['soc_boundary_kwh'][-1])))
with (ROOT/'全年汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
write_state('complete',checks=checks,summary=rows)
print('ALL RUNS AND CHECKS COMPLETE',flush=True)
