"""January selection, February causal evaluation. Never call a dispatch solver."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,csv,hashlib,json,time
import numpy as np
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline
from microgrid.adaptive_forecast import AdaptiveForecaster
from microgrid.report_forecast import ReportAwareForecaster

ORIGIN=datetime(2025,1,1)
def stamp(m):return (ORIGIN+timedelta(minutes=int(m))).isoformat()
def metrics(error):
    return {'mae':float(np.mean(abs(error))),'rmse':float(np.sqrt(np.mean(error**2))),
            'bias_pred_minus_actual':float(np.mean(error)),
            'overprediction_kwh':float(np.maximum(error,0).sum()),
            'underprediction_kwh':float(np.maximum(-error,0).sum()),
            'p95_absolute':float(np.quantile(abs(error),.95))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    b=load_all();tl=build_timeline(b)
    old=AdaptiveForecaster(b,tl);candidate=ReportAwareForecaster(b,tl,nowcast=False)
    minutes=np.arange(10,84970,10);n=len(minutes);idx=minutes//10
    actual_pv=tl.pv_kw.values[idx]/6;actual_load=tl.load_kw.values[idx]/6
    assert np.isfinite(actual_pv).all() and np.isfinite(actual_load).all()
    arrays={k:np.zeros(n) for k in ('raw','old','history','shape','calibrated','blend','load_node')}
    nodes=(minutes//360)*360;nodes[minutes<360]=10
    # Stage one: the declared candidate family is assessed on January only.
    def fill_nodes(mask):
        for node in np.unique(nodes[mask]):
            ix=np.flatnonzero((nodes==node)&mask)
            end=int(minutes[ix[-1]])+10
            anchor=int(node)//360*360; offsets=(minutes[ix]-anchor)//10
            components=candidate._report_components(anchor)
            raw=candidate._published_day(anchor)[offsets]
            arrays['raw'][ix]=np.where(np.isfinite(raw),raw,components['history'][offsets])
            load,pv=old._point(int(node),end,True)
            arr_offset=(minutes[ix]-node)//10
            arrays['old'][ix]=pv[arr_offset];arrays['load_node'][ix]=load[arr_offset]
            for name in ('history','shape','calibrated'):
                arrays[name][ix]=components[name][offsets]
            w=np.repeat(candidate._report_weights(anchor),6)
            arrays['blend'][ix]=(components['history']+w*(components['calibrated']-components['history']))[offsets]
    january=minutes<44650
    fill_nodes(january)
    select=(minutes>=14*1440+10)&january
    scores={name:metrics(arrays[name][select]-actual_pv[select]) for name in ('history','shape','calibrated','blend')}
    winner=min(scores,key=lambda k:scores[k]['mae'])
    selection={'selection_from':stamp(14*1440+10),'selection_to':stamp(44650),
        'criterion':'minimum PV MAE; candidate family and hyperparameters fixed before February evaluation',
        'selected_variant':winner,'january_scores':scores,
        'caveat':'Earlier February diagnosis informed model design; February is a development backtest, not untouched external validation.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('January selection: '+json.dumps(selection),flush=True)
    fill_nodes(~january)
    improved=ReportAwareForecaster(b,tl,variant=winner,nowcast=True)
    arrays['new_node']=arrays[winner].copy()
    for name in ('new_rolling','old_rolling','load_rolling'):
        arrays[name]=np.zeros(n)
    for i,m in enumerate(minutes):
        load,pv=improved._point(int(m),int(m)+10,True)
        oldload,oldpv=old._point(int(m),int(m)+10,True)
        assert abs(load[0]-oldload[0])<1e-8
        arrays['new_rolling'][i]=pv[0];arrays['old_rolling'][i]=oldpv[0];arrays['load_rolling'][i]=load[0]
        if (i+1)%1440==0:print(f'Forecast-only progress: {i+1}/{n}',flush=True)
    monthly={}
    for month,mask in [('2025-01',january),('2025-02',~january)]:
        report={'n':int(mask.sum()),'pv':{},'load':{},'net':{}}
        for name in ('raw','history','old','shape','calibrated','blend','new_node','old_rolling','new_rolling'):
            report['pv'][name]=metrics(arrays[name][mask]-actual_pv[mask])
        for name,col in [('node','load_node'),('rolling','load_rolling')]:
            report['load'][name]=metrics(arrays[col][mask]-actual_load[mask])
        for name,pvcol,loadcol in [('old_node','old','load_node'),('new_node','new_node','load_node'),
                                 ('old_rolling','old_rolling','load_rolling'),('new_rolling','new_rolling','load_rolling')]:
            report['net'][name]=metrics((arrays[loadcol]-arrays[pvcol]-actual_load+actual_pv)[mask])
        morning=mask&(minutes%1440>=540)&(minutes%1440<660)
        report['morning_pv']={name:metrics(arrays[name][morning]-actual_pv[morning]) for name in ('old','new_node','new_rolling')}
        monthly[month]=report
    rows=[]
    for i,m in enumerate(minutes):
        pv,load=float(actual_pv[i]),float(actual_load[i]);v=lambda k:float(arrays[k][i])
        oldpv,newpv,rolling=v('old'),v('new_node'),v('new_rolling');ln,lr=v('load_node'),v('load_rolling')
        rows.append([stamp(m),stamp(m+10),stamp(nodes[i]),pv,v('raw'),v('history'),oldpv,newpv,rolling,
            oldpv-pv,newpv-pv,rolling-pv,load,ln,lr,ln-load,lr-load,load-pv,ln-oldpv,ln-newpv,lr-rolling,
            load-pv-ln+oldpv,load-pv-ln+newpv,load-pv-lr+rolling])
    headers=['开始时间','结束时间','最近决策节点','实际光伏','原始报告光伏','历史光伏','旧融合光伏',
             '优化节点光伏','优化滚动光伏','旧光伏高估误差','优化节点光伏高估误差','优化滚动光伏高估误差',
             '实际负荷','节点负荷预测','滚动负荷预测','节点负荷误差','滚动负荷误差','实际净负荷',
             '旧节点净负荷预测','优化节点净负荷预测','优化滚动净负荷预测','旧节点净负荷低估',
             '优化节点净负荷低估','优化滚动净负荷低估']
    for month,mask in [('2025年1月',january),('2025年2月',~january)]:
        with (out/f'{month}_逐时预测.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(headers);writer.writerows([rows[i] for i in np.flatnonzero(mask)])
    blocks=[]
    for node in np.unique(nodes):
        mask=nodes==node
        for name,col in [('旧融合','old'),('优化节点','new_node')]:
            er=(actual_load-actual_pv)-(arrays['load_node']-arrays[col])
            blocks.append([stamp(node),stamp(minutes[mask][-1]+10),int(mask.sum()),name,
                float(np.mean(abs(er[mask]))),float(np.maximum(np.cumsum(er[mask]),0).max()),float(er[mask].sum())])
    # Separate report-vintage evaluation: the same target may appear in several
    # issued forecasts. Never add this sample count to the unique interval count.
    horizon_errors={}
    for node in np.unique(nodes):
        end=min(int(node)+1440,84970)
        target=np.arange(int(node),end,10)
        load,oldpv=old._point(int(node),end,True)
        _,newpv=improved._point(int(node),end,True)
        actual=tl.pv_kw.values[target//10]/6
        actuald=tl.load_kw.values[target//10]/6
        for month,mm in [('2025-01',target<44650),('2025-02',target>=44650)]:
            for label,a,bound in [('0—6小时',0,360),('6—12小时',360,720),('12—24小时',720,1440)]:
                keep=mm&(target-node>=a)&(target-node<bound)
                if keep.any():
                    for name,pred in [('旧融合',oldpv),('优化',newpv)]:
                        key=(month,label,name)
                        horizon_errors.setdefault(key,[]).append(np.column_stack((pred[keep]-actual[keep],
                            (load-pred-actuald+actual)[keep])))
    horizons=[]
    for (month,horizon,name),parts in horizon_errors.items():
        err=np.concatenate(parts)
        horizons.append({'month':month,'horizon':horizon,'method':name,'n':len(err),
                         'pv':metrics(err[:,0]),'net':metrics(err[:,1])})
    summary={'selection':selection,'period':{'from':stamp(10),'to':stamp(84970),'n':n},'months':monthly,
             'horizons':horizons,'wall_seconds':time.perf_counter()-started,'dispatch_solves':0}
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    payload={'summary':summary,'headers':headers,'rows':rows,'blocks':blocks}
    (out/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(out/'forecast_arrays.npz',minutes=minutes,nodes=nodes,actual_pv=actual_pv,actual_load=actual_load,**arrays)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path('src/microgrid/report_forecast.py'),Path('tools/evaluate_report_forecast.py'),Path('data/附件2.xlsx'),Path('data/附件3.xlsx')]}
    (out/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
