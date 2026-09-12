import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url)),out=path.dirname(here);
const p=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
const wb=Workbook.create(),summary=wb.worksheets.add('检验汇总'),jan=wb.worksheets.add('1月逐时预测'),
 feb=wb.worksheets.add('2月逐时预测'),horizon=wb.worksheets.add('预报提前量'),blocks=wb.worksheets.add('六小时偏差');
const blue='#17365D',numeric='#,##0.000',date=s=>Date.parse(s.length===10?`${s}T00:00:00Z`:`${s}Z`)/86400000+25569;
function setup(s,col,last,title,subtitle,labels){
 const r=s.getRange(`A1:${col}${last}`);r.format.font={name:'Arial',size:10,color:'#222222'};
 r.format.rowHeight=20;r.format.columnWidth=20;r.format.verticalAlignment='center';s.showGridLines=false;
 s.getRange('A1').values=[[title]];s.getRange('A1').format.font={name:'Arial',size:15,bold:true,color:blue};s.getRange('A1').format.rowHeight=28;
 s.getRange('A2').values=[[subtitle]];s.getRange('A2').format.font={name:'Arial',size:10,color:'#666666'};
 s.getRange(`A4:${col}4`).values=[labels];s.getRange(`A4:${col}4`).format={fill:blue,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:48};
 s.getRange(`A4:${col}4`).format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
}
function fill(s,c,f,end){s.getRange(`${c}5`).formulas=[[f]];if(end>5)s.getRange(`${c}5:${c}${end}`).fillDown();}
const months=[{s:jan,key:'2025-01',rows:p.rows.slice(0,4464),label:'一月前置及候选比较'},
 {s:feb,key:'2025-02',rows:p.rows.slice(4464),label:'二月预测检验'}];
const errorFormulas={J:'=G5-D5',K:'=H5-D5',L:'=I5-D5',P:'=N5-M5',Q:'=O5-M5',R:'=M5-D5',
 S:'=N5-G5',T:'=N5-H5',U:'=O5-I5',V:'=R5-S5',W:'=R5-T5',X:'=R5-U5',Y:'=F5-D5',Z:'=E5-D5'};
for(const {s,rows,label} of months){
 const end=rows.length+4;
 setup(s,'Z',end,label,'每段10分钟，数值单位kWh；节点预测形成后固定，滚动预测仅使用段初已完成实测。',
  [...p.headers,'历史光伏高估误差','原始报告光伏高估误差'].map((x,i)=>i<3?x:x+'\n(kWh)'));
 s.getRange(`A5:Z${end}`).values=rows.map(r=>[...r.map((v,i)=>i<3?date(v):v),r[5]-r[3],r[4]-r[3]]);
 s.getRange(`A5:C${end}`).setNumberFormat('yyyy-mm-dd hh:mm');s.getRange(`D5:Z${end}`).setNumberFormat(numeric);
 s.getRange('A:C').format.columnWidth=23;s.getRange('J:L').format.columnWidth=25;s.getRange('S:Z').format.columnWidth=26;
 for(const [col,f] of Object.entries(errorFormulas))fill(s,col,f,end);
 s.tables.add(`A4:Z${end}`,true,s===jan?'JanuaryForecast':'FebruaryForecast').showFilterButton=true;
 s.freezePanes.freezeRows(4);s.freezePanes.freezeColumns(3);
}
setup(summary,'E',40,'光伏预报优化：一月至二月预测测试','一月比较候选，二月按选定方法向前检验；本次未运行购电调度。',
 ['指标','一月','二月','单位','解释']);
summary.getRange('A:A').format.columnWidth=32;summary.getRange('B:C').format.columnWidth=22;
summary.getRange('D:D').format.columnWidth=13;summary.getRange('E:E').format.columnWidth=80;
const spec=[
 ['原始报告光伏MAE','pv','raw','mae','Z','未做校正的小时平均报告'],
 ['历史光伏MAE','pv','history','mae','Y','最近7日同刻历史预测'],
 ['旧融合光伏MAE','pv','old','mae','J','此前问题三采用的全天统一NNLS融合'],
 ['优化节点光伏MAE','pv','new_node','mae','K','最近00/06/12/18节点形成的未来6小时预测'],
 ['优化滚动光伏MAE','pv','new_rolling','mae','L','使用最近已完成区间误差校正当前预测'],
 ['节点负荷MAE','load','node','mae','P','沿用周周期负荷模型，本次未更改'],
 ['滚动负荷MAE','load','rolling','mae','Q','沿用周周期加近期偏差修正'],
 ['旧节点净负荷MAE','net','old_node','mae','V','实际负荷减光伏，与预测净负荷比较'],
 ['优化节点净负荷MAE','net','new_node','mae','W','与旧节点预测使用相同负荷输入'],
 ['优化滚动净负荷MAE','net','new_rolling','mae','X','不代表节点普通购电可随时调整'],
 ['旧光伏正向高估合计','pv','old','overprediction_kwh','J','仅累计预测光伏大于实测的部分'],
 ['优化节点光伏正向高估','pv','new_node','overprediction_kwh','K','不与低估误差相抵销'],
 ['优化滚动光伏正向高估','pv','new_rolling','overprediction_kwh','L','更小MAE不保证每种偏差均更小'],
 ['旧节点净负荷低估合计','net','old_node','underprediction_kwh','V','实际净负荷超过预测的部分'],
 ['优化节点净负荷低估合计','net','new_node','underprediction_kwh','W','用于观察潜在购电不足风险']
];
for(let i=0;i<spec.length;i++){
 const [label,kind,method,metric,col,note]=spec[i],r=i+5;
 summary.getRange(`A${r}:E${r}`).values=[[label,null,null,'kWh',note]];
 for(const [j,{s,key,rows}] of months.entries()){
  const dest=j===0?'B':'C',ref=`'${s.name}'!${col}5:${col}${rows.length+4}`;
  const f=metric==='mae'?`=(SUMIF(${ref},">0")-SUMIF(${ref},"<0"))/COUNT(${ref})`:`=SUMIF(${ref},">0")`;
  summary.getRange(`${dest}${r}`).formulas=[[f]];
 }
}
summary.getRange('A20:E23').values=[['检验时段数',4464,4032,'段','按原result行口径划分'],
 ['优化节点光伏MAE降幅',null,null,'%','相对旧融合；使用相同发布节点比较'],
 ['优化滚动光伏MAE降幅',null,null,'%','相对旧滚动预测，包含实测反馈的价值'],
 ['优化节点相对历史MAE降幅',null,null,'%','检验利用报告是否优于仅用历史']];
for(const c of ['B','C']){
 summary.getRange(`${c}21`).formulas=[[`=1-${c}8/${c}7`]];
 summary.getRange(`${c}22`).formulas=[[`=1-${c}9/${c}7`]];
 summary.getRange(`${c}23`).formulas=[[`=1-${c}8/${c}6`]];
}
summary.getRange('B5:C19').setNumberFormat(numeric);summary.getRange('B21:C23').setNumberFormat('0.0%');
summary.getRange('D21:D23').values=[['比例'],['比例'],['比例']];
summary.getRange('A8:E9').format.fill='#E9EFF7';summary.getRange('A8:C9').format.font={name:'Arial',size:10,bold:true,color:blue};
summary.getRange('A25:E25').values=[['选择与检验口径','','','','']];
summary.getRange('A25:E25').format={fill:blue,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
summary.getRange('A26:B35').values=[
 ['选定方法','分小时校正后动态融合'],['选择数据','一月15日00:10至二月1日00:10'],
 ['选择依据','上述区间光伏MAE最低'],['小时形状','历史十分钟占比，保持原始小时总量'],
 ['小时校正','按发布时间与目标小时分别校正'],['融合依据','过去已成熟的历史预测误差'],
 ['滚动反馈','近1小时已完成误差，按2小时尺度衰减'],['一月用途','冷启动、前置、候选方法比较'],
 ['二月用途','算法选定后，仅用当时成熟历史更新'],['后续应用','候选预测器独立保存，尚未接入主模型']];
summary.getRange('A26:D35').format.rowHeight=32;
summary.getRange('B26:D35').format.wrapText=false;
summary.getRange('E26:E31').values=[
 ['来源：附件1冷启动先验、附件2实测、附件3当时已发布预报。'],
 ['一月从1月1日00:10开始，首个缺失时段跳过。二月报告至3月1日00:10，不含终点。'],
 ['本次设计参考过二月问题诊断，因此二月属于开发回测。独立验证需要另选未参与设计的月份。'],
 ['提前量表统计多个发布版本，目标时间可能重复；不能与逐时表的8496个唯一时段相加。'],
 ['分小时校正可改变预报总量；只有校正前的小时内形状分配保持原始小时总量。'],
 ['只比较预测误差，未计算新策略的购电量、电费或紧急购电量。']];
summary.getRange('E26:E31').format.wrapText=true;summary.getRange('E26:E31').format.rowHeight=46;

const hr=p.summary.horizons.slice().sort((a,b)=>a.month.localeCompare(b.month)||parseInt(a.horizon)-parseInt(b.horizon)||(a.method==='旧融合'?-1:1))
 .map(r=>[r.month,r.horizon,r.method,r.n,r.pv.mae,r.pv.rmse,r.net.mae,r.net.rmse,r.pv.bias_pred_minus_actual,r.net.bias_pred_minus_actual]);
setup(horizon,'J',hr.length+4,'发布报告：不同提前量的预测误差','按目标时段所属月份统计；同一目标可能被不同发布时间预测，样本量为预测版本数。',
 ['月份','提前量','方法','预测样本数','光伏MAE\n(kWh)','光伏RMSE\n(kWh)','净负荷MAE\n(kWh)','净负荷RMSE\n(kWh)','光伏偏差\n预测减实际','净负荷偏差\n预测减实际']);
horizon.getRange(`A5:J${hr.length+4}`).values=hr;horizon.getRange(`E5:J${hr.length+4}`).setNumberFormat(numeric);
horizon.tables.add(`A4:J${hr.length+4}`,true,'ForecastHorizons').showFilterButton=true;horizon.freezePanes.freezeRows(4);
setup(blocks,'G',p.blocks.length+4,'相邻调整节点间的净负荷偏差','累计缺口峰值为从节点开始累计误差的正向最大值；它是预测误差指标，不是实际紧急购电量。',
 ['预测形成时间','核验结束时间','时段数','方法','净负荷MAE\n(kWh)','累计缺口峰值\n(kWh)','累计有符号误差\n实际减预测']);
blocks.getRange(`A5:G${p.blocks.length+4}`).values=p.blocks.map(r=>r.map((v,i)=>i<2?date(v):v));
blocks.getRange(`A5:B${p.blocks.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm');blocks.getRange('A:B').format.columnWidth=24;
blocks.getRange('E:G').format.columnWidth=25;blocks.getRange(`E5:G${p.blocks.length+4}`).setNumberFormat(numeric);
blocks.tables.add(`A4:G${p.blocks.length+4}`,true,'BlockErrors').showFilterButton=true;blocks.freezePanes.freezeRows(4);

const close=(a,b,label,tol=1e-5)=>{if(typeof a!=='number'||!Number.isFinite(a)||Math.abs(a-b)>tol)throw new Error(`${label}: ${a} != ${b}`);};
for(let i=0;i<spec.length;i++)for(const [j,{key}]of months.entries()){
 const [,kind,method,metric]=spec[i];close(summary.getRange(`${j?'C':'B'}${i+5}`).values[0][0],p.summary.months[key][kind][method][metric],`${key}/${kind}/${method}/${metric}`,1e-4);
}
const inspections=[];
for(const [s,r]of [['检验汇总','A4:E23'],['2月逐时预测','D5:L10'],['预报提前量','A4:J12']])
 inspections.push((await wb.inspect({kind:'table',range:`'${s}'!${r}`,include:'values,formulas',tableMaxRows:21,tableMaxCols:10,maxChars:4500})).ndjson);
inspections.push((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
await fs.writeFile(path.join(here,'inspection.jsonl'),inspections.join('\n'));
for(const [s,r,name]of [['检验汇总','A1:E23','summary'],['检验汇总','A25:E35','notes'],['1月逐时预测','A1:I10','january'],
 ['2月逐时预测','A1:I10','february'],['2月逐时预测','M4:X10','load_net'],['预报提前量','A1:J16','horizons'],['六小时偏差','A1:G12','blocks']]){
 if(process.env.PREVIEW_FILTER&&!process.env.PREVIEW_FILTER.split(',').includes(name))continue;
 const blob=await wb.render({sheetName:s,range:r,scale:1,format:'png'});await fs.writeFile(path.join(here,`${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=path.join(out,'光伏与负荷_一月至二月预测检验.xlsx');await(await SpreadsheetFile.exportXlsx(wb)).save(file);
console.log(JSON.stringify({file,sheets:5,intervals:p.rows.length}));
