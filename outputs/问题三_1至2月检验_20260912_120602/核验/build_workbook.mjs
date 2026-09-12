import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url)), output=path.dirname(here);
const p=JSON.parse(await fs.readFile(path.join(output,'month_payload.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('月度汇总');
const detail=wb.worksheets.add('逐时购电与费用');
const daily=wb.worksheets.add('每日汇总');
const penalty=wb.worksheets.add('调减违约明细');
const check=wb.worksheets.add('供电与预测核验');
const end=p.n+4, days=p.n/144, pend=p.penalty_rows.length+4;
const date=s=>Date.parse(s.length===10?`${s}T00:00:00Z`:`${s}Z`)/86400000+25569;
const number='#,##0.000000', money='#,##0.00', blue='#17365D';
function setup(s,col,last,title,subtitle,labels){
  const r=s.getRange(`A1:${col}${last}`);
  r.format.font={name:'Arial',size:10,color:'#222222'};
  r.format.rowHeight=20;r.format.columnWidth=18;r.format.verticalAlignment='center';
  s.showGridLines=false;
  s.getRange('A1').values=[[title]];
  s.getRange('A1').format.font={name:'Arial',size:15,bold:true,color:blue};
  s.getRange('A1').format.rowHeight=28;
  s.getRange('A2').values=[[subtitle]];
  s.getRange('A2').format.font={name:'Arial',size:10,color:'#666666'};
  const h=s.getRange(`A4:${col}4`);h.values=[labels];
  h.format={fill:blue,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,
    horizontalAlignment:'center',verticalAlignment:'center',rowHeight:48};
  h.format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
}
function fill(s,col,f,last=end){
  s.getRange(`${col}5`).formulas=[[f]];
  if(last>5)s.getRange(`${col}5:${col}${last}`).fillDown();
}
function table(s,col,last,name,cols=2){
  s.tables.add(`A4:${col}${last}`,true,name).showFilterButton=true;
  s.freezePanes.freezeRows(4);if(cols)s.freezePanes.freezeColumns(cols);
}
const labels=headers=>headers.map(h=>h.replace('_元每kWh','\n(元/kWh)').replace('_kWh','\n(kWh)').replace('_元','\n(元)'));
setup(detail,'W',end,'问题三：逐时购电与费用','2月检验；调整后普通购电＝保留原计划＋调整增购，实际外网总量再加紧急购电。',labels(p.purchase_headers));
detail.getRange(`A5:W${end}`).values=p.rows.map(r=>r.map((v,i)=>i<3?date(v):v));
detail.getRange(`D5:W${end}`).setNumberFormat(number);
detail.getRange(`A5:B${end}`).setNumberFormat('yyyy-mm-dd hh:mm');
detail.getRange(`C5:C${end}`).setNumberFormat('yyyy-mm-dd');
detail.getRange(`D5:D${end}`).setNumberFormat('0');
for(const c of ['L','N','P'])detail.getRange(`${c}5:${c}${end}`).setNumberFormat('0.0000');
for(const c of ['M','O','Q','S','T'])detail.getRange(`${c}5:${c}${end}`).setNumberFormat(money);
detail.getRange(`U5:U${end}`).setNumberFormat('@');
detail.getRange('A:B').format.columnWidth=23;detail.getRange('C:C').format.columnWidth=16;
detail.getRange('D:D').format.columnWidth=11;detail.getRange('H:K').format.columnWidth=22;
detail.getRange('R:T').format.columnWidth=22;
for(const [c,f] of Object.entries({H:'=F5+G5',I:'=H5-E5',K:'=H5+J5',M:'=F5*L5',
  N:"=L5*'月度汇总'!$F$8",O:'=G5*N5',P:"=L5*'月度汇总'!$F$10",Q:'=J5*P5',
  T:'=M5+O5+Q5+S5',U:'=TEXT(T5,"0.00")&"元"',W:'=K5-V5'}))fill(detail,c,f);
for(let i=0;i<p.n;i++){
  const range=p.penalty_ranges[p.rows[i][0]];
  if(range){const [a,b]=range,r=i+5;
    detail.getRange(`R${r}:S${r}`).formulas=[[`=SUM('调减违约明细'!C${a}:C${b})`,`=SUM('调减违约明细'!F${a}:F${b})`]];
  }
}
detail.getRange(`J5:J${end}`).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:.000001,format:{fill:'#FFF2CC',font:{color:'#9C5700'}}});
detail.getRange(`I5:I${end}`).conditionalFormats.addCustom('ABS(I5)>0.000001',{fill:'#E2EFDA'});
detail.getRange(`T5:T${end}`).format.font={name:'Arial',size:10,bold:true,color:blue};
table(detail,'W',end,'Problem3Purchase');

setup(penalty,'F',Math.max(pend,5),'问题三：调减违约明细','费用计入调减发生时刻；交付段仅用于追溯，不改变费用归属。',labels(p.penalty_headers));
if(p.penalty_rows.length){
  penalty.getRange(`A5:F${pend}`).values=p.penalty_rows.map(r=>r.map((v,i)=>i<2?date(v):v));
  penalty.getRange(`A5:B${pend}`).setNumberFormat('yyyy-mm-dd hh:mm');
  penalty.getRange(`C5:C${pend}`).setNumberFormat(number);
  penalty.getRange(`D5:E${pend}`).setNumberFormat('0.0000');
  penalty.getRange(`F5:F${pend}`).setNumberFormat(money);
  fill(penalty,'E',"=D5*'月度汇总'!$F$9",pend);fill(penalty,'F','=C5*E5',pend);
  table(penalty,'F',pend,'Problem3Penalties');
}else penalty.getRange('A5').values=[['本期无调减违约事件']];
penalty.getRange('A:B').format.columnWidth=26;penalty.getRange('C:F').format.columnWidth=22;

setup(check,'V',end,'问题三：供电与预测核验','日初预测与滚动预测分别列示；误差为实际净负荷减预测净负荷。',labels(p.check_headers));
check.getRange(`A5:V${end}`).values=p.check_rows.map(r=>r.map((v,i)=>i<2?date(v):v));
check.getRange(`C5:V${end}`).setNumberFormat(number);check.getRange(`A5:B${end}`).setNumberFormat('yyyy-mm-dd hh:mm');
check.getRange('A:B').format.columnWidth=23;check.getRange('H:K').format.columnWidth=22;
check.getRange('R:T').format.columnWidth=26;check.getRange('U:V').format.columnWidth=22;
for(const [c,f] of Object.entries({A:"='逐时购电与费用'!A5",G:'=E5-F5',H:'=G5-C5+D5',K:'=G5-I5+J5',
  U:"='逐时购电与费用'!H5+'逐时购电与费用'!J5+F5-P5+O5-E5-N5/'月度汇总'!$F$6-Q5",
  V:"=M5-L5-N5+O5/'月度汇总'!$F$7"}))fill(check,c,f);
for(const c of ['U','V'])check.getRange(`${c}5:${c}${end}`).conditionalFormats.addCustom(`ABS(${c}5)>0.00001`,{fill:'#FCE4D6'});
table(check,'V',end,'Problem3Physics');

setup(daily,'O',days+4,'问题三：每日购电与费用','每行覆盖当日00:10至次日00:10；违约费按发生时间汇总。',
  ['归属日期','日初计划\n(kWh)','保留原计划\n(kWh)','调整增购\n(kWh)','调整后普通购电\n(kWh)','紧急购电\n(kWh)',
   '实际外网总购电\n(kWh)','原计划执行费\n(元)','调整增购费\n(元)','紧急购电费\n(元)','违约费\n(元)',
   '总费用\n(元)','弃购电\n(kWh)','期初SOC\n(kWh)','期末SOC\n(kWh)']);
const df=[];
for(let d=0;d<days;d++){
  const a=5+d*144,b=a+143;
  df.push([`='逐时购电与费用'!C${a}`,...['E','F','G','H','J','K','M','O','Q','S','T','V'].map(c=>`=SUM('逐时购电与费用'!${c}${a}:${c}${b})`),
    `='供电与预测核验'!L${a}`,`='供电与预测核验'!M${b}`]);
}
daily.getRange(`A5:O${days+4}`).formulas=df;daily.getRange(`A5:A${days+4}`).setNumberFormat('yyyy-mm-dd');
daily.getRange(`B5:O${days+4}`).setNumberFormat('#,##0.000');daily.getRange(`H5:L${days+4}`).setNumberFormat(money);
daily.getRange('A:A').format.columnWidth=17;daily.getRange('E:G').format.columnWidth=22;table(daily,'O',days+4,'Problem3Daily',0);

setup(summary,'G',38,'问题三：2025年2月检验','从年初00:10以6000 kWh连续运行；一月前置，二月检验。',
  ['指标','二月结果','单位','','参数或口径','取值','说明']);
summary.getRange('A:A').format.columnWidth=31;summary.getRange('B:B').format.columnWidth=24;
summary.getRange('C:C').format.columnWidth=12;summary.getRange('D:D').format.columnWidth=4;
summary.getRange('E:E').format.columnWidth=25;summary.getRange('F:F').format.columnWidth=24;summary.getRange('G:G').format.columnWidth=58;
summary.getRange('E5:G17').values=[
  ['时段长度',10,'分钟'],['充电效率',.9,'实际存入量与母线输入量之比'],['放电效率',.9,'实际送达量与电池消耗量之比'],
  ['调整增购费率',1.5,'执行时电价乘此倍率'],['调减违约费率',.5,'发生时电价乘此倍率'],['紧急购电费率',5,'执行时电价乘此倍率'],
  ['风险经验分位',.9,'依据历史预测偏差设置备用'],['SOC下限',1200,'kWh'],['SOC上限',10800,'kWh'],
  ['年初初始电量',6000,'kWh；二月不重置'],['报告开始',date(p.from),'含此起点'],['报告结束',date(p.to),'不含此终点'],
  ['检验时段数',p.n,`${days}日，每日144段`]];
summary.getRange('F6:F7').setNumberFormat('0.0%');summary.getRange('F11').setNumberFormat('0.0%');
summary.getRange('F15:F16').setNumberFormat('yyyy-mm-dd hh:mm');
summary.getRange('E19:G29').values=[
  ['调整节点','06:00、12:00、18:00','仅调整当天尚未执行的普通购电计划'],
  ['日初计划','00:00制定','年初缺失首段跳过，首日00:10制定'],
  ['数量口径','普通量＝原计划执行＋增购','外网总量＝普通量＋紧急量'],
  ['增购标签','按O/A递推保留','撤销后恢复归增购，不能只看最终净差额'],
  ['违约费归属','实际调减发生时间','可追溯目标交付段，不转移入账时刻'],
  ['购电成本','执行费用加已发生违约费','规划风险惩罚不作为真实电费'],
  ['弃购电','购入后未被利用的电量','仍然付费，已含在相应购电费用中'],
  ['费用精度','原始精度求和','标签显示2位小数，合计不先逐段舍入'],
  ['预测来源','附件1、2、3','重复电价、历史实测与当时已发布光伏预报'],
  ['备用缺口','预测要求与能力的差额','属于规划窗口指标，不代表实际缺电'],
  ['累计调减量','逐次调减电量求和','后续恢复不抵销，区别于相对日初的净调整量']];
summary.getRange('E19:G29').format.wrapText=true;summary.getRange('E19:G29').format.rowHeight=40;
const metrics=[
  ['日初计划购电量','plan_kwh','kWh','E'],['保留原计划执行量','retained_kwh','kWh','F'],
  ['调整增购执行量','add_kwh','kWh','G'],['调整后普通购电量','ordinary_kwh','kWh','H'],
  ['相对日初净调整量','net_adjustment_kwh','kWh','I'],['紧急购电量','emergency_kwh','kWh','J'],
  ['实际外网总购电量','total_kwh','kWh','K'],['原计划执行费','plan_cost','元','M'],
  ['调整增购费','add_cost','元','O'],['紧急购电费','emergency_cost','元','Q'],
  ['累计调减电量','reduced_kwh','kWh','R'],['调减违约费','penalty_cost','元','S'],
  ['月度总费用','total_cost','元','T'],['弃购电量','spill_kwh','kWh','V'],
  ['月初SOC','soc_start','kWh',null,"='供电与预测核验'!L5"],
  ['月末SOC','soc_end','kWh',null,`='供电与预测核验'!M${end}`],
  ['紧急购电时段数','emergency_intervals','段',null,`=COUNTIF('逐时购电与费用'!J5:J${end},">0.000001")`],
  ['净购电量变化时段数','revised_intervals','段',null,`=COUNTIF('逐时购电与费用'!I5:I${end},">0.000001")+COUNTIF('逐时购电与费用'!I5:I${end},"<-0.000001")`],
  ['产生违约的调整节点数','penalty_nodes','次',null,`=COUNTIF('逐时购电与费用'!S5:S${end},">0")`],
  ['弃购电时段数','spill_intervals','段',null,`=COUNTIF('逐时购电与费用'!V5:V${end},">0.000001")`]
];
for(let i=0;i<metrics.length;i++){
  const [label,key,unit,col,formula]=metrics[i],r=i+5;
  summary.getRange(`A${r}:C${r}`).values=[[label,null,unit]];
  summary.getRange(`B${r}`).formulas=[[formula??`=SUM('逐时购电与费用'!${col}5:${col}${end})`]];
  summary.getRange(`B${r}`).setNumberFormat(unit==='元'?money:unit==='kWh'?'#,##0.000':'0');
}
summary.getRange('A17:C17').format.fill='#E9EFF7';summary.getRange('A17:C17').format.font={name:'Arial',size:10,bold:true,color:blue};
summary.getRange('A30:C30').values=[['一月前置及衔接','数值','单位或口径']];
summary.getRange('A30:C30').format={fill:blue,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
summary.getRange('A31:C36').values=[['执行开始',date(p.warmup.from),'含起点'],['衔接至二月报告',date(p.warmup.to),'不含终点'],
  ['前置时段数',p.warmup.n,'段'],['起始SOC',6000,'kWh'],['衔接SOC',p.warmup.soc_end,'kWh'],['前置费用',p.warmup.total_cost,'元']];
summary.getRange('B31:B32').setNumberFormat('yyyy-mm-dd hh:mm');summary.getRange('B34:B35').setNumberFormat('#,##0.000');
summary.getRange('B36').setNumberFormat(money);
summary.getRange('A38').values=[['来源：主模型问题三运行记录；一月前置按result行口径含2月1日00:00—00:10衔接段。']];
summary.getRange('A38').format.font={name:'Arial',size:10,color:'#666666'};

const close=(a,b,label,tol=1e-5)=>{if(typeof a!=='number'||!Number.isFinite(a)||Math.abs(a-b)>tol)throw new Error(`${label}: ${a} != ${b}`);};
for(let i=0;i<metrics.length;i++)close(summary.getRange(`B${i+5}`).values[0][0],p.current[metrics[i][1]],metrics[i][1],1e-4);
for(const [col,index] of [['H',7],['I',8],['K',10],['M',12],['N',13],['O',14],['P',15],['Q',16],['R',17],['S',18],['T',19],['W',22]]){
  const vals=detail.getRange(`${col}5:${col}${end}`).values;
  for(let i=0;i<p.n;i++)close(vals[i][0],p.rows[i][index],`${col}${i+5}`);
}
for(const c of ['U','V'])for(const row of check.getRange(`${c}5:${c}${end}`).values)close(row[0],0,c);
const inspections=[];
for(const [s,r] of [['月度汇总','A4:C24'],['逐时购电与费用','L36:W42'],['调减违约明细','A4:F10']])
  inspections.push((await wb.inspect({kind:'table',range:`'${s}'!${r}`,include:'values,formulas',tableMaxRows:22,tableMaxCols:12,maxChars:5000})).ndjson);
inspections.push((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:3000})).ndjson);
await fs.writeFile(path.join(here,'artifact_inspection.jsonl'),inspections.join('\n'));
for(const [s,r,name] of [['月度汇总','A1:G38','summary'],['逐时购电与费用','A1:K10','purchase'],
  ['逐时购电与费用','L4:W10','fees'],['逐时购电与费用','L36:W42','adjustment_fee'],['每日汇总','A1:O10','daily'],
  ['调减违约明细','A1:F11','penalties'],['供电与预测核验','A1:K10','forecasts'],['供电与预测核验','L4:V10','physics']]){
  const blob=await wb.render({sheetName:s,range:r,scale:1,format:'png'});
  await fs.writeFile(path.join(here,`${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=path.join(output,'问题三_2025年2月_逐时购电与电费.xlsx');
await(await SpreadsheetFile.exportXlsx(wb)).save(file);
await fs.writeFile(path.join(here,'artifact_checks.json'),JSON.stringify({nIntervals:p.n,formulaValuesMatched:true,sheets:5,workbook:file},null,2));
console.log(JSON.stringify({file,nIntervals:p.n,sheets:5}));
