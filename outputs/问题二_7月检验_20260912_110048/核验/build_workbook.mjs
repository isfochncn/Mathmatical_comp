import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const output = path.dirname(here);
const payload = JSON.parse(await fs.readFile(path.join(output, 'month_payload.json'), 'utf8'));
const wb = Workbook.create();
const summary = wb.worksheets.add('月度汇总');
const detail = wb.worksheets.add('逐时购电与费用');
const daily = wb.worksheets.add('每日汇总');
const check = wb.worksheets.add('供电与预测核验');
const n = payload.n;
const end = n + 4;
const date = text => new Date(text.length === 10 ? `${text}T00:00:00Z` : `${text}Z`);
const normal = '#,##0.000000';
const money = '#,##0.00';
const blue = '#17365D';

function setup(sheet, lastCol, lastRow, title, subtitle) {
  const used = sheet.getRange(`A1:${lastCol}${lastRow}`);
  used.format.font = { name: 'Arial', size: 10, color: '#222222' };
  used.format.rowHeight = 20;
  used.format.columnWidth = 17;
  used.format.verticalAlignment = 'center';
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [[title]];
  sheet.getRange('A1').format.font = { name: 'Arial', size: 15, bold: true, color: '#17365D' };
  sheet.getRange('A1').format.rowHeight = 28;
  sheet.getRange('A2').values = [[subtitle]];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 10, color: '#666666' };
}
function headers(sheet, lastCol, labels) {
  const range = sheet.getRange(`A4:${lastCol}4`);
  range.values = [labels];
  range.format = {fill:blue, font:{name:'Arial', size:10, bold:true, color:'#FFFFFF'},
    wrapText:true, horizontalAlignment:'center', verticalAlignment:'center', rowHeight:44};
  range.format.borders = {insideVertical:{style:'thin',color:'#FFFFFF'},bottom:{style:'thin',color:blue}};
}
function formulaColumn(sheet, col, formula, last=end) {
  sheet.getRange(`${col}5`).formulas = [[formula]];
  if (last > 5) sheet.getRange(`${col}5:${col}${last}`).fillDown();
}
function freeze(sheet) {
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);
}

setup(detail,'Q',end,'问题二：逐时购电与费用','2025年7月归属行；普通购电日内冻结，外网总购电包含紧急补购。');
headers(detail,'Q',['开始时间','结束时间','归属日期','日内时段','计划购电\n(kWh)','实际普通购电\n(kWh)',
  '普通调整差额\n(kWh)','紧急购电\n(kWh)','实际外网总购电\n(kWh)','普通电价\n(元/kWh)',
  '普通电费\n(元)','紧急单价\n(元/kWh)','紧急电费\n(元)','本段总电费\n(元)','电费标签','弃购电\n(kWh)','购电利用量\n(kWh)']);
const purchaseRows = payload.rows.map(row => row.map((v,i) => i < 3 ? date(v) : v));
detail.getRange(`A5:Q${end}`).values = purchaseRows;
detail.getRange(`D5:Q${end}`).setNumberFormat(normal);
detail.getRange(`A5:B${end}`).setNumberFormat('yyyy-mm-dd hh:mm');
detail.getRange(`C5:C${end}`).setNumberFormat('yyyy-mm-dd');
detail.getRange(`D5:D${end}`).setNumberFormat('0');
detail.getRange(`J5:J${end}`).setNumberFormat('0.0000');
detail.getRange(`L5:L${end}`).setNumberFormat('0.0000');
for (const col of ['K','M','N']) detail.getRange(`${col}5:${col}${end}`).setNumberFormat(money);
for (const [col, formula] of Object.entries({G:'=F5-E5',I:'=F5+H5',K:'=F5*J5',
    L:"=J5*'月度汇总'!$G$8",M:'=H5*L5',N:'=K5+M5',O:'=TEXT(N5,"0.00")&"元"',Q:'=I5-P5'})) {
  formulaColumn(detail,col,formula);
}
detail.getRange(`O5:O${end}`).setNumberFormat('@');
detail.getRange('A:B').format.columnWidth = 23;
detail.getRange('C:C').format.columnWidth = 16;
detail.getRange('D:D').format.columnWidth = 11;
detail.getRange('I:I').format.columnWidth = 21;
detail.getRange('O:O').format.columnWidth = 18;
detail.getRange(`N5:N${end}`).format.font = {name:'Arial',size:10,bold:true,color:blue};
detail.getRange(`H5:H${end}`).conditionalFormats.add('cellIs',{
  operator:'greaterThan', formula:0.000001, format:{fill:'#FFF2CC',font:{color:'#9C5700'}}});
detail.getRange(`G5:G${end}`).conditionalFormats.addCustom('ABS(G5)>0.000001',
  {fill:'#FCE4D6',font:{color:'#9C0006'}});
detail.tables.add(`A4:Q${end}`,true,'PurchaseAndFee').showFilterButton = true;
freeze(detail);

setup(check,'V',end,'问题二：供电与预测核验','预测、实测与储能按同一开始时间对应；误差为实际净负荷减预测净负荷。');
headers(check,'V',['开始时间','日计划制定时间','计划负荷预测\n(kWh)','计划光伏预测\n(kWh)','实际负荷\n(kWh)',
  '实际光伏\n(kWh)','实际净负荷\n(kWh)','计划净负荷误差\n(kWh)','执行时负荷预测\n(kWh)','执行时光伏预测\n(kWh)',
  '执行时净负荷误差\n(kWh)','段初SOC\n(kWh)','段末SOC\n(kWh)','实际存入\n(kWh)','实际送达\n(kWh)',
  '弃光\n(kWh)','弃购电\n(kWh)','当段累计备用需求\n(送达kWh)','窗口最大规划备用缺口\n(kWh)',
  '当段规划供电缺口\n(kWh)','母线平衡残差\n(kWh)','SOC递推残差\n(kWh)']);
check.getRange(`A5:V${end}`).values = payload.check_rows.map(row => row.map((v,i) => i < 2 ? date(v) : v));
check.getRange(`C5:V${end}`).setNumberFormat(normal);
check.getRange(`A5:B${end}`).setNumberFormat('yyyy-mm-dd hh:mm');
check.getRange('A:B').format.columnWidth = 23;
check.getRange('H:K').format.columnWidth = 21;
check.getRange('R:T').format.columnWidth = 25;
check.getRange('U:V').format.columnWidth = 21;
for (const [col, formula] of Object.entries({A:"='逐时购电与费用'!A5",G:'=E5-F5',H:'=G5-C5+D5',K:'=G5-I5+J5',
  U:"='逐时购电与费用'!F5+'逐时购电与费用'!H5+F5-P5+O5-E5-N5/'月度汇总'!$G$6-Q5",
  V:"=M5-L5-N5+O5/'月度汇总'!$G$7"})) formulaColumn(check,col,formula);
for (const col of ['U','V']) check.getRange(`${col}5:${col}${end}`).conditionalFormats.addCustom(`ABS(${col}5)>0.00001`,
  {fill:'#FCE4D6',font:{color:'#9C0006'}});
check.tables.add(`A4:V${end}`,true,'SupplyAndForecast').showFilterButton = true;
freeze(check);

const days = n/144;
setup(daily,'L',days+4,'问题二：每日购电与费用','每个日期汇总当日00:10至次日00:10的144个时段。');
headers(daily,'L',['归属日期','计划购电\n(kWh)','实际普通购电\n(kWh)','紧急购电\n(kWh)','实际总购电\n(kWh)',
  '普通电费\n(元)','紧急电费\n(元)','总电费\n(元)','弃购电\n(kWh)','期初SOC\n(kWh)','期末SOC\n(kWh)','紧急购电时段数']);
const dailyFormulas = [];
for (let d=0; d<days; d++) {
  const a=5+d*144, b=a+143;
  dailyFormulas.push([`='逐时购电与费用'!C${a}`, ...['E','F','H','I','K','M','N','P'].map(c=>`=SUM('逐时购电与费用'!${c}${a}:${c}${b})`),
    `='供电与预测核验'!L${a}`, `='供电与预测核验'!M${b}`, `=COUNTIF('逐时购电与费用'!H${a}:H${b},">0.000001")`]);
}
daily.getRange(`A5:L${days+4}`).formulas = dailyFormulas;
daily.getRange(`A5:A${days+4}`).setNumberFormat('yyyy-mm-dd');
daily.getRange(`B5:K${days+4}`).setNumberFormat('#,##0.000');
daily.getRange(`F5:H${days+4}`).setNumberFormat(money);
daily.getRange(`L5:L${days+4}`).setNumberFormat('0');
daily.getRange('A:A').format.columnWidth = 17;
daily.tables.add(`A4:L${days+4}`,true,'DailyTotals').showFilterButton = true;
daily.freezePanes.freezeRows(4);

setup(summary,'H',28,'问题二：2025年7月检验','本次从2025年1月1日00:10、6000 kWh连续运行至8月1日00:10。');
headers(summary,'D',['指标','本次模型','旧版同期','本次减旧版']);
summary.getRange('A:A').format.columnWidth=31;
summary.getRange('B:D').format.columnWidth=24;
summary.getRange('E:E').format.columnWidth=4;
summary.getRange('F:F').format.columnWidth=24;
summary.getRange('G:G').format.columnWidth=24;
summary.getRange('H:H').format.columnWidth=58;
summary.getRange('F5:H5').values=[['参数或口径','取值','说明']];
summary.getRange('F5:H5').format={fill:blue,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
summary.getRange('F6:H16').values=[
  ['充电效率',.9,'实际存入量与母线输入量之比'],
  ['放电效率',.9,'实际送达量与电池减少量之比'],
  ['紧急电价倍率',5,'紧急单价＝本段普通电价×5'],
  ['风险经验分位',.90,'历史净负荷误差用于计算备用'],
  ['时段长度',10,'分钟'],
  ['SOC下限',1200,'kWh'],
  ['SOC上限',10800,'kWh'],
  ['单段最大送达放电',750,'kWh'],
  ['年初储电量',6000,'kWh；7月不重新设置初始电量'],
  ['全年执行起点',date(payload.execution_start),'跳过年初缺失的00:00—00:10'],
  ['本月时段数',n,'31日×144段']];
summary.getRange('G6:G7').setNumberFormat('0.0%');
summary.getRange('G9').setNumberFormat('0.0%');
summary.getRange('G15').setNumberFormat('yyyy-mm-dd hh:mm');
summary.getRange('F18:H26').values=[
  ['月度开始',date(payload.from),'含此起点'],
  ['月度结束',date(payload.to),'不含此终点'],
  ['问题二计划','普通购电日内冻结','普通调整量应为0；紧急购电单独列示'],
  ['购电量口径','外网总购电＝普通＋紧急','弃购电仍已付费；购电利用量另列'],
  ['费用精度','原始精度求和','电费标签显示2位小数；汇总不先逐段舍入'],
  ['风险费用','仅用于规划权衡','规划风险惩罚不计入本工作簿电费'],
  ['旧版对照','旧版全年问题二轨迹','两版本均由年初运行；7月初SOC可能不同'],
  ['来源','附件1、附件2及运行记录','价格重复附件1；负荷与光伏实测来自附件2'],
  ['备用缺口','规划要求与能力的差额','是预测窗口指标，不等于实际缺电量']];
summary.getRange('G18:G19').setNumberFormat('yyyy-mm-dd hh:mm');
summary.getRange('F18:H26').format.wrapText=true;
summary.getRange('F18:H26').format.rowHeight=40;
const metrics = [
  ['计划购电量 (kWh)','plan_kwh',`=SUM('逐时购电与费用'!E5:E${end})`],
  ['实际普通购电量 (kWh)','ordinary_kwh',`=SUM('逐时购电与费用'!F5:F${end})`],
  ['紧急购电量 (kWh)','emergency_kwh',`=SUM('逐时购电与费用'!H5:H${end})`],
  ['实际外网总购电量 (kWh)','total_kwh',`=SUM('逐时购电与费用'!I5:I${end})`],
  ['普通电费 (元)','ordinary_cost',`=SUM('逐时购电与费用'!K5:K${end})`],
  ['紧急电费 (元)','emergency_cost',`=SUM('逐时购电与费用'!M5:M${end})`],
  ['月度总电费 (元)','total_cost',`=SUM('逐时购电与费用'!N5:N${end})`],
  ['弃购电量 (kWh)','spill_kwh',`=SUM('逐时购电与费用'!P5:P${end})`],
  ['月初SOC (kWh)','soc_start',"='供电与预测核验'!L5"],
  ['月末SOC (kWh)','soc_end',`='供电与预测核验'!M${end}`],
  ['紧急购电时段数','emergency_intervals',`=COUNTIF('逐时购电与费用'!H5:H${end},">0.000001")`],
  ['普通计划变更时段数','revised_intervals',`=COUNTIF('逐时购电与费用'!G5:G${end},">0.000001")+COUNTIF('逐时购电与费用'!G5:G${end},"<-0.000001")`]
];
for (let i=0;i<metrics.length;i++) {
  const r=5+i, [label,key,formula]=metrics[i];
  summary.getRange(`A${r}:D${r}`).values=[[label,null,payload.legacy[key],null]];
  summary.getRange(`B${r}`).formulas=[[formula]];
  summary.getRange(`D${r}`).formulas=[[`=B${r}-C${r}`]];
}
summary.getRange('B5:D16').setNumberFormat('#,##0.000');
summary.getRange('B9:D11').setNumberFormat(money);
summary.getRange('B15:D16').setNumberFormat('0');
summary.getRange('A11:D11').format.fill='#E9EFF7';
summary.getRange('A11:D11').format.font={name:'Arial',size:10,bold:true,color:blue};

// Independent value checks against the executed dispatch records.
const checkClose=(actual,expected,label,tol=1e-5)=>{
  if (typeof actual!=='number' || !Number.isFinite(actual) || Math.abs(actual-expected)>tol)
    throw new Error(`${label}: ${actual} != ${expected}`);
};
for (let i=0;i<metrics.length;i++) checkClose(summary.getRange(`B${5+i}`).values[0][0],payload.current[metrics[i][1]],metrics[i][1],1e-4);
for (const [col,index] of [['G',6],['I',8],['K',10],['L',11],['M',12],['N',13],['Q',16]]) {
  const values=detail.getRange(`${col}5:${col}${end}`).values;
  for(let i=0;i<n;i++) checkClose(values[i][0],payload.rows[i][index],`${col}${i+5}`);
}
for(const col of ['U','V']) for(const row of check.getRange(`${col}5:${col}${end}`).values) checkClose(row[0],0,col,1e-5);
const inspections=[];
for(const [sheet,range] of [['月度汇总','A4:D16'],['逐时购电与费用','J4:Q10'],['逐时购电与费用',`A${end-2}:Q${end}`],['每日汇总','A4:L8'],['供电与预测核验','R4:V10']]) {
  inspections.push((await wb.inspect({kind:'table',range:`'${sheet}'!${range}`,include:'values,formulas',tableMaxRows:14,tableMaxCols:17,maxChars:5000})).ndjson);
}
inspections.push((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000})).ndjson);
await fs.writeFile(path.join(here,'artifact_inspection.jsonl'),inspections.join('\n'));
for(const [sheet,range,name] of [['月度汇总','A1:H26','summary'],['逐时购电与费用','A1:I10','purchase'],
  ['逐时购电与费用','J4:Q12','fees'],['每日汇总','A1:L10','daily'],['供电与预测核验','A1:K10','forecasts'],
  ['供电与预测核验','L4:V12','physics']]) {
  const blob=await wb.render({sheetName:sheet,range,scale:1,format:'png'});
  await fs.writeFile(path.join(here,`${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=path.join(output,'问题二_2025年7月_逐时购电与电费.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(file);
await fs.writeFile(path.join(here,'artifact_checks.json'),JSON.stringify({formulaValuesMatched:true,nIntervals:n,sheets:4,workbook:file},null,2));
console.log(JSON.stringify({file,nIntervals:n,sheets:4}));
