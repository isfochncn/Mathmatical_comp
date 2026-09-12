import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

if(process.argv.includes('--help')){
  console.log('Usage: bundled-node tools/build_strict_result.mjs OUTPUT_DIRECTORY [--layout-only]\nOptional ARTIFACT_TOOL_NODE_MODULES points to the bundled node_modules.');
  process.exit(0);
}
// Resolve the bundled runtime directly; never copy or junction dependencies into results.
const modules=process.env.ARTIFACT_TOOL_NODE_MODULES ?? path.resolve(path.dirname(process.execPath),'../node_modules');
const require=createRequire(path.join(modules,'_artifact_resolver.cjs'));
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
if(process.argv.includes('--check-runtime')){console.log('Artifact runtime available');process.exit(0);}
if(!process.argv[2])throw new Error('An output directory is required; see --help.');

const output=path.resolve(process.argv[2]);
const layoutOnly=process.argv.includes('--layout-only');
const p=JSON.parse(await fs.readFile(path.join(output,'result_payload.json'),'utf8'));
const here=path.join(output,'核验');await fs.mkdir(here,{recursive:true});
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(p.template));
const date=s=>Date.parse(`${s}T00:00:00Z`)/86400000+25569;
for(const item of p.sheets){
  const s=wb.worksheets.getItem(item.name);
  const cols=item.columns===147?'EQ':item.columns===6?'F':'C';
  const n=item.values.length;
  s.getRange(`A2:${cols}${item.template_rows}`).clear({applyTo:'contents'});
  if(n&&item.columns!==147){
    const body=s.getRange(`A2:${cols}${n+1}`),edge={style:'thin',color:'#000000'};
    body.format.font=item.body_font;
    body.format.rowHeight=item.body_height;body.format.horizontalAlignment='center';
    body.format.borders={preset:'none'};
    body.format.borders={left:edge,right:edge,insideVertical:edge};
    // Match the template's date-group outlines when expanding its placeholders.
    if(item.name==='充放电量'){
      for(let d=0;d<p.days;d++)s.getRange(`A${2+d*6}:F${7+d*6}`).format.borders={top:edge,bottom:edge};
    }else{
      const starts=item.values.map((r,i)=>r[0]!==null?i+2:null).filter(r=>r!==null);
      for(let k=0;k<starts.length;k++)s.getRange(`A${starts[k]}:C${k+1<starts.length?starts[k+1]-1:n+1}`).format.borders={top:edge,bottom:edge};
    }
  }
  const values=item.values.map(row=>row.map((v,i)=>i===0&&v!==null?date(v):v));
  if(n)s.getRange(`A2:${cols}${n+1}`).values=values;
  // Import/copy does not reliably propagate the date/time number formats to
  // expanded rows. Extend the template's formats explicitly to their columns.
  if(n&&item.columns!==147)s.getRange(`A2:A${n+1}`).setNumberFormat('m/d/yy');
  if(item.name==='充放电量')for(let d=0;d<p.days;d++)s.getRange(`E${2+d*6}`).setNumberFormat('h:mm');
  const header=s.getRange(`A1:${cols}1`).values[0];
  if(JSON.stringify(header)!==JSON.stringify(item.headers))throw new Error(`Changed header: ${item.name}`);
  if(n){
    const saved=s.getRange(`A2:${cols}${n+1}`).values;
    for(let i=0;i<n;i++)for(let j=0;j<item.columns;j++){
      const a=saved[i][j]??null,b=values[i][j];
      if(typeof b==='number'?(typeof a!=='number'||Math.abs(a-b)>1e-7):a!==b)
        throw new Error(`Value mismatch: ${item.name} r${i+2} c${j+1}`);
    }
  }
}
const previews=[];
if(process.env.RESULT_INSPECT_STYLES)console.log((await wb.inspect({kind:'computedStyle',sheetId:'紧急购电量',range:'A1:C14',maxChars:5500,tableMaxRows:14,tableMaxCols:3})).ndjson);
for(const item of p.sheets){
  const ranges=item.columns===147?['A1:H8','EL1:EQ8']:[`A1:${item.columns===6?'F':'C'}${Math.min(14,item.values.length+1)||2}`];
  for(let k=0;k<ranges.length;k++){
    const selected=process.env.RESULT_PREVIEW_SHEETS?.split(',');
    if(selected&&!selected.includes(item.name))continue;
    const blob=await wb.render({sheetName:item.name,range:ranges[k],scale:1.5,format:'png'});
    const file=path.join(here,`${layoutOnly?'layout':'result'}_${item.name}_${k}.png`);
    await fs.writeFile(file,new Uint8Array(await blob.arrayBuffer()));previews.push(file);
  }
}
const errors=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:10},maxChars:1500})).ndjson;
await fs.writeFile(path.join(here,'artifact_errors.jsonl'),errors);
if(!layoutOnly){
  const dest=path.join(output,p.file_name);
  await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
  await fs.writeFile(path.join(here,'artifact_result_checks.json'),JSON.stringify({file:dest,days:p.days,headers_preserved:true,
    values_match_payload:true,sheets:p.sheets.map(s=>s.name),previews},null,2));
  console.log(JSON.stringify({file:dest,days:p.days}));
}else console.log(JSON.stringify({layout_verified_in_memory:true,previews}));
