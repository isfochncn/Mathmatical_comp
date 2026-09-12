import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.dirname(here);
const project=path.resolve(root,'../..');
const mode=process.argv[2] || 'templates';
const files=['result1.xlsx','result2.xlsx','result3.xlsx','result4-2.xlsx','result4-3.xlsx'];
for(const file of files){
  const target=mode==='templates'?path.join(project,'data','附件5',file):path.join(root,'result',file);
  try {await fs.access(target);} catch {continue;}
  const prior=path.join(here,`${mode}_${file}_计划购电量.png`);
  try {if ((await fs.stat(prior)).mtimeMs >= (await fs.stat(target)).mtimeMs) {console.log(mode,file,'already inspected'); continue;}} catch {}
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(target));
  const names=['计划购电量',...(['result3.xlsx','result4-3.xlsx'].includes(file)?['调整购电量']:[]),'充放电量',...(file==='result1.xlsx'?[]:['紧急购电量'])];
  for(const sheet of names){
    const range=file==='result1.xlsx'?'A1:F8':'A1:H8';
    const inspected=await wb.inspect({kind:'table',range:`'${sheet}'!${range}`,include:'values,formulas',maxChars:1600,tableMaxRows:4,tableMaxCols:5});
    await fs.writeFile(path.join(here,`${mode}_${file}_${sheet}.jsonl`),inspected.ndjson);
    const image=await wb.render({sheetName:sheet,range,scale:1,format:'png'});
    await fs.writeFile(path.join(here,`${mode}_${file}_${sheet}.png`),new Uint8Array(await image.arrayBuffer()));
  }
  console.log(mode,file,'inspected and rendered');
}
