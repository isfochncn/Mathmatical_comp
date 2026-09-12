import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const project='D:/WorkSpace/Project/Mathmatical/Mathmatical_comp';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(`${project}/data/附件5/result3.xlsx`));
console.log(wb.help('range.delete',{include:'index,examples,notes',maxChars:2500}).ndjson);
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:1000})).ndjson);
