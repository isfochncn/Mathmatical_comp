import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load('D:/WorkSpace/Project/Mathmatical/Mathmatical_comp/data/附件5/result1.xlsx'));
console.log(wb.help('worksheet.freezePanes', {include:'index,examples,notes',maxChars:2800}).ndjson);
