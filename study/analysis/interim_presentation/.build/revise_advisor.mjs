import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url)),workspaceDir=path.dirname(build);
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const p=Presentation.create({slideSize:{width:960,height:720}}),blue='#214F76',ink='#182126';
function text(s,v,x,y,w,h,size=38,color=ink){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=v;q.text.style={typeface:'Meiryo',fontSize:size,bold:true,color,verticalAlignment:'middle',autoFit:'none'};return q;}
function slide(title,page){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,40,24,880,74,42,blue);text(s,'23238007　朝隈 大耀',40,678,740,27,18);text(s,String(page),860,667,60,40,28);return s;}
function table(s,v,widths,y,rh){const t=s.tables.add({rows:v.length,columns:v[0].length,left:40,top:y,width:880,height:v.length*rh,columnWidths:widths,values:v});t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({style:'solid',fill:'#8093A1',width:1});for(let r=0;r<v.length;r++){t.rows[r].height=rh;for(let c=0;c<v[0].length;c++){const q=t.getCell(r,c);q.fill=r===0?'#EAF0F4':'#FFFFFF';q.text.style={typeface:'Meiryo',fontSize:38,bold:true,color:ink,alignment:'center',verticalAlignment:'middle',autoFit:'none'};}}}
{
 const s=slide('手法の概要：入力と出力',6);
 text(s,'前衛の予測に関係する動作を分析',40,112,880,60,38,blue);
 table(s,[['項目','今回の手法'],['入力','前衛の位置系列\n相手の打球時刻・方向'],['処理','移動方向と打球方向を照合'],['出力','時点別・選手別の一致率']],[165,715],190,102);
 text(s,'将来の動作を出力する予測モデルは未実装',40,618,880,53,34,blue);
}
{
 const s=slide('新規性の着眼点と目指す活用',7);
 table(s,[['比較対象','分析の着眼点'],['先行研究［3］\n（1987）','動きの種類・出現傾向'],['本研究\n（今回）','打球前の時点別・選手別に\n打球方向との対応を数値化']],[280,600],155,130);
 text(s,'選手：相手に応じた出方を検討\n指導者：映像と時刻で具体的に助言',40,560,880,105,38,blue);
}
{
 const s=slide('考察：時点によって集計対象が異なる',15);
 table(s,[['評価時点','集計した試技'],['−0.2秒','15試技'],['−0.1秒','22試技']],[340,540],145,95);
 text(s,'両時点で共通する試技は11件',40,450,880,58,40,blue);
 text(s,'同じ試技だけを比較した一致率ではない',40,523,880,52,36);
 text(s,'不一致にはフェイント・戻りの可能性',40,596,880,52,36);
}
{
 const s=slide('今後の課題',16);
 text(s,'複数試合でデータを増やす\n同じ試技で時点別の一致率を比較する',55,170,850,125);
 text(s,'ポーチの開始時刻と成功・失敗を記録する\n選手・対戦相手・開始位置の違いを比較する',55,370,850,140);
}
{
 const s=slide('まとめと目指す成果',17);
 text(s,'今回',45,135,870,55,38,blue);
 text(s,'打球前の方向の一致率を\n時点別・選手別に示した',45,205,870,110,40);
 text(s,'卒論で目指すこと',45,370,870,55,38,blue);
 text(s,'この相手には，いつ出ると成功しやすいか\n映像と数値で検討できるようにする',45,440,870,120,38);
}
{
 const s=slide('参考文献',18);
 text(s,'［1］Jocher, G., Qiu, J.  Ultralytics YOLO11\nhttps://github.com/ultralytics/ultralytics',45,130,875,90,27);
 text(s,'［2］Hartley, R., Zisserman, A.\nMultiple View Geometry in Computer Vision, 2nd ed., 2004.',45,240,875,90,27);
 text(s,'［3］井箟 敬・出村 慎一・山本 裕二・工藤 敏巳\n軟式テニスの前衛の動きの分析：\nモーションの類型化と出現傾向\n金沢大学教養部論集 人文科学篇, 24(2), 141–156, 1987.',45,350,875,155,26);
 text(s,'映像：2025年 全日本選手権 男子決勝\nhttps://www.youtube.com/watch?v=dLcEKFyCyCs',45,560,875,90,27);
}
await(await PresentationFile.exportPptx(p)).save(path.join(build,'advisor_donor.pptx'));
execFileSync(python,[path.join(build,'patch_advisor.py')],{stdio:'inherit'});
const candidatePath=path.join(build,'advisor_candidate.pptx'),finalPath=path.join(workspaceDir,'output','23238007R_MID_先生指摘対応_v11.pptx');
const owners=[6,7,8,10,11,12,13,14,15,19];
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:owners,verifyArtifactToolImport:true,receiptPath:path.join(build,'advisor_validation_v11.json')});
const f=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir(path.join(build,'advisor_final'),{recursive:true});
for(let i=0;i<f.slides.items.length;i++){const b=await f.export({slide:f.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,'advisor_final',`${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
console.log(finalPath);
