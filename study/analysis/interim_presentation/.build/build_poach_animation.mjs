import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url)), workspaceDir=path.dirname(build);
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const p=Presentation.create({slideSize:{width:960,height:720}});
const blue='#214F76',orange='#B45309',ink='#182126';
function shape(s,name,geometry,x,y,w,h,fill,line='none',lw=0){return s.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:lw}});}
function text(s,name,v,x,y,w,h,size=38,color=ink,align='left'){const z=shape(s,name,'textbox',x,y,w,h,'none');z.text=v;z.text.style={typeface:'Meiryo',fontSize:size,bold:true,color,alignment:align,verticalAlignment:'middle',autoFit:'none'};return z;}
function player(s,name,label,cx,cy,color){const z=shape(s,name,'ellipse',cx-30,cy-30,60,60,color);z.text=label;z.text.style={typeface:'Meiryo',fontSize:34,bold:true,color:'#FFFFFF',alignment:'center',verticalAlignment:'middle',autoFit:'none'};return z;}
for(const mode of ['early','late','goal']){
 const early=mode==='early', goal=mode==='goal';
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'title',goal?'最適なタイミングを知りたい！':early?'早く出ると，相手に読まれる？':'遅く出ると，ボールに届かない？',40,22,880,65,43,blue);
 text(s,'legend','青：対象側　　橙：相手側　　黒：ボール',40,96,880,55,34);
 shape(s,'court','rect',65,165,500,440,'#F0F6F4','#587F72',3);
 for(const x of [112,518])shape(s,'sideline'+x,'line',x,165,0,440,'none','#587F72',2);
 for(const y of [275,495])shape(s,'service'+y,'line',112,y,406,0,'none','#587F72',2);
 shape(s,'center','line',315,275,0,220,'none','#587F72',2);
 shape(s,'net','line',50,385,530,0,'none','#4B5563',5);
 player(s,'target_front','前',230,438,blue);
 player(s,'near_back','後',450,575,blue);
 player(s,'far_back','後',200,195,orange);
 player(s,'far_front','前',440,335,orange);
 shape(s,'ball','ellipse',194,226,16,16,'#141414','#FFFFFF',2);
 text(s,'step1',goal?'相手に動きを\n読まれにくい':early?'① 打球前に\n　 前衛が動く':'① 相手後衛が\n　 先に打つ',595,177,340,126,38,blue);
 text(s,'step2',goal?'ボールにも\n間に合う':early?'② 相手後衛が\n　 空いた側へ':'② 動き出しが\n　 遅れる',595,320,340,126,38,orange);
 text(s,'result',goal?'相手に応じた\n出方の検討へ':early?'動きを見て\n打ち分ける例':'横切る球に\n届かない例',595,474,340,112,38,ink);
 text(s,'purpose',goal?'この相手には，いつ出ると成功しやすいか？':'相手に読まれにくく，球にも届くタイミングは？',40,621,880,55,34,blue);
 text(s,'disclosure','模式例：位置・速さは説明用であり，測定結果ではない',40,682,880,27,20,'#555555');
 s.speakerNotes.textFrame.setText('スライド表示後に約5秒のアニメーションが自動再生します。編集画面では動きません。次のスライドへは手動で進めます。\n'+(goal?'そこで，相手に動きを読まれにくく，球にも届くタイミングを知りたいと考えています。選手や指導者が，この相手にはいつ出ると成功しやすいかを検討できるようにすることが最終目標です。今回の一致率分析はその第一段階です。':early?'前衛が打球前に動くと，相手後衛がそれを見て空いた側に打ち分ける場合があります。':'動き出しが遅いと，球の方向が分かってもボールに届かない場合があります。')+'\nこの模式例は研究動機を説明するもので，今回のデータで最適なタイミングを求めたという意味ではありません。前後衛を円で表した編集可能な模式図です。');
}
const raw=path.join(build,'poach_animation_raw.pptx'),candidatePath=path.join(build,'poach_animation_candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(raw);
execFileSync(python,[path.join(build,'add_poach_timing.py'),raw,candidatePath]);
const finalPath=path.join(workspaceDir,'output','ポーチのタイミング_説明アニメーション_3枚版.pptx');
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Meiryo']},verifyArtifactToolImport:true,receiptPath:path.join(build,'poach_animation_3_validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<3;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1.25});await fs.writeFile(path.join(build,`poach_animation_${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
console.log(finalPath);
