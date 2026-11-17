import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url)),workspaceDir=path.dirname(build),study=path.resolve(workspaceDir,'../..');
const output=path.join(workspaceDir,'output'),assets=path.join(output,'実映像画像');
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const featurePath=path.join(study,'analysis/match1_court_plane_preimpact_motion_features.csv');
const lines=(await fs.readFile(featurePath,'utf8')).trim().split(/\r?\n/),header=lines.shift().split(',');
const rows=lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[header[i],v])));
const feature=rows.find(r=>r.event_id==='match1_direction_match1_remaining_01_105'&&r.timepoint_sec==='-0.300');
assert.equal(feature.motion_start_source_frame,'71902');assert.equal(feature.motion_end_source_frame,'71908');
assert.equal(Number(feature.impact_source_frame)-71908,18);
assert.ok(Number(feature.lateral_displacement_court_width)>.015);
const source=path.join(output,'23238007R_中間発表_実画像入り_v4.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const before=await p.inspect({kind:'slide,image,textbox,notes',maxChars:200000});
await fs.writeFile(path.join(build,'before_central.ndjson'),before.ndjson);
const items=before.ndjson.split('\n').filter(Boolean).map(s=>JSON.parse(s));
const slide6=items.filter(r=>r.slide===6);
const figures=slide6.filter(r=>r.kind==='image').sort((a,b)=>a.bbox[0]-b.bbox[0]);assert.equal(figures.length,2);
const files=[];
for(const [i,f] of [71902,71908].entries()){
 const file=path.join(assets,i===0?'画像③-A_中央_前0.4秒_F71902.png':'画像③-B_中央_前0.3秒_F71908.png');
 const r=spawnSync('C:/ffmpeg/bin/ffmpeg.exe',['-y','-hide_banner','-loglevel','info','-copyts','-ss',String((f-60)*1001/60000),'-i',path.join(study,'data/raw/match1.mp4'),'-vf',`select=eq(pts\\,${f*1001}),showinfo,crop=768:432:1500:1050`,'-frames:v','1','-update','1',file],{encoding:'utf8',maxBuffer:1e7});
 assert.equal(r.status,0,r.stderr);assert.match(r.stderr,new RegExp('pts:\\s*'+f*1001+'\\s'));
 await fs.writeFile(path.join(build,`central_F${f}.log`),r.stderr);
 const image=p.resolve(figures[i].id);
 image.replace({blob:await fs.readFile(file),contentType:'image/png',alt:`原動画F${f}，確定インパクトF${feature.impact_source_frame}．同一範囲 x1500 y1050 w768 h432．中央付近の前衛`,fit:'contain'});
 files.push(file);
}
for(const [match,value] of [
 ['F104663','−0.4秒（F71902）'],['F104669','−0.3秒（F71908）'],
 ['例：右へコート幅','例：コート中央付近で右へ'+(Number(feature.lateral_displacement_court_width)*100).toFixed(2)+'％移動']]){
 const r=slide6.find(r=>r.kind==='textbox'&&r.text?.includes(match));assert.ok(r,match);p.resolve(r.id).text=value;
}
const note=slide6.find(r=>r.kind==='notes');
const intro=note.text.split('\n\n出典・定義')[0].replace('例えば0.2秒前の判定には，0.3秒前と0.2秒前の位置の差を使います．','例えば今回の0.3秒前の判定には，0.4秒前と0.3秒前の位置の差を使います．中央の白線を基準にすると，前衛が右へ移動したことを確認できます．');
p.resolve(note.id).setText(intro+'\n\n出典・定義\n'+featurePath+'\n画像：data/raw/match1.mp4．試技 '+feature.event_id+'，確定インパクトF'+feature.impact_source_frame+'．AはF71902（約−0.4004秒），BはF71908（約−0.3003秒）．両画像はx1500 y1050 w768 h432で等倍率の切り出し．フレーム番号は0始まり．コート幅の2.48％は既存の射影変換・3フレーム中央値処理後の横変位0.024770から算出．左の写真は両足を広げた構えに近いが，前後フレームでは小さな移動があるため完全な静止とは呼ばない．初動開始時刻やポーチ成功の例としては扱わない．');
await fs.writeFile(path.join(assets,'画像③_中央例の出典.json'),JSON.stringify({source:'data/raw/match1.mp4',featurePath,feature,frames:[71902,71908],files,crop:[768,432,1500,1050],fps:'60000/1001',interval_seconds:.1001,selection:'Court centre, left frame in a wide ready-like stance, subsequent rightward displacement. The left frame is not claimed to be stationary.',processing:'Exact PTS extraction and crop only. No generative retouching.'},null,2));
const candidatePath=path.join(build,'candidate_central_v1.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(output,'23238007R_中間発表_中央移動例_v5.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit','--require-native-table-slide','7','--require-native-table-slide','8','--require-native-table-slide','11'],requiredNativeTableOwnerSlides:[7,8,11],fontPolicy:{basis:'design',families:['Meiryo']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation_central_v1.json')});
console.log(JSON.stringify({finalPath,layout:result.presentationLayout}));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const render=path.join(build,'render_central_v1');await fs.mkdir(render,{recursive:true});
const sha=b=>createHash('sha256').update(b).digest('hex');
for(let i=0;i<14;i++){
 const img=await final.export({slide:final.slides.items[i],format:'png',scale:1.25});const buf=new Uint8Array(await img.arrayBuffer());
 const filename=`slide-${String(i+1).padStart(2,'0')}.png`;await fs.writeFile(path.join(render,filename),buf);
 if(i!==5)assert.equal(sha(buf),sha(await fs.readFile(path.join(build,'render_images_v2',filename))),'Unrelated slide changed: '+(i+1));
}
console.log('All 13 unrelated slides render identically.');
