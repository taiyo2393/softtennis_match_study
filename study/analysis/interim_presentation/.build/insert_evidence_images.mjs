import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url));
const workspaceDir=path.dirname(build);
const output=path.join(workspaceDir,'output');
const study=path.resolve(workspaceDir,'../..');
const images=path.join(output,'実映像画像');
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font='Meiryo';
const source=path.join(output,'23238007R_中間発表_画像差し替え版_v2.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot=await p.inspect({kind:'slide,shape,textbox,notes',maxChars:200000});
const rows=snapshot.ndjson.split('\n').filter(Boolean).map(s=>JSON.parse(s));
function resolve(id){assert.ok(rows.some(r=>r.id===id),id);return p.resolve(id);}
function csv(text){const lines=text.trim().split(/\r?\n/),h=lines.shift().replace(/^\uFEFF/,'').split(',');return lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[h[i],v])));}
const poseSource=path.join(study,'data/pose/target_front/match1_full_match_near_front_final_01_pose_yolo11m.csv');
const trackSource=path.join(study,'data/tracking/front_target_highres/match1_full_match_near_front_final_01_tracks_yolo11m.csv');
const featureSource=path.join(study,'analysis/match1_court_plane_preimpact_motion_features.csv');
const pose=csv(await fs.readFile(poseSource,'utf8')).find(r=>r.event_id==='match1_direction_match1_remaining_01_003'&&r.source_frame==='399');
const track=csv(await fs.readFile(trackSource,'utf8')).find(r=>r.event_id===pose.event_id&&r.source_frame===pose.source_frame);
assert.equal(pose.ankle_pair_visible,'1');assert.equal(track.visible,'1');
const feature=csv(await fs.readFile(featureSource,'utf8')).find(r=>r.event_id==='match1_direction_pilot_01_020'&&r.timepoint_sec==='-0.200');
assert.equal(feature.impact_source_frame,'104681');
assert.equal(feature.motion_start_source_frame,'104663');assert.equal(feature.motion_end_source_frame,'104669');
assert.ok(Number(feature.lateral_displacement_court_width)>.015);

// Scientific overlay only: markers reproduce the stored numeric detections.
// No illustrative asset generation, inference rerun, or retouching of people.
const evidence=Presentation.create({slideSize:{width:960,height:540}});
const e=evidence.slides.add();
e.images.add({blob:await fs.readFile(path.join(build,'image_candidates/pose399_raw_crop.png')),contentType:'image/png',alt:'原動画 F399，切り出し x1095 y975 w960 h540',fit:'contain',position:{left:0,top:0,width:960,height:540}});
e.shapes.add({geometry:'rect',position:{left:Number(track.x1_px)-1095,top:Number(track.y1_px)-975,width:Number(track.x2_px)-Number(track.x1_px),height:Number(track.y2_px)-Number(track.y1_px)},fill:'none',line:{fill:'#FFE000',width:5}});
for(const side of ['left','right']){
 const x=Number(pose[side+'_ankle_x_px'])-1095,y=Number(pose[side+'_ankle_y_px'])-975;
 assert.ok(x>15&&x<945&&y>15&&y<525);
 e.shapes.add({geometry:'ellipse',position:{left:x-14,top:y-14,width:28,height:28},fill:'none',line:{fill:'#FF38DD',width:5}});
}
const annotated='画像②_足首追跡_4K原画から_F399.png';
const evpng=await evidence.export({slide:e,format:'png',scale:1});
await fs.writeFile(path.join(images,annotated),new Uint8Array(await evpng.arrayBuffer()));
await(await PresentationFile.exportPptx(evidence)).save(path.join(build,'pose_overlay_editable.pptx'));
const s2=resolve('sl/hwbqtkby'),s5=resolve('sl/i107q5of'),s6=resolve('sl/x8f69ofe');
function clear(ids){for(const id of ids){const sh=resolve(id);sh.text='';sh.fill='none';sh.line={fill:'none',width:0};}}
function text(s,value,x,y,w,h,size=24,color='#214F76',fill='none'){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});sh.text=value;
 sh.text.style={typeface:font,fontSize:size,bold:false,color,autoFit:'none',verticalAlignment:'middle',alignment:'left'};return sh;
}
async function addImage(s,name,x,y,w,h,alt){s.images.add({blob:await fs.readFile(path.join(images,name)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
clear(['sh/ozy1ofad','sh/b29kza94','sh/a10jqpsj']);
await addImage(s2,'画像①_コート全景_F145668.png',50,145,560,315,'2025年全日本選手権男子決勝，原動画F145668．4人の位置関係');
text(s2,'前衛',215,269,100,40,24,'#214F76','#FFFFFF');
text(s2,'後衛',373,411,100,40,24,'#214F76','#FFFFFF');
text(s2,'分析対象の試合映像（手前側の役割を表示）',50,467,565,38,20,'#54616A');

clear(['sh/032tgr6d','sh/n6dcr65o','sh/m5kbi1oj']);
await addImage(s5,annotated,480,284,425,239,'原動画F399の対象前衛．黄枠は既存の人物追跡結果，紫丸は保存済みの左右足首推定位置');
text(s5,'黄枠：対象前衛　紫丸：両足首',480,523,425,24,19,'#54616A');

clear(['sh/ml07i9sv','sh/oryp8fah','sh/pc76hkr2','sh/w32twb6l']);
const example=resolve('sh/98rqt4r6');example.text='例：右へコート幅の'+(Number(feature.lateral_displacement_court_width)*100).toFixed(2)+'％移動（白い服の前衛）';example.position={left:55,top:247,width:850,height:35};
for(const [id,value,x] of [['sh/7m98ru9g','−0.3秒（F104663）',55],['sh/h4bupgn6','−0.2秒（F104669）',505]]){
 const sh=resolve(id);sh.text=value;sh.position={left:x,top:281,width:400,height:30};sh.text.style={typeface:font,fontSize:23,bold:true,color:'#214F76',alignment:'center',verticalAlignment:'middle',autoFit:'none'};
}
await addImage(s6,'画像③-A_インパクト前0.3秒_F104663.png',55,313,400,225,'同一試技のF104663，インパクトF104681の18フレーム前．800x450，x780 y1060の共通切り出し');
await addImage(s6,'画像③-B_インパクト前0.2秒_F104669.png',505,313,400,225,'同一試技のF104669，インパクトF104681の12フレーム前．Aと同じ範囲・倍率');
const noteAdditions={
 2:'画像出典：保存済み原動画 data/raw/match1.mp4，F145668（0始まり）．前衛・後衛のラベルは手前側の2選手を示す．',
 5:'画像出典：原動画F399をx1095 y975 w960 h540で切り出した．黄枠は '+trackSource+'，紫丸は '+poseSource+' の該当イベント・フレームの座標．画像の人物や足首位置を生成AIで描き直していない．追跡イベント '+pose.event_id+'．',
 6:'画像出典：原動画 match1.mp4，イベント '+feature.event_id+'．確定インパクトF104681，左F104663（−0.3003秒），右F104669（−0.2002秒）．双方ともx780 y1060 w800 h450で切り出した．変位3.86％は既存の射影変換・3フレーム中央値処理後の値0.038573に基づく．写真そのものの画像座標差や単純な足首中点差ではない．動きのある原映像のためブレが残るが，補完・鮮明化は行っていない．'
};
for(const n of [2,5,6]){
 const rec=rows.find(r=>r.kind==='notes'&&r.slide===n);const no=resolve(rec.id);
 no.setText(rec.text.replace('画像①はユーザー提供予定．','').replace('画像②はユーザー提供予定．','')+'\n\n'+noteAdditions[n]);
}
const provenance=JSON.parse(await fs.readFile(path.join(images,'画像出典.json'),'utf8'));
provenance.entries.push({file:path.join(images,annotated),source:path.join(study,'data/raw/match1.mp4'),source_frame_zero_based:399,crop_width_height_x_y:[960,540,1095,975],pose_source:poseSource,track_source:trackSource,pose,track,processing:'Raw source frame with numeric pose and box annotations rendered as native evidence objects. No generative enhancement or inference rerun.'});
await fs.writeFile(path.join(images,'画像出典_高解像度版.json'),JSON.stringify(provenance,null,2));
await fs.writeFile(path.join(build,'after_images.ndjson'),(await p.inspect({kind:'slide,shape,textbox,image,notes',maxChars:200000})).ndjson);
const candidatePath=path.join(build,'candidate_images_v2.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(output,'23238007R_中間発表_実画像入り_v4.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit','--require-native-table-slide','7','--require-native-table-slide','8','--require-native-table-slide','11'],
 requiredNativeTableOwnerSlides:[7,8,11],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation_images_v2.json')});
console.log(JSON.stringify(result));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderDir=path.join(build,'render_images_v2');await fs.mkdir(renderDir,{recursive:true});
for(let i=0;i<final.slides.items.length;i++){
 const png=await final.export({slide:final.slides.items[i],format:'png',scale:1.25});
 await fs.writeFile(path.join(renderDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
