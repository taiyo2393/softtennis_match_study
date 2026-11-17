import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const buildDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.dirname(buildDir);
const projectDir = path.resolve(workspaceDir, '../..');
const outputDir = path.join(workspaceDir, 'output');
const skillDir = 'C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python = 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE = process.execPath;
process.env.RUNTIME_PYTHON = python;
const { finalizePresentation, resolvePresentationFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({fontFamily: 'Meiryo', availableFonts: ['Meiryo']});

function csvParse(text) {
  const rows=[]; let row=[], cell='', quoted=false;
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(c==='"') { if(quoted && text[i+1]==='"'){cell+='"';i++;} else quoted=!quoted; }
    else if(c===',' && !quoted){row.push(cell);cell='';}
    else if(c==='\n' && !quoted){row.push(cell.replace(/\r$/,''));if(row.some(Boolean))rows.push(row);row=[];cell='';}
    else cell+=c;
  }
  if(cell || row.length){row.push(cell.replace(/\r$/,''));rows.push(row);}
  const headers=rows.shift().map(x=>x.replace(/^\uFEFF/,''));
  return rows.map(values=>{assert.equal(values.length,headers.length);return Object.fromEntries(headers.map((h,i)=>[h,values[i]]));});
}
const source = path.join(projectDir,'analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv');
const identitySource = path.join(projectDir,'annotations/match1_target_front_identity_review.csv');
const pooledSource = path.join(projectDir,'analysis/match1_court_plane_directional_motion_t0p015_summary.csv');
const events=csvParse(await fs.readFile(source,'utf8'));
const identities=csvParse(await fs.readFile(identitySource,'utf8'));
const pooled=csvParse(await fs.readFile(pooledSource,'utf8'));
const identityById=new Map(identities.map(x=>[x.event_id,x]));
assert.equal(events.length,392);
assert.equal(new Set(events.map(x=>x.event_id)).size,98);
assert.equal(new Set(events.map(x=>x.event_id+'@'+x.timepoint_sec)).size,392);
for(const e of events){
  const id=identityById.get(e.event_id);
  assert.equal(id.identity_status,'confirmed');
  for(const key of ['target_front_player_id','target_front_camera_end','impact_source_frame'])assert.equal(e[key],id[key]);
  const dx=Number(e.lateral_displacement_court_width);
  const pos=e.target_front_camera_end==='near'?'front_right':'front_left';
  const neg=pos==='front_right'?'front_left':'front_right';
  const movement=dx>=.015?pos:dx<=-.015?neg:'stay';
  assert.equal(movement,e.movement_direction_target_front_view);
  assert.equal(e.direction_matches_actual,movement==='stay'?'':String(Number(movement===e.actual_direction_target_front_view)));
}
const times=[-.5,-.3,-.2,-.1];
const groups=['P1','P2','ALL'];
function stats(player,t){
  const rows=events.filter(e=>(player==='ALL'||e.target_front_player_id===player)&&Number(e.timepoint_sec)===t);
  const n=rows.length, d=rows.filter(e=>e.movement_is_directional==='1').length;
  const k=rows.filter(e=>e.direction_matches_actual==='1').length;
  return {player,t,n,d,k,agreement:d?k/d:null,coverage:d/n};
}
const data=times.map(t=>groups.map(g=>stats(g,t)));
for(const row of data){
  const [p1,p2,all]=row;
  for(const key of ['n','d','k'])assert.equal(p1[key]+p2[key],all[key]);
  assert.equal(p1.n,42);assert.equal(p2.n,56);
  const existing=pooled.find(x=>Number(x.timepoint_sec)===all.t);
  assert.equal(all.n,Number(existing.events));
  assert.equal(all.d,Number(existing.directional_events));
  assert.equal(all.k,Number(existing.matching_direction_events));
  assert.equal(all.agreement.toFixed(3),existing.conditional_match_rate);
}
await fs.writeFile(path.join(buildDir,'verified_aggregates.json'),JSON.stringify({source,identitySource,pooledSource,threshold:.015,data},null,2));

const presentation=Presentation.create({slideSize:{width:960,height:720}});
const titles=['選手別・全体の方向の一致率','方向を判定できた試技の割合'];
const subtitles=['コート幅1.5％以上の横移動を判定','コート幅1.5％未満の移動は左右判定の対象外'];
const footers=[
  '数値：一致率（一致数／判定試技数）\n0秒＝相手後衛のインパクト．手前・奥を統合．',
  '数値：判定できた割合（判定試技数／全試技数）\n判定対象は各時点で異なる．手前・奥を統合．',
];
function text(slide,value,x,y,w,h,size,bold=false){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=value;
  s.text.style={typeface:font,fontSize:size,bold,color:'#151515',autoFit:'none',verticalAlignment:'middle'};
  return s;
}
const headers=['時点\n（秒）','P1\n全42試技','P2\n全56試技','全体\n全98試技'];
for(let index=0;index<2;index++){
  const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
  text(slide,titles[index],55,34,850,66,40,true);
  text(slide,subtitles[index],55,105,850,40,25);
  const values=[headers,...data.map((row,i)=>[
    '−'+Math.abs(times[i]).toFixed(1),
    ...row.map(s=>index===0?`${(s.agreement*100).toFixed(1)}％\n(${s.k} / ${s.d})`:`${(s.coverage*100).toFixed(1)}％\n(${s.d} / ${s.n})`),
  ])];
  const table=slide.tables.add({rows:5,columns:4,left:55,top:154,width:850,height:440,columnWidths:[175,225,225,225],values});
  table.styleOptions={headerRow:false,bandedRows:false,firstColumn:false,lastColumn:false};
  table.borders.assign({style:'solid',fill:'#D2D2D2',width:1});
  for(let r=0;r<5;r++){
    table.rows[r].height=88;
    for(let c=0;c<4;c++){
      const cell=table.getCell(r,c);
      cell.fill=r===0?'#EEEEEE':'#FFFFFF';
      cell.text.style={typeface:font,fontSize:r===0?27:31,bold:r===0,color:'#151515',alignment:'center',verticalAlignment:'middle',autoFit:'none'};
    }
  }
  text(slide,footers[index],55,620,850,72,22);
  slide.speakerNotes.textFrame.setText([
    '出典：'+source, '選手ラベル：'+identitySource,
    '全体照合：'+pooledSource,
    '2025年全日本選手権男子決勝の単一試合98試技．P1/P2は確認済み選手IDであり実名への対応は未指定．',
    '直前約0.1秒間の射影変換後の横変位で判定．閾値0.015コート幅．本人から見た左右に統一．',
    '一致率の分母は各時点で閾値を満たした試技．全試技に対する予測正解率でもポーチ成功率でもない．',
    '観察された選手別の推移であり、有意差や最適な開始時刻は検証していない．',
  ].join('\n'));
}
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(outputDir,'選手別・全体_結果表_v3.pptx');
const result=await finalizePresentation({
  workspaceDir,candidatePath,finalPath,pythonExecutable:python,
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit','--require-native-table-slide','1','--require-native-table-slide','2'],
  requiredNativeTableOwnerSlides:[1,2],
  fontPolicy:{basis:'design',families:[font]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,'validation_v3.json'),
});
console.log(JSON.stringify(result));
const verified=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<2;i++){
  const slide=verified.slides.items[i];
  const preview=await verified.export({slide,format:'png',scale:2});
  await fs.writeFile(path.join(outputDir,i===0?'選手別・全体_一致率.png':'補足_方向判定率.png'),new Uint8Array(await preview.arrayBuffer()));
}
console.log(finalPath);
