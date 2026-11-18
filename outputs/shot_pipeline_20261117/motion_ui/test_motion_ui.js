"use strict";
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
require('./app.js');
const C=globalThis.MotionCore;
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'data.json'),'utf8'));
const count=data.items.length, storageKey='softtennis-motion-human-20261118-v4';
assert.equal(count,28);
assert.deepEqual([...data.excluded_ids].sort(),['p08_1_01','p08_1_03','p08_1_05','p08_1_07']);
assert.ok(data.items.every(i=>!i.exclude_p8_near&&!data.excluded_ids.includes(i.id)));
const n=['straight_front','right_front','right_lateral','right_back','straight_back','left_back','left_lateral','left_front'];
C.DIRS.forEach((d,i)=>{assert.equal(C.convert(d,'N'),n[i]);assert.equal(C.convert(d,'F'),n[(i+4)%8]);});
assert.equal(C.dir3('right_front'),'front');assert.equal(C.dir3('left_back'),'back');assert.equal(C.dir3('left_lateral'),'lateral');
let t={ms:0,last:null};C.touch(t,1000);assert.equal(C.elapsed(t,3000),2000);C.touch(t,4000);assert.equal(t.ms,3000);C.touch(t,64000);assert.equal(t.ms,3000);C.touch(t,65000);assert.equal(t.ms,4000);C.suspend(t,66000);assert.equal(t.ms,5000);C.touch(t,900000);assert.equal(t.ms,5000);
const p={motion:'motion',complexity:'single',direction3:'front',direction8:'straight_front',feint:'no',front_side:'N',front_slot:'near_front'};
const item={id:'dummy',point:1,contact_frame:100,front_side:'N',start_frame:40,stop_frame:160,predictions:{bbox:{...p},ankle:{...p}}};
function accepted(i) { const l={player_judgment:'正しい',contact_judgment:'正しい',confidence:'high'};for(const s of C.sources(i))for(const a of C.ASPECTS)l[`${s}_${a}_judgment`]='正しい';return l; }
let l=accepted(item);
assert.equal(C.valid(item,l),'');assert.equal(C.resolve(item,l).manual,false);
assert.equal(C.row(item,l,12500,'本人').motion,'motion');assert.equal(C.row(item,l,0,'').single_compound,'single');assert.equal(C.row(item,l,0,'').player_direction8,'straight_front');
const conflict=structuredClone(item);conflict.predictions.ankle.direction3='back';conflict.predictions.ankle.direction8='straight_back';
assert.equal(C.resolve(conflict,l).conflict,true);assert.match(C.valid(conflict,l),/正しい答え/);
const eight=structuredClone(item);eight.predictions.ankle.direction8='right_front';assert.equal(C.resolve(eight,l).conflict,true);
l={...accepted(item),bbox_motion_judgment:'違う'};assert.equal(C.resolve(item,l).manual,true);assert.match(C.valid(item,l),/正しい答え/);
l={...l,answer_class:'compound',screen_direction:'up_left',feint:'yes',contact_judgment:'違う'};assert.match(C.valid(item,l),/フレーム/);l.corrected_contact_frame=97;assert.equal(C.valid(item,l),'');
assert.equal(C.row(item,l,0,'').corrected_contact_frame,97);assert.equal(C.row(item,l,0,'').player_direction8,'left_front');
l.corrected_contact_frame=39;assert.match(C.valid(item,l),/フレーム/);
l={...accepted(item),manual:true,answer_class:'none',feint:'yes'};assert.match(C.valid(item,l),/複/);
const absent=structuredClone(item);for(const s of C.sources(absent))absent.predictions[s]={...p,motion:'none',complexity:'',direction8:'',direction3:''};
assert.equal(C.resolve(absent,accepted(absent)).motion,'none');assert.equal(C.resolve(absent,accepted(absent)).direction8,'');
const unknown=Object.fromEntries(Object.entries(accepted(item)).map(([k,v])=>[k,k.endsWith('_judgment')&&!['player_judgment','contact_judgment'].includes(k)?'分からない':v]));
assert.equal(C.valid(item,unknown),'');assert.equal(C.resolve(item,unknown).motion,'unknown');assert.equal(C.row(item,{},0,'').motion,'');
const rejected=structuredClone(item);
for(const s of C.sources(rejected))rejected.predictions[s]={...p,motion:'unclassifiable',complexity:'',direction8:'',direction3:'',feint:'unknown'};
assert.equal(C.valid(rejected,accepted(rejected)),'');
assert.equal(C.resolve(rejected,accepted(rejected)).motion,'unknown');
const rows=data.items.map((i,k)=>{
  const x=accepted(i);x.note='引用符"、カンマ, 改行\n次の行';x.saved_at='2026-11-18T00:00:00Z';
  // ダミーは全項目を確認し、手動回答を明示して比較側の分母と入出力を確認する。
  Object.assign(x,{manual:true,answer_class:['none','single','compound','unknown'][k%4],screen_direction:'down_left',feint:'unknown'});
  if(k%3===0)Object.assign(x,{contact_judgment:'違う',corrected_contact_frame:i.contact_frame-3});
  assert.equal(C.valid(i,x),'',i.id);return C.row(i,x,12500,'ダミー');
});
assert.equal(rows.length,count);assert.equal(Object.keys(rows[0]).length,C.FIELDS.length);
assert.equal(data.schema_version,3);assert.equal(data.order_seed,20261118);assert.equal(data.label_condition,'without_candidates');
assert.deepEqual(C.SOURCES,['bbox','ankle']);assert.ok(C.FIELDS.every(k=>!k.startsWith('claude_')));
for(const i of data.items){
  const fps=i.fps_num/i.fps_den;
  assert.ok(i.source_first_frame<=i.start_frame&&i.start_frame<=i.contact_frame&&i.contact_frame<=i.end_frame&&i.end_frame<=i.stop_frame&&i.stop_frame<i.source_first_frame+i.frame_count);
  assert.equal(i.clip,`../label_ui/clips/point_${String(i.point).padStart(2,'0')}.mp4`);
  assert.deepEqual(Object.keys(i.predictions),['bbox','ankle']);
  assert.ok(i.role_note.length>0);
  assert.ok((i.stop_frame-i.source_first_frame+.5)/fps<i.frame_count/fps);
  assert.equal(i.start_frame,Math.max(i.source_first_frame,Math.min(i.contact_frame-Math.round(fps),i.previous_contact_frame===null?i.contact_frame-Math.round(fps):i.previous_contact_frame-Math.round(.2*fps))));
  assert.equal(i.stop_frame,Math.min(i.source_first_frame+i.frame_count-1,Math.max(i.contact_frame+Math.round(fps),i.end_frame,i.next_contact_frame===null?i.contact_frame+Math.round(fps):i.next_contact_frame+Math.round(.2*fps))));
  assert.equal(typeof i.box_suspect,'boolean');
  assert.equal(i.box_suspect,i.box_suspect_check.outside_frames*2>=i.box_suspect_check.interval_frames);
  for(const points of Object.values(i.trajectories))assert.ok(points.every(p=>p[0]>=i.contact_frame&&p[0]<=i.end_frame));
}
assert.ok(fs.statSync(path.join(__dirname,'data.json')).size<256000);
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');const js=fs.readFileSync(path.join(__dirname,'app.js'),'utf8');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
for(const m of js.matchAll(/\$\("([^"]+)"\)/g))assert.ok(ids.includes(m[1]),'HTML id missing: '+m[1]);
assert.ok(js.includes(storageKey));assert.ok(!js.includes('softtennis-motion-human-20261118-v3'));assert.ok(!js.includes('softtennis-motion-confirm-1986-v2'));
assert.ok(!/claude|推定/.test(js+html));assert.ok(!js.includes('softtennis-motion-reference-1986-v1'));
assert.ok(js.includes('response.blob()'));assert.ok(js.includes('a.download = "motion_review.csv"'));
const csv=C.csv(rows);assert.ok(csv.startsWith('\ufeff'));assert.ok(csv.includes('引用符""、カンマ, 改行\n次の行'));
assert.equal(C.zoomRegion({boxes:[]}),null);
function checkZoom(i,boxes) {
  const region=C.zoomRegion(i), eps=1e-9;
  assert.ok(Number.isFinite(region.sx)&&Number.isFinite(region.sy)&&Number.isFinite(region.size));
  assert.ok(region.size+eps>=3*Math.max(...boxes.map(b=>b[4]-b[2])),'区間の最大高さの3倍以上: '+i.id);
  for(const [,x1,y1,x2,y2] of boxes) {
    assert.ok(region.sx<=x1+eps&&region.sy<=y1+eps&&region.sx+region.size+eps>=x2&&region.sy+region.size+eps>=y2,'区間のすべての箱を含む: '+i.id);
    for(const [x,y] of [[x1,y1],[x2,y2]])
      C.zoomPoint(region,x,y,640).forEach(v=>assert.ok(v>=-eps&&v<=640+eps));
  }
  return region;
}
// 打球前・分類後の遠い箱を無視し、両端を含む区間全体と最大高さを使う。
const span={id:'synthetic-span',contact_frame:100,end_frame:110,boxes:[
  [40,-2000,-2000,-1900,-1800],[100,100,200,140,300],
  [105,400,300,450,500],[110,900,800,940,900],[111,4000,4000,4100,4300]]};
const inSpan=span.boxes.slice(1,4), spanRegion=checkZoom(span,inSpan);
assert.deepEqual(spanRegion,C.zoomRegion({...span,boxes:inSpan}),'表示範囲の最初の箱に影響されない');
assert.ok(spanRegion.size>3*200,'広く移動する場合は高さ3倍より範囲を広げる');
const growing={id:'synthetic-max-height',contact_frame:100,end_frame:101,
  boxes:[[100,100,200,150,300],[101,100,200,180,360]]};
checkZoom(growing,growing.boxes);
// 区間内に箱がない場合、打球に最も近い箱で中心と高さ3倍を決める。
for(const [boxes,nearestIndex] of [
  [[[40,0,0,40,100],[99,200,300,250,500],[115,900,900,940,1000]],1],
  [[[40,0,0,40,100],[85,900,900,940,1000],[111,200,300,250,500]],2]
]) {
  const fallback={id:'fallback',contact_frame:100,end_frame:110,boxes};
  const nearest=boxes[nearestIndex];
  const region=checkZoom(fallback,[nearest]), [,x1,y1,x2,y2]=nearest;
  assert.equal(region.size,3*(y2-y1));
  assert.deepEqual(C.zoomPoint(region,(x1+x2)/2,(y1+y2)/2,640),[320,320]);
}
for(const i of data.items) {
  assert.ok(i.boxes.every((b,k)=>k===0||i.boxes[k-1][0]<b[0]),'箱はフレーム順');
  if(!i.boxes.length)continue;
  const interval=i.boxes.filter(b=>b[0]>=i.contact_frame&&b[0]<=i.end_frame);
  const use=interval.length?interval:[i.boxes.reduce((a,b)=>Math.abs(a[0]-i.contact_frame)<=Math.abs(b[0]-i.contact_frame)?a:b)];
  checkZoom(i,use);
}

// ブラウザを起動せず、実際のイベント配線・フレーム更新とcanvas呼出しを検証する。
async function checkUI(frameCallbacks, firstAction, fixture=data) {
  const vm=require('node:vm');
  let now=1000, interval, animation;
  const docListeners={};
  class Element {
    constructor(tag='div') { this.tagName=tag;this.listeners={};this.children=[];this.dataset={};this.style={};this.value='';this.checked=true;this.width=this.height=640; }
    append(...els) { this.children.push(...els); }
    replaceChildren(...els) { this.children=els; }
    setAttribute(k,v) { this[k]=v; }
    getAttribute(k) { return this[k]; }
    querySelectorAll() { return []; }
    addEventListener(k,fn) { (this.listeners[k]??=[]).push(fn); }
    async fire(k) { for(const fn of this.listeners[k]||[])await fn({target:this}); }
    getContext() {
      if(this.context)return this.context;
      const ctx={calls:[]};
      for(const name of ['clearRect','fillRect','drawImage','strokeRect','beginPath','moveTo','lineTo','stroke','arc','fill','fillText','setLineDash'])
        ctx[name]=(...args)=>ctx.calls.push({name,args,color:ctx.strokeStyle});
      return this.context=ctx;
    }
  }
  const els=Object.fromEntries(ids.map(id=>[id,new Element()]));
  const choices={};
  for(const m of html.matchAll(/data-(answer_class|feint|confidence)="([^"]+)"/g)) {
    const b=new Element('button');b.dataset[m[1]]=m[2];(choices[m[1]]??=[]).push(b);
  }
  const stepButton=new Element('button');stepButton.dataset.step='1';
  const video=els.video;
  Object.assign(video,{paused:true,seeking:false,readyState:2,duration:1e9,currentTime:0,
    pause() { this.paused=true; },
    async play() { this.paused=false; await this.fire('play'); },
    load() { this.onloadedmetadata(); }
  });
  if(frameCallbacks)video.requestVideoFrameCallback=fn=>{animation=fn;};
  els.speed.value='1';
  const oldKey='softtennis-motion-human-20261118-v3', oldValue='{"unfinished":"previous session"}';
  const storage=new Map([[oldKey,oldValue]]);
  const context=vm.createContext({
    document:{getElementById:id=>els[id],createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==='[data-step]'?[stepButton]:choices[selector.match(/^\[data-(\w+)\]$/)?.[1]]||[],addEventListener(k,fn){docListeners[k]=fn;}},
    window:{addEventListener(){}},
    localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
    fetch:async file=>({ok:true,json:async()=>structuredClone(fixture),blob:async()=>({})}),
    URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},
    Date:class extends Date { static now() { return now; } },
    requestAnimationFrame:fn=>{animation=fn;},setInterval:fn=>{interval=fn;},setTimeout(){}
  });
  vm.runInContext(js,context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(els.error.hidden,true,els.error.textContent);
  await video.fire('seeked');
  const waiting='作業時間 —（最初の操作から計時）';
  assert.equal(els.timing.textContent,waiting);
  now+=35000;interval();
  const saved=()=>JSON.parse(storage.get(storageKey));
  assert.equal(storage.get(oldKey),oldValue,'旧版の途中記録を読み込まず、上書きもしない');
  const first=fixture.items[0], zoom=els.zoom.getContext(), overlay=els.overlay.getContext();
  assert.equal(saved().labels[first.id].seconds,0,'開いて35秒待っても0秒');
  assert.equal(els.timing.textContent,waiting);
  assert.equal(els.roleReason.textContent,first.role_note);
  assert.ok(els.progress.textContent.endsWith(`${count}件`));
  assert.equal(els.predictions.children.length,2);assert.equal(els.judgments.children.length,2);
  assert.equal(els.zoomOrientation.textContent,`画面の上＝前衛から見て ${first.front_side==='F'?'後ろ':'前'}`);
  const crop=zoom.calls.find(c=>c.name==='drawImage').args.slice(1);
  const region=C.zoomRegion(first);
  assert.deepEqual(crop,[region.sx,region.sy,region.size,region.size,0,0,640,640]);
  if(firstAction==='play')await els.play.fire('click');
  if(firstAction==='step') { await stepButton.fire('click');await video.fire('seeked'); }
  if(firstAction!=='seek')assert.match(els.timing.textContent,/作業時間 0\.0秒/,firstAction+' だけでも計時開始');
  const frame=first.boxes[0][0];
  els.timeline.value=String(frame);await els.timeline.fire('input');await video.fire('seeked');
  assert.match(els.timing.textContent,/作業時間 0\.0秒/,'最初の操作で計時開始');
  const scale=640/region.size, box=first.boxes[0];
  const rect=zoom.calls.filter(c=>c.name==='strokeRect').at(-1).args;
  const expected=[(box[1]-region.sx)*scale,(box[2]-region.sy)*scale,(box[3]-box[1])*scale,(box[4]-box[2])*scale];
  rect.forEach((v,k)=>assert.ok(Math.abs(v-expected[k])<1e-9,'枠と動画の座標変換が一致'));
  const raw=first.trajectories.bbox[0];
  assert.ok(zoom.calls.some(c=>c.name==='moveTo'&&c.color==='#ffac39'&&Math.abs(c.args[0]-(raw[1]-region.sx)*scale)<1e-9&&Math.abs(c.args[1]-(raw[2]-region.sy)*scale)<1e-9),'軌跡も同じ倍率');
  now+=5000;interval();assert.equal(saved().labels[first.id].seconds,5);
  // 操作と自動のフレーム更新を区別。コマを変えても切り抜きは同じ。
  for(const [id,color] of [['showbbox','#ffac39'],['showankle','#4ee5ff'],['showBox','#67e6ff']]) {
    zoom.calls=[];overlay.calls=[];els[id].checked=false;await els[id].fire('change');
    for(const ctx of [zoom,overlay])assert.ok(!ctx.calls.some(c=>id==='showBox'?c.name==='strokeRect':c.name==='stroke'&&c.color===color),id+' は両画面に反映');
  }
  const nextFrame=Math.min(first.stop_frame,frame+1);
  video.currentTime=(nextFrame-first.source_first_frame+.5)/(first.fps_num/first.fps_den);
  zoom.calls=[];
  animation(0,{mediaTime:(nextFrame-first.source_first_frame)/(first.fps_num/first.fps_den)});
  assert.deepEqual(zoom.calls.find(c=>c.name==='drawImage').args.slice(1),crop,'フレーム更新後も中心・倍率が固定');
  now+=60000;interval();assert.equal(saved().labels[first.id].seconds,5,'60秒無操作は区間全体を除外');
  // 件の移動で次の件は未開始。戻った件の既存秒数も操作まで増えない。
  els.playerChoices.children[0].children[0].onclick();
  for(const fieldset of els.judgments.children)for(const line of fieldset.children.slice(1))line.children[1].children[0].onclick();
  els.contactChoices.children[0].children[0].onclick();
  els.manual.onchange({target:{checked:true}});
  choices.answer_class.find(b=>b.dataset.answer_class==='unknown').onclick();
  choices.feint.find(b=>b.dataset.feint==='unknown').onclick();
  choices.confidence.find(b=>b.dataset.confidence==='high').onclick();
  await els.next.fire('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(saved().index,1,els.validation.textContent);
  now+=35000;interval();assert.equal(els.timing.textContent,waiting);
  assert.equal(saved().labels[fixture.items[1].id].seconds,0,'次の件も最初の操作まで未開始');
  await els.prev.fire('click');
  await new Promise(resolve=>setImmediate(resolve));
  now+=35000;interval();
  assert.equal(saved().index,0);
  assert.equal(saved().labels[first.id].seconds,5,'再訪しても操作まで既存秒数は増えない');
  // 実際の T キー配線で本人ラベルの修正を保存。元フレームは保持する。
  els.contactChoices.children[0].children[1].onclick();
  els.timeline.value=String(first.contact_frame-2);
  await els.timeline.fire('input');await video.fire('seeked');
  docListeners.keydown({key:'t',target:{closest:()=>null},preventDefault(){}});
  assert.equal(saved().labels[first.id].corrected_contact_frame,first.contact_frame-2);
  assert.equal(fixture.items[0].contact_frame,first.contact_frame);
  els.contactChoices.children[0].children[0].onclick();
  // 全件を移動。警告の表示・解除、枠なし・軌跡なし・判定不能も確認。
  for(let k=0;k<fixture.items.length;k++) {
    const i=fixture.items[k];
    assert.equal(saved().index,k);
    assert.equal(els.roleReason.textContent,i.role_note);
    assert.equal(els.boxWarning.hidden,!i.box_suspect);
    assert.equal(els.boxWarning.textContent,i.box_suspect?'枠が選手でない可能性（コートの外）':'');
    assert.equal(els.predictions.children.length,2);
    assert.equal(els.judgments.children.length,2);
    if(!i.boxes.length)assert.match(els.zoomStatus.textContent,/拡大できません/);
    if(i.predictions.bbox.motion==='unclassifiable')
      assert.ok(els.predictions.children[0].children.some(e=>e.textContent?.includes('判定できない')));
    els.playerChoices.children[0].children[i.front_slot?0:1].onclick();
    els.contactChoices.children[0].children[0].onclick();
    for(const fieldset of els.judgments.children)for(const line of fieldset.children.slice(1))line.children[1].children[0].onclick();
    els.manual.onchange({target:{checked:true}});
    choices.answer_class.find(b=>b.dataset.answer_class==='unknown').onclick();
    choices.feint.find(b=>b.dataset.feint==='unknown').onclick();
    choices.confidence.find(b=>b.dataset.confidence==='high').onclick();
    await els.next.fire('click');await new Promise(resolve=>setImmediate(resolve));
    await video.fire('seeked');
    assert.equal(saved().labels[i.id].completed,true,els.validation.textContent);
  }
  assert.equal(saved().index,count-1);
  assert.equal(els.progress.textContent,`完了 ${count} / ${count}件 · 表示 ${count} / ${count}件`);
  assert.match(els.validation.textContent,/全件が完了/);
}

(async()=>{
  await checkUI(true,'play');await checkUI(true,'step');await checkUI(false,'seek');
  const warningData=structuredClone(data);warningData.items[0].box_suspect=true;
  await checkUI(true,'play',warningData);
  if(process.argv.includes('--csv'))process.stdout.write(csv);
  else console.log(`成功：本人打球の判定・CSV・全${count}件検証、警告表示・保存キー分離・拡大範囲と欠測時の代替・固定切抜き・表示連動・両フレームAPI・初操作まで計時なし・60秒境界`);
})().catch(e=>{console.error(e);process.exitCode=1;});
