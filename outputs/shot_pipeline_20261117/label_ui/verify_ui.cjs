/* node verify_ui.cjs — logic checks and a DOM stub, not browser visual QA. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
require("./study_core.js");
const C = globalThis.LabelStudyCore;
const data = JSON.parse(fs.readFileSync(path.join(__dirname,"data.json"),"utf8"));
const clip=data.clips[0];
const make=()=>C.makeSession(clip,"研究者","with_candidates","v1",0,"test");
let checks=0;
function test(name, fn) { fn(); checks++; console.log("PASS",name); }
test("frame calculation for all frames in six clips",()=>{
  for (const c of data.clips) for (let i=0;i<c.frame_count;i++) {
    assert.equal(C.frameFromTime(i*c.fps_den/c.fps_num,c),i);
    assert.equal(C.frameFromTime((i+.99)*c.fps_den/c.fps_num,c),i);
  }
});
test("first action, completion and exact 60-second idle boundary",()=>{
  const s=make(); assert.equal(C.timing(s,100000).elapsed,0);
  C.record(s,"play",1000); C.record(s,"near",11000); C.record(s,"complete",71000); s.end=71000;
  assert.deepEqual(C.timing(s,900000),{elapsed:70000,active:10000,idle:60000,away:0});
  assert.equal(C.record(s,"play",900000),false); assert.equal(C.counts(s).play,1);
});
test("59.999 seconds stays active; entire long gap becomes idle",()=>{
  const s=make(); C.record(s,"play",0);
  assert.equal(C.timing(s,59999).active,59999);
  assert.equal(C.timing(s,60000).idle,60000);
  C.record(s,"near",75000); assert.equal(C.timing(s,85000).active,10000);
});
test("away and long inactivity are unioned, never double counted",()=>{
  const s=make(); C.record(s,"play",0); C.suspend(s,10000,"switch"); C.resume(s,80000);
  C.record(s,"near",80000); C.record(s,"complete",90000); s.end=90000;
  assert.deepEqual(C.timing(s,90000),{elapsed:90000,active:10000,idle:80000,away:70000});
});
test("short away interval and interrupted reload are excluded",()=>{
  const s=make(); C.record(s,"play",0); C.suspend(s,10000,"hidden"); C.resume(s,15000);
  C.record(s,"near",20000); C.suspend(s,22000,"reload"); C.resume(s,42000);
  C.record(s,"complete",45000); s.end=45000;
  assert.deepEqual(C.timing(s,45000),{elapsed:45000,active:20000,idle:25000,away:25000});
});
test("move within ±10, side change, serve, final outcome and delete",()=>{
  const s=make(); C.edit(s,"near",100); C.edit(s,"far",110);
  assert.equal(s.labels.length,1); assert.equal(s.labels[0].side,"F");
  C.edit(s,"serve",110); assert.equal(s.labels[0].is_serve,true);
  C.edit(s,"near",121); C.edit(s,"outcome",110);
  assert.equal(s.labels[0].outcome,"continue"); assert.equal(s.labels[1].outcome,"winner");
  C.edit(s,"delete",121); assert.equal(s.labels.at(-1).outcome,"unknown");
  s.end=1; assert.equal(C.edit(s,"delete",110),false); assert.equal(s.labels.length,1);
});
test("condition records are independent and CSV carries trial identity",()=>{
  const a=make(), b=C.makeSession(clip,"研究者","without_candidates","v2",0,"other");
  C.record(a,"near",0); C.edit(a,"near",22866); C.record(a,"complete",3000); a.end=3000;
  assert.equal(b.labels.length,0); assert.equal(b.candidate_version,"none");
  a.playback_ms=2400; const rows=C.exportRows([a,b],4000);
  assert.equal(rows.labels.length,1); assert.equal(rows.work.length,1);
  assert.equal(rows.work[0].active_seconds,"3.000"); assert.equal(rows.work[0].playback_seconds,"2.400");
  assert.equal(rows.labels[0].trial_id,"test");
  const csv=C.csv([{name:'=SUM(1,2)',value:'日本語,"改行\n'}],["name","value"]);
  assert.ok(csv.startsWith("\ufeff")); assert.ok(csv.includes("'=SUM(1,2)")); assert.ok(csv.includes('""改行\n'));
});

// A deliberately small stub exercises the actual app event handlers without
// pretending to validate video decoding, rendered layout or browser downloads.
class Element {
  constructor(tag="div") { this.tagName=tag.toUpperCase(); this.value=""; this.children=[]; this.events={}; this.style={}; this.dataset={}; this.disabled=false; this.hidden=false; this.attributes={}; }
  addEventListener(type,fn) { (this.events[type]??=[]).push(fn); }
  fire(type,event={}) { for (const fn of this.events[type]??[]) fn({target:this,...event}); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  setAttribute(k,v) { this.attributes[k]=String(v); }
  querySelector() { return new Element("option"); }
  getBoundingClientRect() { return {left:0,width:1000}; }
  focus() {} remove() {} click() { this.fire("click"); }
}
async function appCheck({blockedStorage=false,rvfc=false}={}) {
  const ids=[...fs.readFileSync(path.join(__dirname,"index.html"),"utf8").matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  elements.condition.value="with_candidates"; elements.candidateVersion.value="v1"; elements.rallySelect.value=clip.id;
  const video=elements.video; video.paused=true; video.currentTime=0; video.seeking=false;
  video.load=()=>{}; video.pause=()=>{video.paused=true; video.fire("pause");};
  video.play=()=>{video.paused=false;video.fire("playing");return Promise.resolve();};
  if(rvfc) video.requestVideoFrameCallback=fn=>{video.callback=fn;};
  const doc=new Element(); doc.hidden=false; doc.body=new Element();
  doc.getElementById=id=>{assert.ok(elements[id],`missing id ${id}`);return elements[id];};
  doc.createElement=tag=>new Element(tag); doc.querySelectorAll=()=>[];
  const saved=new Map(), blobs=[], win=new Element(); let now=1000000;
  const context={document:doc,window:win,navigator:{},console,Blob,
    performance:{now:()=>now},Date:class extends Date { static now(){return now;} },
    localStorage:{getItem:k=>{if(blockedStorage)throw Error("blocked");return saved.get(k)??null;},setItem:(k,v)=>{if(blockedStorage)throw Error("blocked");saved.set(k,v);}},
    URL:{createObjectURL:b=>{blobs.push(b);return "blob:test";},revokeObjectURL:()=>{}},
    fetch:async()=>({ok:true,json:async()=>structuredClone(data)}),
    setTimeout:()=>0,clearTimeout:()=>{},setInterval:()=>0,crypto:{randomUUID:()=>`trial-${now}`}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,"study_core.js"),"utf8"),context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,"app.js"),"utf8"),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements.error.hidden,false); // stub default; content must remain empty
  assert.equal(elements.error.textContent,undefined);
  video.fire("loadeddata");
  const key=k=>doc.fire("keydown",{key:k,target:new Element("video"),preventDefault(){}});
  key("ArrowRight"); video.fire("seeked");
  if(rvfc) video.callback(0,{mediaTime:video.currentTime});
  key("n"); now+=5000; key("s"); key("e"); key("Enter");
  assert.equal(elements.status.textContent,"完了");
  assert.equal(elements.frame.textContent,`元動画 F ${clip.source_first_frame+1}`);
  const before=elements.active.textContent; now+=999000;
  elements.compare.click(); key("n"); assert.equal(elements.active.textContent,before);
  elements.export.click(); assert.equal(blobs.length,2);
  assert.ok((await blobs[0].text()).includes('"winner"'));
  assert.ok((await blobs[1].text()).includes('"5.000"'));
  elements.condition.value="without_candidates"; elements.condition.fire("change"); video.fire("loadeddata");
  assert.equal(elements.labelRows.children.length,0); assert.equal(elements.candidateLegend.hidden,true);
  assert.equal(elements.status.textContent,"未開始");
  key("k"); assert.equal(elements.status.textContent,"未開始");
  if(blockedStorage) assert.ok(elements.saveStatus.textContent.includes("失敗"));
  else {
    const state=JSON.parse([...saved.values()][0]);
    assert.equal(state.sessions[0].end,1005000); assert.equal(state.sessions[0].labels.length,1);
    assert.equal(state.sessions[1].labels.length,0);
  }
  checks++; console.log("PASS app handlers",{blockedStorage,rvfc});
}
(async()=>{
  await appCheck(); await appCheck({rvfc:true}); await appCheck({blockedStorage:true});
  console.log(`PASS ${checks} checks (not a browser playback/layout test)`);
})().catch(e=>{console.error(e);process.exitCode=1;});
