"use strict";
(() => {
  const DIRS = ["up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"];
  const ARROWS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"];
  const PLAYER = ["straight_front", "right_front", "right_lateral", "right_back", "straight_back", "left_back", "left_lateral", "left_front"];
  const JA = {straight_front:"まっすぐ前",right_front:"右斜め前",right_lateral:"右横",right_back:"右斜め後ろ",straight_back:"まっすぐ後ろ",left_back:"左斜め後ろ",left_lateral:"左横",left_front:"左斜め前"};
  const SOURCES = ["v1","v2","claude"], ASPECTS = ["motion","direction3","complexity"];
  const JUDGES = ["正しい","違う","分からない"];
  const FIELDS = ["id","point","contact_frame","corrected_contact_frame","contact_judgment","player_judgment",
    ...SOURCES.flatMap(s => ASPECTS.map(a => `${s}_${a}_judgment`)),
    "motion","single_compound","screen_direction","player_direction8","player_direction3","feint","answer_origin",
    "confidence","note","seconds","annotator","saved_at"];
  function convert(dir, side) {
    const i = DIRS.indexOf(dir);
    return i < 0 || !["N","F"].includes(side) ? "" : PLAYER[(i+(side==="F"?4:0))%8];
  }
  function dir3(d) { return d.endsWith("front") ? "front" : d.endsWith("back") ? "back" : d.endsWith("lateral") ? "lateral" : ""; }
  function elapsed(t,now) { const gap=t.last===null?0:Math.max(0,now-t.last); return t.ms+(gap<60000?gap:0); }
  function touch(t,now) { t.ms=elapsed(t,now); t.last=now; }
  function suspend(t,now) { t.ms=elapsed(t,now); t.last=null; }
  function openTimer(seconds=0) { return {ms:seconds*1000,last:null}; }
  function timingText(t,now) {
    return t.last===null && t.ms===0 ? "作業時間 —（最初の操作から計時）" :
      `作業時間 ${(elapsed(t,now)/1000).toFixed(1)}秒（60秒以上の無操作は除外）`;
  }
  function zoomRegion(item) {
    // 区間の最初の箱で中心・倍率を固定する。再生・シークで追従させない。
    const first=item.boxes[0];
    if (!first) return null;
    const [,x1,y1,x2,y2]=first, size=3*(y2-y1);
    return {sx:(x1+x2-size)/2,sy:(y1+y2-size)/2,size};
  }
  function zoomPoint(region,x,y,width) { return [(x-region.sx)*width/region.size,(y-region.sy)*width/region.size]; }
  function sources(item) { return SOURCES.filter(s => item.predictions[s]); }
  function applicable(p,a) { return a==="motion" || p.motion!=="none"; }
  function resolve(item,l) {
    const values={motion:[],complexity:[],direction3:[],direction8:[],feint:[]};
    let manual=!!l.manual, conflict=false;
    for (const s of sources(item)) {
      const p=item.predictions[s];
      for (const a of ASPECTS) {
        const j=l[`${s}_${a}_judgment`];
        if (j==="違う") manual=true;
        if (j==="正しい" && applicable(p,a) && p[a] && !["unknown","unclassifiable"].includes(p[a])) values[a].push(p[a]);
      }
      if (ASPECTS.filter(a=>applicable(p,a)).every(a=>l[`${s}_${a}_judgment`]==="正しい")) {
        if (p.direction8 && p.motion!=="none") values.direction8.push(p.direction8);
        if (p.feint && p.feint!=="unknown") values.feint.push(p.feint);
      }
    }
    for (const a of Object.keys(values)) if (new Set(values[a]).size>1) conflict=true;
    const pick=a=>values[a][0]||"";
    if ((pick("motion")==="none" && (pick("complexity") || pick("direction3"))) ||
        (pick("direction8") && pick("direction3") && dir3(pick("direction8"))!==pick("direction3"))) conflict=true;
    manual ||= conflict;
    if (manual) {
      const moving=["single","compound"].includes(l.answer_class);
      const d=moving?convert(l.screen_direction,item.front_side):"";
      return {manual,conflict,motion:l.answer_class==="none"?"none":moving?"motion":l.answer_class==="unknown"?"unknown":"",
        complexity:moving?l.answer_class:"",direction8:d,direction3:dir3(d),feint:l.feint||"",origin:"corrected"};
    }
    const motion=pick("motion")||(pick("complexity")?"motion":"unknown");
    return {manual,conflict,motion,complexity:motion==="motion"?pick("complexity"):"",
      direction8:motion==="motion"?pick("direction8"):"",direction3:motion==="motion"?pick("direction3"):"",
      feint:pick("feint")||"unknown",origin:"accepted"};
  }
  function valid(item,l) {
    if (!["正しい","違う"].includes(l.player_judgment)) return "前衛の選択を確認してください。";
    if (!JUDGES.includes(l.contact_judgment)) return "打球の瞬間を確認してください。";
    if (l.contact_judgment==="違う" && (!Number.isInteger(l.corrected_contact_frame) || l.corrected_contact_frame<item.start_frame || l.corrected_contact_frame>item.stop_frame)) return "正しい打球フレームを T で記録してください。";
    for (const s of sources(item)) for (const a of ASPECTS) {
      if (!JUDGES.includes(l[`${s}_${a}_judgment`])) return "各判定の3項目を確認してください（分からないも選べます）。";
    }
    const r=resolve(item,l);
    if (r.manual) {
      if (!["none","single","compound","unknown"].includes(l.answer_class)) return "正しい答えを入力してください。";
      if (r.motion==="motion" && !DIRS.includes(l.screen_direction)) return "正しい向きを選んでください。";
      if (!["yes","no","unknown"].includes(l.feint)) return "正しいフェイントの答えを選んでください。";
      if (l.feint==="yes" && l.answer_class!=="compound") return "フェイント「はい」は複で選んでください。";
    }
    if (!["high","medium","low"].includes(l.confidence)) return "自信を選んでください。";
    return "";
  }
  function row(item,l,ms,annotator) {
    const r=resolve(item,l), out={id:item.id,point:item.point,contact_frame:item.contact_frame,
      corrected_contact_frame:l.contact_judgment==="違う"?(l.corrected_contact_frame??""):"",
      contact_judgment:l.contact_judgment||"",player_judgment:l.player_judgment||""};
    for (const s of SOURCES) for (const a of ASPECTS) out[`${s}_${a}_judgment`]=item.predictions[s]?(l[`${s}_${a}_judgment`]||""):"";
    const started=sources(item).some(s=>ASPECTS.some(a=>l[`${s}_${a}_judgment`]))||l.manual;
    return {...out,motion:started?r.motion:"",single_compound:r.complexity,
      screen_direction:r.direction8?DIRS.find(d=>convert(d,item.front_side)===r.direction8):"",
      player_direction8:r.direction8,player_direction3:r.direction3,feint:started?r.feint:"",answer_origin:started?r.origin:"",
      confidence:l.confidence||"",note:l.note||"",seconds:(ms/1000).toFixed(3),annotator,saved_at:l.saved_at||""};
  }
  function csv(rows) {
    const quote=v=>'"'+String(v??"").replace(/"/g,'""')+'"';
    return "\ufeff"+[FIELDS,...rows.map(r=>FIELDS.map(k=>r[k]))].map(r=>r.map(quote).join(",")).join("\r\n")+"\r\n";
  }
  const Core={DIRS,ARROWS,PLAYER,JA,FIELDS,SOURCES,ASPECTS,convert,dir3,elapsed,touch,suspend,openTimer,timingText,zoomRegion,zoomPoint,sources,applicable,resolve,valid,row,csv};
  globalThis.MotionCore=Core;
  if (typeof document==="undefined") return;

  const $ = id => document.getElementById(id);
  const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  const zoomCanvas = $("zoom"), zoomCtx = zoomCanvas.getContext("2d");
  const STORAGE = "softtennis-motion-confirm-1986-v2";
  let data, index = 0, ready = false, blocked = false, storageWarning = "", db, selected = false;
  let shownFrame = 0, seekTarget = null, token = 0, blobUrl = null, blobFile = null;
  let timer = {ms:0,last:null}, frameCallbacks = "requestVideoFrameCallback" in video;
  let seekQueue = null;
  const current = () => data.items[index];
  const label = () => db.labels[current().id] || (db.labels[current().id] = {seconds:0});
  const fps = () => current().fps_num / current().fps_den;
  function error(message) { $("error").hidden = false; $("error").textContent = message; }
  function clearError() { $("error").hidden = true; }
  function persist() {
    if (!data || blocked) return;
    label().seconds = elapsed(timer, Date.now()) / 1000;
    label().saved_at = new Date().toISOString();
    db.index = index;
    try {
      localStorage.setItem(STORAGE, JSON.stringify(db));
      $("storage").textContent = "このブラウザに自動保存済み。終了前に CSV も書き出してください。";
    } catch (e) { $("storage").textContent = `自動保存に失敗（${e.name}）。閉じる前に CSV を書き出してください。`; }
  }
  function act() {
    if (!data || blocked) return;
    touch(timer, Date.now());
    renderTiming();
    label().saved_at = new Date().toISOString();
  }
  function renderTiming() { $("timing").textContent = timingText(timer,Date.now()); }
  function change(field, value) {
    act();
    const l = label();
    l[field] = value;
    l.completed = false;
    if (field === "contact_judgment" && value !== "違う") delete l.corrected_contact_frame;
    $("validation").textContent = "";
    persist(); renderForm(); draw(shownFrame);
  }
  function buttons(field, choices) {
    const wrap=document.createElement("div"); wrap.className="choices";
    for (const [v,text] of choices) {
      const b=document.createElement("button"); b.type="button"; b.textContent=text;
      b.setAttribute("aria-pressed",String(label()[field]===v)); b.onclick=()=>change(field,v); wrap.append(b);
    }
    return wrap;
  }
  const names={v1:"自動 v1（箱の下端）",v2:"自動 v2（足首）",v2_bbox:"v2（箱・参考）",claude:"Claude の目視"};
  const words={none:"なし",motion:"あり",single:"単",compound:"複",unknown:"判定できない",unclassifiable:"判定できない",front:"前",lateral:"横",back:"後ろ",yes:"はい",no:"いいえ"};
  function renderPredictions() {
    $("predictions").replaceChildren();
    for (const s of ["v1","v2","v2_bbox","claude"]) {
      const p=current().predictions[s]; if (!p) continue;
      const card=document.createElement("article"); card.className="prediction "+s;
      const h=document.createElement("h3"); h.textContent=names[s]; card.append(h);
      const same=p.front_slot===current().front_slot && p.front_side===current().front_side && !!p.front_slot;
      const target=document.createElement("p"); target.textContent=`対象：${p.front_side} / ${p.front_slot||"未確定"}${same?"":" ⚠ 枠の対象と異なる、または未確定"}`; card.append(target);
      const info=document.createElement("p");
      const screen=p.direction8?DIRS.find(d=>convert(d,p.front_side)===p.direction8):"";
      const threeArrow=p.direction3==="lateral"?"↔":p.direction3==="front"?(p.front_side==="N"?"↑":"↓"):p.direction3==="back"?(p.front_side==="N"?"↓":"↑"):"";
      info.textContent=`動き：${words[p.motion]||p.motion||"未記入"} ／ 単・複：${words[p.complexity]||"—"} ／ 向き：${screen?ARROWS[DIRS.indexOf(screen)]+" ":""}${JA[p.direction8]||"—"}（3方向：${threeArrow} ${words[p.direction3]||"—"}） ／ フェイント：${words[p.feint]||"分からない"}`;
      card.append(info);
      if (s==="claude") {
        for (const k of ["claude_motion","claude_direction3","claude_single_compound","claude_issue","claude_note"]) {
          const line=document.createElement("p"); line.textContent=`${k}：${p[k]||"未記入"}`; card.append(line);
        }
      } else if (p.issue) { const line=document.createElement("p"); line.textContent="判定不能の理由："+p.issue; card.append(line); }
      $("predictions").append(card);
    }
  }
  function renderForm() {
    const l=label(), item=current(), r=resolve(item,l);
    $("playerChoices").replaceChildren(buttons("player_judgment",[["正しい","正しい"],["違う","違う"]]));
    $("contactChoices").replaceChildren(buttons("contact_judgment",[["正しい","正しい"],["違う","ずれている"],["分からない","分からない"]]));
    $("setContact").hidden=l.contact_judgment!=="違う";
    $("setContact").disabled=!ready;
    $("correctedInfo").textContent=l.contact_judgment==="違う" && Number.isInteger(l.corrected_contact_frame)?`修正 F${l.corrected_contact_frame}（元から ${l.corrected_contact_frame-item.contact_frame>=0?"+":""}${l.corrected_contact_frame-item.contact_frame} コマ）`:"";
    $("judgments").replaceChildren();
    for (const s of sources(item)) {
      const field=document.createElement("fieldset"), legend=document.createElement("legend"); legend.textContent=names[s]; field.append(legend);
      for (const [a,text] of [["motion","動きの有無"],["direction3","向き（3方向）"],["complexity","単／複"]]) {
        const line=document.createElement("div"); line.className="judge-row";
        const title=document.createElement("span"); title.textContent=text; line.append(title);
        line.append(buttons(`${s}_${a}_judgment`,JUDGES.map(x=>[x,x])));
        field.append(line);
      }
      $("judgments").append(field);
    }
    $("manual").checked=!!l.manual;
    $("answerWrap").hidden=!r.manual;
    $("answerReason").textContent=r.conflict?"正しいとした内容が食い違っています。正しい答えを入力してください。":"違う項目がある場合は、正しい答えを入力してください。";
    $("answerSummary").textContent=r.manual?"本人の答え：修正入力を使用":`本人の答え：${words[r.motion]||"未確認"} ／ ${words[r.complexity]||"—"} ／ ${JA[r.direction8]||words[r.direction3]||"—"}（正しいと確認した内容を採用。未確定項目は空欄／不明）`;
    $("directionWrap").hidden=!["single","compound"].includes(l.answer_class);
    for (const field of ["answer_class","feint","confidence"]) document.querySelectorAll(`[data-${field}]`).forEach(b=>b.setAttribute("aria-pressed",String(b.dataset[field]===l[field])));
    $("finalDirections").querySelectorAll("button").forEach(b=>{
      b.setAttribute("aria-pressed",String(b.dataset.dir===l.screen_direction));
      b.querySelector("small").textContent=JA[convert(b.dataset.dir,item.front_side)];
    });
    const far=item.front_side==="F";
    $("courtDot").setAttribute("cy",far?"48":"132");
    $("courtArrow").setAttribute("d",far?"M60 48 V82 l-5 -8 m5 8 l5 -8":"M60 132 V98 l-5 8 m5 -8 l5 8");
    $("courtSide").textContent=far?"奥の前衛":"手前の前衛";
    if ($("note").value!==(l.note||"")) $("note").value=l.note||"";
    const n=data.items.filter(i=>db.labels[i.id]?.completed && !valid(i,db.labels[i.id])).length;
    $("progress").textContent=`完了 ${n} / 30件 · 表示 ${index+1} / 30件`;
    $("prev").disabled=index===0; $("next").textContent=index===29?"完了する":"次へ";
  }
  function effectiveContact() { return label().contact_judgment==="違う" && Number.isInteger(label().corrected_contact_frame)?label().corrected_contact_frame:current().contact_frame; }
  function setContact() {
    if (!ready || label().contact_judgment!=="違う") return;
    pause(); change("corrected_contact_frame",shownFrame);
  }
  let markerKey="";
  function markers() {
    const i=current(), c=effectiveContact(), key=i.id+":"+c;
    if (key===markerKey) return; markerKey=key;
    const wrap=$("markers"); wrap.replaceChildren();
    const band=document.createElement("span"); band.className="window-band";
    const left=Math.max(i.start_frame,Math.min(i.stop_frame,c)), right=Math.max(left,Math.min(i.stop_frame,c+Math.round(.4*fps())));
    band.style.left=`${100*(left-i.start_frame)/(i.stop_frame-i.start_frame)}%`;
    band.style.width=`${100*(right-left)/(i.stop_frame-i.start_frame)}%`;
    wrap.append(band);
    const specs=[[i.previous_contact_frame,"前の打球（前衛側）","previous"],[i.contact_frame,`相手（ラベル${i.contact_confidence==="inferred"?"・推定":""}）`,"contact"],[i.next_contact_frame,"次の打球","following"],[c,"分類区間の始点","window-start"],[c+Math.round(.4*fps()),"＋0.4秒","endpoint"]];
    if (c!==i.contact_frame) specs.push([c,"修正した打球","corrected"]);
    for (const [frame,text,cls] of specs) {
      if (frame===null) continue;
      const b=document.createElement("button"); b.type="button"; b.className="marker "+cls;
      b.style.left=`${100*(Math.max(i.start_frame,Math.min(i.stop_frame,frame))-i.start_frame)/(i.stop_frame-i.start_frame)}%`;
      b.textContent=text; b.title=`${text} F${frame}`;
      b.onclick=()=>{act();pause();seek(frame);persist();}; wrap.append(b);
    }
    $("windowInfo").textContent=`分類の目安：F${c}〜F${c+Math.round(.4*fps())}（0.4秒）。元の軌跡：F${i.contact_frame}〜F${i.end_frame}。打球修正後も分類・軌跡は再計算しません。`;
  }
  function arrow(ctx,x1,y1,x2,y2,color) {
    const angle=Math.atan2(y2-y1,x2-x1);
    ctx.strokeStyle=color;ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
    ctx.moveTo(x2-14*Math.cos(angle-.5),y2-14*Math.sin(angle-.5));ctx.lineTo(x2,y2);ctx.lineTo(x2-14*Math.cos(angle+.5),y2-14*Math.sin(angle+.5));ctx.stroke();
  }
  function trajectories(ctx, project=(x,y)=>[x,y], includeDirections=true) {
    for (const [s,color] of [["v1","#ffac39"],["v2","#4ee5ff"]]) {
      if (!$("show"+s).checked) continue;
      const points=current().trajectories[s];
      ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
      points.forEach((p,n)=>{const prev=points[n-1], xy=project(p[1],p[2]); if (prev && p[0]===prev[0]+1) ctx.lineTo(...xy); else ctx.moveTo(...xy);});ctx.stroke();
      if (points.length) {const a=project(points[0][1],points[0][2]),last=points[points.length-1],b=project(last[1],last[2]);ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],6,0,Math.PI*2);ctx.fill(); if(points.length>1)arrow(ctx,a[0],a[1],b[0],b[1],color);ctx.font="22px sans-serif";ctx.fillText(s+" 生座標",a[0]+12,a[1]-10);
        const p=current().predictions[s], d=DIRS.findIndex(d=>convert(d,p.front_side)===p.direction8);
        if(includeDirections && d>=0 && p.motion==="motion") {
          const theta=(d*45-90)*Math.PI/180;
          ctx.setLineDash([7,5]);arrow(ctx,a[0],a[1],a[0]+85*Math.cos(theta),a[1]+85*Math.sin(theta),color);ctx.setLineDash([]);
          ctx.fillText(s+" 最終方向",a[0]+12,a[1]+26);
        }
      }
    }
    if (includeDirections && $("showReference").checked) for (const [s,color,base] of [["v2_bbox","#c8c8c8","v2"],["claude","#d5a1ff","v1"]]) {
      const p=current().predictions[s], a=current().trajectories[base][0];
      if (!p || !a || p.motion!=="motion") continue;
      const exact=DIRS.findIndex(d=>convert(d,p.front_side)===p.direction8);
      const dirs=exact>=0?[exact]:p.direction3==="front"?[p.front_side==="N"?0:4]:p.direction3==="back"?[p.front_side==="N"?4:0]:p.direction3==="lateral"?[2,6]:[];
      ctx.setLineDash([4,7]);
      for(const d of dirs) {const theta=(d*45-90)*Math.PI/180;arrow(ctx,a[1],a[2],a[1]+110*Math.cos(theta),a[2]+110*Math.sin(theta),color);}
      ctx.setLineDash([]);if(dirs.length){ctx.fillStyle=color;ctx.font="20px sans-serif";ctx.fillText(s==="claude"?"Claude 3方向":"v2箱 最終方向",a[1]+12,a[2]+(s==="claude"?52:76));}
    }
  }
  function drawBox(ctx,box,project=(x,y)=>[x,y]) {
    if (!$("showBox").checked || !box) return;
    const [x1,y1]=project(box[0],box[1]), [x2,y2]=project(box[2],box[3]);
    ctx.strokeStyle="#06121b"; ctx.lineWidth=4; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.strokeStyle="#67e6ff"; ctx.lineWidth=2; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  }
  function drawZoom(box) {
    const region=zoomRegion(current()), W=zoomCanvas.width, H=zoomCanvas.height;
    zoomCtx.fillStyle="#000"; zoomCtx.fillRect(0,0,W,H);
    $("zoomStatus").textContent=!region ? "この区間には追跡枠がないため拡大できません。" :
      !ready || video.readyState<2 ? "動画を読み込み中…" : "";
    if (!region || !ready || video.readyState<2) return;
    // 画像端を越えた部分は黒地。切り詰めによる中心・倍率の変更はしない。
    zoomCtx.drawImage(video,region.sx,region.sy,region.size,region.size,0,0,W,H);
    const project=(x,y)=>zoomPoint(region,x,y,W);
    trajectories(zoomCtx,project,false);
    drawBox(zoomCtx,box,project);
  }
  function lockVideo() {
    $("play").disabled = !ready;
    $("timeline").disabled = !ready;
    document.querySelectorAll("[data-step]").forEach(b => { b.disabled = !ready; });
  }
  function draw(frame) {
    if (!data) return;
    const item = current();
    frame = Math.max(item.start_frame, Math.min(item.stop_frame, frame));
    shownFrame = frame;
    $("timeline").value = frame;
    const relative = (frame - effectiveContact()) / fps();
    $("frameInfo").textContent = `元動画 F${frame} · 打球から ${relative >= 0 ? "+" : ""}${relative.toFixed(3)}秒 · 基準 F${effectiveContact()}（元ラベル F${item.contact_frame}）`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    trajectories(ctx); markers();
    const box = item.boxes.find(b=>b[0]===frame)?.slice(1);
    drawBox(ctx,box);
    drawZoom(box);
    $("videoStatus").textContent = !ready ? "動画を読み込み中…" : !item.front_slot ? "対象slotが未確定です。追跡枠は表示できません。判定できない場合は「分からない」を選び、メモを記録してください。" : $("showBox").checked && !box ? "このフレームには対象slotの追跡枠がありません（補間しません）。" : "";
  }
  function seek(frame) {
    if (!ready) return;
    const item = current();
    frame = Math.max(item.start_frame, Math.min(item.stop_frame, Math.round(frame)));
    if (video.seeking) { seekQueue = frame; return; }
    seekTarget = frame;
    // フレーム先頭の浮動小数誤差を避け、フレームの中心へ移動する。
    video.currentTime = (frame - item.source_first_frame + .5) / fps();
  }
  function pause() { video.pause(); $("play").textContent = "再生"; }
  async function togglePlay() {
    if (!ready) return;
    act(); persist();
    if (!video.paused) { pause(); return; }
    if (shownFrame >= current().stop_frame) seek(current().start_frame);
    try { await video.play(); } catch (e) { error(`再生できません：${e.message}`); }
  }
  function step(delta) { if (!ready) return; act(); pause(); seek((seekQueue ?? seekTarget ?? shownFrame) + delta); persist(); }
  function onFrame(frame) {
    if (!ready || video.seeking || seekTarget !== null) return;
    const item = current();
    if (!video.paused && frame >= item.stop_frame) {
      if ($("loop").checked) seek(item.start_frame);
      else { pause(); seek(item.stop_frame); }
      return;
    }
    draw(frame);
  }
  if (frameCallbacks) {
    const tick = (_, meta) => {
      if (data) onFrame(current().source_first_frame + Math.round(meta.mediaTime * fps()));
      video.requestVideoFrameCallback(tick);
    };
    video.requestVideoFrameCallback(tick);
  } else {
    const tick = () => { if (data) onFrame(current().source_first_frame + Math.floor(video.currentTime * fps() + 1e-5)); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  video.addEventListener("seeked", () => {
    if (!ready) return;
    const actual = current().source_first_frame + Math.floor(video.currentTime * fps() + 1e-5);
    draw(seekTarget ?? actual); seekTarget = null;
    if (seekQueue !== null) { const queued = seekQueue; seekQueue = null; seek(queued); }
  });
  video.addEventListener("play", () => { $("play").textContent = "一時停止"; });
  video.addEventListener("pause", () => { $("play").textContent = "再生"; });
  video.addEventListener("error", () => { if (data && video.getAttribute("src")) error("動画を読み込めません。サーバーの起点・動画ファイルを確認してください。"); });
  async function setVideoSource(file) {
    const request = ++token;
    if (blobFile === file && video.readyState >= 1) { ready = true; lockVideo(); renderForm(); seek(current().start_frame); return; }
    try {
      const response = await fetch(file);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      if (request !== token) return;
      const previous = blobUrl;
      blobUrl = URL.createObjectURL(blob); blobFile = file;
      video.src = blobUrl;
      if (previous) URL.revokeObjectURL(previous);
      video.onloadedmetadata = () => {
        if (request !== token) return;
        const item = current();
        if ((item.stop_frame - item.source_first_frame + .5) / fps() >= video.duration) {
          error("表示範囲が動画の長さを超えています。data.json を再生成してください。"); return;
        }
        ready = true; video.playbackRate = Number($("speed").value); lockVideo(); renderForm(); seek(item.start_frame);
      };
      video.load();
    } catch (e) { if (request === token) error(`動画の取得に失敗：${e.message}。HTTPで開き、サーバーの起点を確認してください。`); }
  }
  function choose(nextIndex) {
    if (selected) { suspend(timer, Date.now()); persist(); }
    pause(); ready = false; seekTarget = seekQueue = null;
    index = nextIndex;
    selected = true;
    const item = current();
    timer = openTimer(label().seconds || 0);
    renderTiming();
    $("zoomOrientation").textContent = `画面の上＝前衛から見て ${item.front_side==="F"?"後ろ":"前"}`;
    $("itemTitle").textContent = `${index + 1}. 第${item.point}点 / ラリー${item.rally} / ${item.id}`;
    $("side").textContent = item.front_side === "N" ? "手前の前衛 N" : "奥の前衛 F";
    $("target").textContent = `追跡slot：${item.front_slot || "未確定"} · 表示範囲：F${item.start_frame}〜F${item.stop_frame}`;
    canvas.width = item.box_width; canvas.height = item.box_height;
    $("timeline").min = item.start_frame; $("timeline").max = item.stop_frame;
    renderPredictions();
    $("validation").textContent = "";
    draw(item.start_frame); renderForm(); lockVideo(); persist();
    setVideoSource(item.clip);
  }
  function next() {
    act();
    const l = label(), message = valid(current(),l);
    if (message) { $("validation").textContent = message; persist(); return; }
    suspend(timer, Date.now()); l.completed = true; l.saved_at = new Date().toISOString(); persist();
    if (index < data.items.length - 1) choose(index + 1);
    else {
      pause(); renderForm();
      const count = data.items.filter(i => db.labels[i.id]?.completed).length;
      $("validation").textContent = count === data.items.length ? "全件が完了しました。CSV を書き出してください。" : `この件を完了しました。未完了が${data.items.length-count}件あります。前へ戻って確認してください。`;
    }
  }
  function exportCsv() {
    if (!data) return;
    persist();
    const rows = data.items.map(item => row(item, db.labels[item.id] || {}, item.id === current().id ? elapsed(timer,Date.now()) : (db.labels[item.id]?.seconds || 0)*1000, db.annotator));
    const url = URL.createObjectURL(new Blob([csv(rows)],{type:"text/csv;charset=utf-8"}));
    const a = document.createElement("a"); a.href = url; a.download = "motion_review.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const positions=[[1,2],[1,3],[2,3],[3,3],[3,2],[3,1],[2,1],[1,1]];
  DIRS.forEach((dir,i)=>{
    const b=document.createElement("button");b.type="button";b.dataset.dir=dir;
    const symbol=document.createElement("span"), small=document.createElement("small");symbol.textContent=ARROWS[i];b.append(symbol,small);
    b.style.gridRow=positions[i][0];b.style.gridColumn=positions[i][1];
    b.onclick=()=>change("screen_direction",dir);$("finalDirections").append(b);
  });
  for (const field of ["answer_class","feint","confidence"]) document.querySelectorAll(`[data-${field}]`).forEach(b=>b.onclick=()=>change(field,b.dataset[field]));
  $("manual").onchange=e=>change("manual",e.target.checked);
  $("setContact").onclick=setContact;
  $("note").addEventListener("input", e => change("note",e.target.value));
  $("annotator").addEventListener("input", e => { act(); db.annotator = e.target.value; persist(); });
  $("play").addEventListener("click",togglePlay); video.addEventListener("click",togglePlay);
  document.querySelectorAll("[data-step]").forEach(b => b.addEventListener("click", () => step(Number(b.dataset.step))));
  $("timeline").addEventListener("input", e => { act(); pause(); seek(Number(e.target.value)); persist(); });
  $("speed").addEventListener("change", e => { act(); video.playbackRate = Number(e.target.value); persist(); });
  for (const id of ["loop","showBox","showv1","showv2","showReference"]) $(id).addEventListener("change", () => { act(); draw(shownFrame); persist(); });
  $("next").addEventListener("click",next);
  $("prev").addEventListener("click", () => { if (index > 0) { act(); choose(index-1); } });
  $("export").addEventListener("click",() => { act(); exportCsv(); });
  $("labelForm").addEventListener("submit", e => e.preventDefault());
  document.addEventListener("keydown", e => {
    if (!data || blocked || e.altKey || e.ctrlKey || e.metaKey || e.isComposing || e.repeat) return;
    if (e.target.closest("input,textarea,select,summary,[contenteditable=true]")) return;
    if ([" ","ArrowLeft","ArrowRight","t","T","Enter"].includes(e.key)) e.preventDefault();
    if (e.key === " ") togglePlay();
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") step((e.key === "ArrowLeft" ? -1 : 1)*(e.shiftKey?6:1));
    else if (e.key.toLowerCase() === "t") setContact();
    else if (e.key === "Enter") next();
  });
  document.addEventListener("visibilitychange", () => { if (data && document.hidden) { pause(); suspend(timer,Date.now()); persist(); } });
  window.addEventListener("pagehide", () => { if (data) { suspend(timer,Date.now()); persist(); } });
  window.addEventListener("storage", e => {
    if (e.key === STORAGE && data) {
      pause(); suspend(timer,Date.now()); blocked = true;
      document.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = true; });
      $("export").disabled = false;
      error("別タブで保存データが更新されました。このタブの上書きを停止しました。必要ならCSVを退避し、1つのタブだけで開き直してください。");
    }
  });
  async function init() {
    // 読み込み失敗時にも未初期化データを操作させない。
    document.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = true; });
    try {
      const response = await fetch("data.json", {cache:"no-store"});
      if (!response.ok) throw new Error(`data.json: HTTP ${response.status}`);
      data = await response.json();
      if (data.items?.length !== 30 || new Set(data.items.map(i => i.id)).size !== 30) throw new Error("data.json は30件の一意な打球である必要があります");
      db = {version:2,dataset_id:data.dataset_id,annotator:"",index:0,labels:{}};
      let raw;
      try { raw = localStorage.getItem(STORAGE); } catch (e) { storageWarning = `保存領域を読めません（${e.name}）。CSVで保存してください。`; }
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.version !== 2 || saved.dataset_id !== data.dataset_id || !saved.labels || typeof saved.labels !== "object" || typeof saved.annotator !== "string") throw new Error("保存形式・対象データが一致しません。既存の保存を上書きせず停止しました");
        db = saved;
      }
      $("annotator").value = db.annotator;
      document.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = false; });
      clearError(); choose(Math.max(0,Math.min(data.items.length-1,Number(db.index)||0)));
      if (storageWarning) $("storage").textContent = storageWarning;
      setInterval(() => { if (!blocked) { renderTiming(); persist(); } }, 5000);
    } catch (e) { blocked = true; error(`開始できません：${e.message}。README の起動方法を確認してください。`); }
  }
  init();
})();
