"use strict";
(() => {
  const C = globalThis.LabelStudyCore;
  const $ = id => document.getElementById(id);
  const video = $("video");
  const STORAGE = "match2-human-label-study-v1";
  let db = {version:1, participant:"", sessions:[]}, data, clip, session;
  let shownLocal = 0, pendingSeek = null, referenceVisible = false;
  let playbackStarted = null, lastCheckpoint = performance.now(), seekTimer = null;
  let storageBlocked = false, foreignTab = false, ready = false;
  let channel = null;
  const tabId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  function message(text) { $("message").textContent = text; }
  function fail(text) { $("error").hidden = false; $("error").textContent = text; }
  function persist() {
    if (foreignTab) return;
    try {
      localStorage.setItem(STORAGE, JSON.stringify(db));
      storageBlocked = false;
      $("saveStatus").textContent = "このブラウザに自動保存済み。終了時は CSV も書き出してください。";
    } catch (e) {
      storageBlocked = true;
      $("saveStatus").textContent = `自動保存に失敗しました（${e.name}）。画面を閉じる前に CSV / JSON を保存してください。`;
    }
  }
  // python -m http.server does not answer Range requests, so a directly
  // referenced mp4 is not seekable. Play each clip from an in-memory Blob.
  const blobUrls = new Map();
  let clipToken = 0;
  async function setVideoSource(file) {
    const token = ++clipToken;
    let url = blobUrls.get(file);
    if (!url) {
      try {
        const response = await fetch(file);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        url = URL.createObjectURL(await response.blob());
        blobUrls.set(file, url);
      } catch (e) {
        url = file;
        message(`動画を Blob として読み込めませんでした（${e.message}）。コマ送りができない場合があります。`);
      }
    }
    if (token !== clipToken) return;
    video.src = url;
    video.load();
  }
  function load() {
    let raw = null;
    try {
      raw = localStorage.getItem(STORAGE);
    } catch (e) {
      storageBlocked = true;
      $("saveStatus").textContent = `保存領域を読めません（${e.name}）。作業後に CSV / JSON を保存してください。`;
    }
    try {
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed.version !== 1 || !Array.isArray(parsed.sessions)) throw new Error("保存形式が不明です");
        db = parsed;
      }
    } catch (e) {
      // Never overwrite an unreadable saved study automatically.
      fail(`保存済みデータを読み込めませんでした。自動上書きを停止しました。${e.message}`);
      foreignTab = true;
    }
    const now = Date.now();
    db.sessions.forEach(s => C.suspend(s, s.last_seen ?? now, "reload_or_close"));
    $("participant").value = db.participant;
  }
  function currentFrame() { return clip.source_first_frame + shownLocal; }
  function candidates() {
    return session.condition === "with_candidates" ? clip.candidates[session.candidate_version] : [];
  }
  function tallyPlayback() {
    if (playbackStarted != null && session && session.end == null) {
      const now = performance.now();
      session.playback_ms += Math.max(0,now-playbackStarted);
      playbackStarted = now;
    }
  }
  function stopPlayback() { tallyPlayback(); playbackStarted = null; }
  function pause() { stopPlayback(); video.pause(); }
  function act(type, detail={}) {
    if (!session || foreignTab || document.hidden) return false;
    if (session.start == null) session.reference_exposed_before_start = db.sessions.some(s => s.clip_id === clip.id && s.compared_at != null);
    const ok = C.record(session,type,Date.now(),detail);
    if (ok) { persist(); renderMetrics(); lockControls(); }
    return ok;
  }
  function editable() {
    if (!ready || !clip || foreignTab) return false;
    if (session.end != null) { message("完了済みです。編集する場合は新しい試行を作成してください。"); return false; }
    if (pendingSeek != null || video.seeking) { message("フレームの表示を待ってからラベルを付けてください。"); return false; }
    return true;
  }
  function choose(forceNew=false, saved=null) {
    pause();
    if (session) C.suspend(session,Date.now(),"switch_trial");
    ready = false; pendingSeek = null; clearTimeout(seekTimer);
    clip = data.clips.find(c => c.id === (saved?.clip_id ?? $("rallySelect").value));
    if (!clip) return;
    const participant = $("participant").value.trim();
    const condition = $("condition").value;
    const version = condition === "with_candidates" ? $("candidateVersion").value : "none";
    session = saved || (!forceNew && [...db.sessions].reverse().find(s => s.clip_id === clip.id &&
      s.participant === participant && s.condition === condition && s.candidate_version === version));
    if (!session) {
      session = C.makeSession(clip,participant,condition,version,Date.now(),
        globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`);
      session.reference_exposed_before_start = db.sessions.some(s => s.clip_id === clip.id && s.compared_at != null);
      db.sessions.push(session);
    }
    $("participant").value = session.participant;
    $("condition").value = session.condition;
    $("rallySelect").value = clip.id;
    if (session.candidate_version !== "none") $("candidateVersion").value = session.candidate_version;
    db.participant = session.participant;
    if (!document.hidden) C.resume(session,Date.now());
    referenceVisible = false; shownLocal = 0;
    $("compare").textContent = "Claude のラベルと比べる";
    setVideoSource(clip.file);
    $("timeline").setAttribute("aria-valuemin",clip.source_first_frame);
    $("timeline").setAttribute("aria-valuemax",clip.source_first_frame+clip.frame_count-1);
    $("intervalText").textContent = `ラリー F${clip.interval_start_frame}–${clip.interval_end_frame_exclusive-1} ／ クリップ F${clip.source_first_frame}–${clip.source_first_frame+clip.frame_count-1} ／ 60000/1001 fps`;
    message(session.end != null ? "完了済み：記録を固定しています。映像の確認・比較は計時に含みません。" : "最初の再生・移動・ラベル操作から計時します。まず映像をクリックするとキーを使えます。");
    persist(); renderAll();
  }
  function lockControls() {
    const available = ready && !foreignTab;
    $("rallySelect").disabled = foreignTab;
    $("condition").disabled = foreignTab;
    document.querySelectorAll("[data-step]").forEach(b => { b.disabled = !available; });
    document.querySelectorAll("[data-action]").forEach(b => { b.disabled = !available || session?.end != null; });
    $("play").disabled = !available;
    $("complete").disabled = !available || session?.end != null;
    $("compare").disabled = !available || session?.end == null;
    $("newTrial").disabled = !available || session?.end == null;
    $("participant").disabled = foreignTab || (session?.start != null && session?.end == null);
    $("candidateVersion").disabled = foreignTab || session?.condition !== "with_candidates" || session?.start != null;
  }
  function renderMetrics() {
    if (!session) return;
    const t = C.timing(session,Date.now());
    $("status").textContent = session.end != null ? "完了" : session.start == null ? "未開始" : "作業中";
    for (const [id,key] of [["elapsed","elapsed"],["active","active"],["idle","idle"]]) $(id).textContent = `${(t[key]/1000).toFixed(1)} 秒`;
    $("playback").textContent = `${(session.playback_ms/1000).toFixed(1)} 秒`;
    $("actionCount").textContent = session.actions.length;
    $("trialInfo").textContent = `試行 ${session.id} ／ ${session.participant || "（名前未入力）"} ／ ${session.condition === "with_candidates" ? "候補あり・"+session.candidate_version : "候補なし"}` + (session.reference_exposed_before_start ? " ／ このラリーは参照ラベル閲覧後の試行です" : "");
  }
  function renderPosition() {
    if (!clip) return;
    const f = currentFrame(), sec=f*clip.fps_den/clip.fps_num;
    $("frame").textContent = `元動画 F ${f}`;
    $("time").textContent = `元動画 ${sec.toFixed(3)} 秒 ／ クリップ ${(shownLocal*clip.fps_den/clip.fps_num).toFixed(3)} 秒`;
    $("cursor").style.left = `${shownLocal/Math.max(1,clip.frame_count-1)*100}%`;
    $("timeline").setAttribute("aria-valuenow",f);
    $("seekState").textContent = pendingSeek != null ? "移動中…" : ("requestVideoFrameCallback" in video ? "表示フレームに同期" : "currentTime から計算");
  }
  function renderMarkers() {
    if (!clip) return;
    const root=$("markers"); root.replaceChildren();
    const pct=f => (f-clip.source_first_frame)/Math.max(1,clip.frame_count-1)*100;
    const band=$("rallyBand"); band.style.left=`${pct(clip.interval_start_frame)}%`;
    band.style.width=`${pct(clip.interval_end_frame_exclusive-1)-pct(clip.interval_start_frame)}%`;
    function marker(frame,cls,symbol,title) {
      const el=document.createElement("span"); el.className=`marker ${cls}`;
      el.style.left=`${pct(frame)}%`; el.textContent=symbol; el.title=title; root.append(el);
    }
    candidates().forEach(c => marker(c.frame,"candidate","",`音の候補 F${c.frame}`));
    session.labels.forEach(l => marker(l.frame,`label ${l.side === "N" ? "near" : "far"}`, l.is_serve ? "★" : l.side === "N" ? "●" : "◆",`F${l.frame} ${l.side} ${l.outcome}`));
    if (referenceVisible) clip.claude_labels.forEach(l => marker(l.frame,`label reference ${l.side === "N" ? "near" : "far"}`,l.is_serve ? "☆" : "◇",`Claude F${l.frame} ${l.confidence}`));
    $("candidateLegend").hidden = session.condition !== "with_candidates";
  }
  function renderLabels() {
    const root=$("labelRows"); root.replaceChildren();
    session.labels.forEach(l => {
      const tr=document.createElement("tr");
      [l.frame,l.side === "N" ? "手前 N" : "奥 F",l.is_serve ? "★" : "",l.outcome].forEach(t => {
        const td=document.createElement("td"); td.textContent=t; tr.append(td);
      });
      const td=document.createElement("td"), b=document.createElement("button");
      b.type="button"; b.textContent="このFへ"; b.addEventListener("click",()=>seek(l.frame,"label_seek"));
      td.append(b); tr.append(td); root.append(tr);
    });
  }
  function renderSaved() {
    const root=$("savedTrials"); root.replaceChildren();
    db.sessions.filter(s => s.start != null).slice().reverse().forEach(s => {
      const row=document.createElement("div"); row.className="saved";
      row.textContent=`第${s.point}点 ${s.rally} ／ ${s.participant || "未入力"} ／ ${s.condition === "with_candidates" ? "候補あり・"+s.candidate_version : "候補なし"} ／ ${s.end == null ? "未完了" : "完了"} ／ ${C.iso(s.start)}`;
      const b=document.createElement("button"); b.type="button"; b.textContent="開く";
      b.addEventListener("click",()=>choose(false,s)); row.append(b); root.append(row);
    });
  }
  function renderAll() { renderPosition(); renderMarkers(); renderLabels(); renderMetrics(); renderSaved(); lockControls(); }
  function positionFromTime(time) { shownLocal=C.frameFromTime(time,clip); renderPosition(); }
  function acknowledgeSeek() {
    if (!ready || video.seeking) return;
    positionFromTime(video.currentTime); pendingSeek=null; renderPosition();
  }
  function seek(frame,type="timeline_seek") {
    if (!ready || foreignTab) return;
    pause();
    const local=Math.max(0,Math.min(clip.frame_count-1,Math.round(frame-clip.source_first_frame)));
    act(type,{frame:clip.source_first_frame+local});
    pendingSeek=local;
    // One-microsecond inset avoids selecting the preceding frame through rounding.
    video.currentTime=local*clip.fps_den/clip.fps_num+0.000001;
    clearTimeout(seekTimer);
    seekTimer=setTimeout(acknowledgeSeek,250);
    renderPosition();
  }
  function step(n) {
    seek(clip.source_first_frame+(pendingSeek ?? shownLocal)+n,Math.abs(n) === 6 ? "step_6" : "step_1");
  }
  function jump(direction) {
    if (!ready || session.condition !== "with_candidates") return;
    const f=clip.source_first_frame+(pendingSeek ?? shownLocal);
    const list=candidates().filter(c => direction < 0 ? c.frame < f : c.frame > f);
    const c=direction < 0 ? list[list.length-1] : list[0];
    if (c) seek(c.frame,direction < 0 ? "candidate_previous" : "candidate_next");
    else { pause(); act("candidate_boundary"); message("この方向に候補はありません。"); }
  }
  function togglePlay() {
    if (!ready || foreignTab) return;
    if (!video.paused) { act("pause"); pause(); }
    else {
      act("play");
      if (video.ended) seek(clip.source_first_frame,"replay_seek");
      video.play().catch(e => message(`再生できません：${e.message}`));
    }
  }
  function labelAction(kind) {
    if (clip && ready && !("requestVideoFrameCallback" in video)) positionFromTime(video.currentTime);
    if (!editable()) return;
    pause();
    const f=currentFrame(), changed=C.edit(session,kind,f);
    act(kind,{frame:f,changed});
    persist(); renderAll();
    message(changed ? (kind === "outcome" ? `最後の結果：${session.labels.at(-1).outcome}` : `F${f} の操作を記録しました。`) : "近くにラベルがありません。先に N / F で打球を付けてください。");
  }
  function complete() {
    if (!editable()) return;
    if (!session.labels.length) { message("打球ラベルを1件以上付けてから完了してください。"); return; }
    pause();
    act("complete",{frame:currentFrame()});
    session.end=session.actions.at(-1).at;
    persist(); renderAll();
    message("ラリー完了。記録を固定しました。CSV を書き出してください。比較・再生は計時に含みません。");
  }
  function download(name,text,mime) {
    const url=URL.createObjectURL(new Blob([text],{type:mime}));
    const a=document.createElement("a"); a.href=url; a.download=name; document.body.append(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),10000);
  }
  function exportCSV() {
    tallyPlayback(); persist();
    const rows=C.exportRows(db.sessions,Date.now()), stamp=new Date().toISOString().replace(/[:.]/g,"-");
    const common=["trial_id","participant","point","rally","condition","candidate_version","status"];
    download(`match2_labels_${stamp}.csv`,C.csv(rows.labels,["point","rally","frame","side","is_serve","outcome",...common.filter(k => !["point","rally"].includes(k))]),"text/csv;charset=utf-8");
    download(`match2_work_${stamp}.csv`,C.csv(rows.work,["point","rally","condition","start","end","active_seconds","idle_seconds","n_actions_by_type","participant","trial_id","candidate_version","status","elapsed_seconds","away_seconds","playback_seconds","compared_at","reference_exposed_before_start"]),"text/csv;charset=utf-8");
    message("ラベルと作業記録の2ファイルを書き出しました。複数ダウンロードの許可が出たら、このローカル画面を許可してください。");
  }
  async function init() {
    load();
    const response=await fetch("data.json",{cache:"no-store"});
    if (!response.ok) throw new Error(`data.json: HTTP ${response.status}`);
    data=await response.json();
    data.clips.forEach(c => {
      for (const list of Object.values(c.candidates)) list.sort((a,b)=>a.frame-b.frame);
      const o=document.createElement("option"); o.value=c.id; o.textContent=`第${c.point}点 ・ ラリー ${c.rally}`; $("rallySelect").append(o);
    });
    if (!data.v2_available) $("candidateVersion").querySelector('[value="v2"]').disabled=true;
    $("rallySelect").disabled=foreignTab; $("condition").disabled=foreignTab;
    $("export").disabled=false; $("backup").disabled=false;
    $("rallySelect").addEventListener("change",()=>{ choose(); video.focus(); });
    $("condition").addEventListener("change",()=>{ choose(); video.focus(); });
    $("candidateVersion").addEventListener("change",()=>{ choose(); video.focus(); });
    $("participant").addEventListener("change",()=>choose());
    $("play").addEventListener("click",togglePlay);
    $("complete").addEventListener("click",complete);
    document.querySelectorAll("[data-step]").forEach(b=>b.addEventListener("click",()=>step(Number(b.dataset.step))));
    document.querySelectorAll("[data-action]").forEach(b=>b.addEventListener("click",()=>labelAction(b.dataset.action)));
    $("volume").addEventListener("input",()=>{ video.volume=Number($("volume").value); act("volume"); });
    video.addEventListener("click",()=>{ video.focus(); togglePlay(); }); video.tabIndex=0;
    video.addEventListener("loadeddata",()=>{ ready=true; positionFromTime(video.currentTime); lockControls(); });
    video.addEventListener("error",()=>{ ready=false; lockControls(); fail(`動画を再生できません。生成済みクリップとローカルHTTPサーバーを確認してください。code=${video.error?.code}`); });
    video.addEventListener("playing",()=>{ if (!document.hidden && session.end == null) playbackStarted=performance.now(); });
    ["pause","waiting","seeking","ended"].forEach(type=>video.addEventListener(type,()=>{
      stopPlayback();
      if (clip && ready && !("requestVideoFrameCallback" in video)) positionFromTime(video.currentTime);
    }));
    video.addEventListener("seeked",()=>{
      if (!("requestVideoFrameCallback" in video)) acknowledgeSeek();
      if (!video.paused && !document.hidden && session.end == null) playbackStarted=performance.now();
    });
    if ("requestVideoFrameCallback" in video) {
      const callback=(now,meta)=>{
        if (clip && ready) {
          positionFromTime(meta.mediaTime);
          if (!video.seeking && (pendingSeek == null || shownLocal === pendingSeek)) pendingSeek=null;
          renderPosition();
        }
        video.requestVideoFrameCallback(callback);
      };
      video.requestVideoFrameCallback(callback);
    } else video.addEventListener("timeupdate",()=>{ if (clip && ready) positionFromTime(video.currentTime); });
    $("timeline").addEventListener("click",e=>{
      const rect=$("timeline").getBoundingClientRect();
      seek(clip.source_first_frame+Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*(clip.frame_count-1));
      $("timeline").focus();
    });
    document.addEventListener("keydown",e=>{
      if (e.ctrlKey || e.altKey || e.metaKey || e.isComposing || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName) || e.target.isContentEditable) return;
      const k=e.key.toLowerCase();
      if (![" ","arrowleft","arrowright","j","k","n","f","s","backspace","delete","e","enter"].includes(k)) return;
      e.preventDefault(); if (!ready || foreignTab) return;
      if (e.repeat && !["arrowleft","arrowright"].includes(k)) return;
      if (k === " ") togglePlay();
      else if (k === "arrowleft" || k === "arrowright") step((k === "arrowleft" ? -1 : 1)*(e.shiftKey ? 6 : 1));
      else if (k === "j" || k === "k") jump(k === "j" ? -1 : 1);
      else if (k === "enter") complete();
      else labelAction(({n:"near",f:"far",s:"serve",backspace:"delete",delete:"delete",e:"outcome"})[k]);
    });
    $("compare").addEventListener("click",()=>{
      if (session.end == null) return;
      referenceVisible=!referenceVisible;
      if (referenceVisible && session.compared_at == null) session.compared_at=Date.now();
      $("compare").textContent=referenceVisible ? "Claude のラベルを隠す" : "Claude のラベルと比べる";
      persist(); renderMarkers(); message("薄い印は Claude の比較用ラベルです。独立した正解として扱わないでください。");
    });
    $("newTrial").addEventListener("click",()=>{ if (session.end != null) choose(true); });
    $("export").addEventListener("click",exportCSV);
    $("backup").addEventListener("click",()=>{
      tallyPlayback(); persist(); download(`match2_backup_${Date.now()}.json`,JSON.stringify(db,null,2),"application/json;charset=utf-8");
    });
    document.addEventListener("visibilitychange",()=>{
      if (!session) return;
      if (document.hidden) { pause(); C.suspend(session,Date.now(),"hidden_tab"); }
      else C.resume(session,Date.now());
      persist();
    });
    window.addEventListener("pagehide",()=>{ pause(); if (session) C.suspend(session,Date.now(),"pagehide"); persist(); });
    window.addEventListener("pageshow",()=>{ if (session && !document.hidden) C.resume(session,Date.now()); });
    window.addEventListener("storage",e=>{
      if (e.key !== STORAGE || foreignTab) return;
      pause(); foreignTab=true; lockControls(); fail("別のタブで作業記録が変更されました。競合を避けるため、このタブの操作・保存を停止しました。作業するタブを1つにしてください。");
    });
    // A second tab must not overwrite the first tab's localStorage snapshot.
    if (!("locks" in navigator) && "BroadcastChannel" in globalThis) {
      channel=new BroadcastChannel(STORAGE);
      channel.onmessage=e=>{
        if (e.data.id === tabId) return;
        if (e.data.type === "hello") channel.postMessage({type:"occupied",id:tabId});
        else if (e.data.type === "occupied") { pause(); foreignTab=true; lockControls(); fail("同じ画面が別のタブで開いています。このタブを閉じ、先に開いたタブで作業してください。"); }
      };
      channel.postMessage({type:"hello",id:tabId});
      await new Promise(resolve=>setTimeout(resolve,200));
    }
    if (foreignTab) { lockControls(); return; }
    choose();
    setInterval(()=>{
      if (!session || foreignTab || document.hidden) return;
      tallyPlayback();
      if (session.start != null && session.end == null) {
        session.last_seen=Date.now();
        if (performance.now()-lastCheckpoint >= 1000) { persist(); lastCheckpoint=performance.now(); }
      }
      renderMetrics();
    },250);
    if (storageBlocked) message("保存が使えません。作業後、必ず CSV と JSON を保存してください。");
  }
  // Holding a Web Lock prevents two tabs from overwriting the same snapshot.
  async function boot() {
    if (navigator.locks?.request) {
      await navigator.locks.request(STORAGE,{ifAvailable:true},async lock=>{
        if (!lock) { fail("同じ画面が別のタブで開いています。先に開いたタブで作業してください。"); return; }
        await init();
        // The browser releases this lock when the document is destroyed.
        await new Promise(()=>{});
      });
    } else await init();
  }
  boot().catch(e=>fail(`起動できません：${e.message}\nREADME の手順で生成と HTTP サーバーの起動を確認してください。`));
})();
