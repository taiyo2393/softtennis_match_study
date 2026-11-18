"use strict";
(() => {
  const DIRS = ["up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"];
  const ARROWS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"];
  const PLAYER = ["straight_front", "right_front", "right_lateral", "right_back", "straight_back", "left_back", "left_lateral", "left_front"];
  const JA = {straight_front:"まっすぐ前",right_front:"右斜め前",right_lateral:"右横",right_back:"右斜め後ろ",straight_back:"まっすぐ後ろ",left_back:"左斜め後ろ",left_lateral:"左横",left_front:"左斜め前"};
  const FIELDS = ["id","point","contact_frame","front_side","motion","unknown_reason","screen_dir_final","screen_dir_prelim","player_dir8_final","player_dir3_final","player_dir8_prelim","feint","confidence","note","seconds","annotator","saved_at"];
  function convert(dir, side) {
    const index = DIRS.indexOf(dir);
    if (index < 0 || !["N", "F"].includes(side)) return "";
    return PLAYER[(index + (side === "F" ? 4 : 0)) % 8];
  }
  function dir3(dir8) {
    return dir8.endsWith("front") ? "front" : dir8.endsWith("back") ? "back" : dir8.endsWith("lateral") ? "lateral" : "";
  }
  // 60秒以上の無操作区間は、最初の60秒も含め区間全体を除外する。
  function elapsed(timer, now) {
    const gap = timer.last === null ? 0 : Math.max(0, now - timer.last);
    return timer.ms + (gap < 60000 ? gap : 0);
  }
  function touch(timer, now) { timer.ms = elapsed(timer, now); timer.last = now; }
  function suspend(timer, now) { timer.ms = elapsed(timer, now); timer.last = null; }
  function valid(label) {
    if (!["none","single","compound","unknown"].includes(label.motion)) return "モーションを選んでください。";
    if (label.motion === "unknown" && !["not_visible","wrong_player","other"].includes(label.unknown_reason)) return "判定できない理由を選んでください。";
    if (["single","compound"].includes(label.motion) && !DIRS.includes(label.screen_dir_final)) return "決定の向きを選んでください。";
    if (label.motion === "compound" && !DIRS.includes(label.screen_dir_prelim)) return "予備の向きを選んでください。";
    if (!["yes","no","unknown"].includes(label.feint)) return "誘い（フェイント）かを選んでください。";
    if (label.feint === "yes" && label.motion !== "compound") return "誘い「はい」は複モーションで選んでください。";
    if (!["high","medium","low"].includes(label.confidence)) return "自信を選んでください。";
    return "";
  }
  function row(item, label, seconds, annotator) {
    const moving = ["single", "compound"].includes(label.motion);
    const final = moving ? label.screen_dir_final || "" : "";
    const prelim = label.motion === "compound" ? label.screen_dir_prelim || "" : "";
    const player = convert(final, item.front_side);
    return {id:item.id,point:item.point,contact_frame:item.contact_frame,front_side:item.front_side,
      motion:label.motion || "",unknown_reason:label.motion === "unknown" ? label.unknown_reason || "" : "",
      screen_dir_final:final,screen_dir_prelim:prelim,player_dir8_final:player,player_dir3_final:dir3(player),
      player_dir8_prelim:convert(prelim,item.front_side),feint:label.feint || "",confidence:label.confidence || "",
      note:label.note || "",seconds:(seconds / 1000).toFixed(3),annotator,saved_at:label.saved_at || ""};
  }
  function csv(rows) {
    const quote = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
    return "\ufeff" + [FIELDS, ...rows.map(r => FIELDS.map(k => r[k]))].map(r => r.map(quote).join(",")).join("\r\n") + "\r\n";
  }
  const Core = {DIRS,PLAYER,convert,dir3,elapsed,touch,suspend,valid,row,csv,FIELDS};
  globalThis.MotionCore = Core;
  if (typeof document === "undefined") return; // Node によるロジック検証用。ブラウザ依存なし。

  const $ = id => document.getElementById(id);
  const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  const STORAGE = "softtennis-motion-reference-1986-v1";
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
    const converted = row(current(),label(),label().seconds*1000,db.annotator);
    for (const key of ["player_dir8_final","player_dir3_final","player_dir8_prelim"]) label()[key] = converted[key];
    db.index = index;
    try {
      localStorage.setItem(STORAGE, JSON.stringify(db));
      $("storage").textContent = "このブラウザに自動保存済み。終了前に CSV も書き出してください。";
    } catch (e) { $("storage").textContent = `自動保存に失敗（${e.name}）。閉じる前に CSV を書き出してください。`; }
  }
  function act() {
    if (!data || blocked) return;
    touch(timer, Date.now());
    label().saved_at = new Date().toISOString();
  }
  function change(field, value) {
    act();
    const l = label();
    l[field] = value;
    l.completed = false;
    if (field === "motion") {
      if (value !== "unknown") l.unknown_reason = "";
      if (!["single","compound"].includes(value)) l.screen_dir_final = "";
      if (value !== "compound") { l.screen_dir_prelim = ""; if (l.feint === "yes") l.feint = ""; }
    }
    $("validation").textContent = "";
    persist(); renderForm();
  }
  function renderForm() {
    const l = label(), item = current();
    $("reasonWrap").hidden = l.motion !== "unknown";
    $("unknownReason").value = l.unknown_reason || "";
    $("prelimWrap").hidden = l.motion !== "compound";
    $("finalWrap").hidden = !["single","compound"].includes(l.motion);
    $("finalLegend").textContent = l.motion === "compound" ? "決定の向き（画面で見た向き）" : "向き（画面で見た向き）";
    $("orientation").textContent = item.front_side === "N" ? "手前：画面の上＝前、画面の右＝本人の右" : "奥：画面の下＝前、画面の左＝本人の右";
    for (const [prefix, field] of [["final","screen_dir_final"],["prelim","screen_dir_prelim"]]) {
      $(prefix + "Player").textContent = "前衛から見て：" + (JA[convert(l[field], item.front_side)] || "未選択");
      $(prefix + "Directions").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.dir === l[field])));
    }
    for (const field of ["motion","feint","confidence"]) {
      document.querySelectorAll(`[data-${field}]`).forEach(b => b.setAttribute("aria-pressed", String(b.dataset[field] === l[field])));
    }
    document.querySelector('[data-feint="yes"]').disabled = l.motion !== "compound";
    if ($("note").value !== (l.note || "")) $("note").value = l.note || "";
    const complete = data.items.filter(i => db.labels[i.id]?.completed && !valid(db.labels[i.id])).length;
    $("progress").textContent = `完了 ${complete} / ${data.items.length}件 · 表示 ${index + 1} / ${data.items.length}件`;
    $("prev").disabled = index === 0;
    $("next").textContent = index === data.items.length - 1 ? "完了する" : "次へ";
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
    const relative = (frame - item.contact_frame) / fps();
    $("frameInfo").textContent = `元動画 F${frame} · 打球から ${relative >= 0 ? "+" : ""}${relative.toFixed(3)}秒 · 打球 F${item.contact_frame}`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const box = item.boxes[String(frame)];
    if ($("showBox").checked && box) {
      const [x1,y1,x2,y2] = box;
      // 黒い下枠で、白いユニフォーム・明るいコートでも細い青枠を識別する。
      ctx.strokeStyle = "#06121b"; ctx.lineWidth = 4;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.strokeStyle = "#67e6ff"; ctx.lineWidth = 2;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    }
    $("videoStatus").textContent = !ready ? "動画を読み込み中…" : !item.front_slot ? "対象slotが未確定です。追跡枠は表示できません。判定できない場合は理由を記録してください。" : $("showBox").checked && !box ? "このフレームには対象slotの追跡枠がありません（補間しません）。" : "";
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
    if (blobFile === file && video.readyState >= 1) { ready = true; lockVideo(); seek(current().start_frame); return; }
    try {
      const response = await fetch(file);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      if (request !== token) return;
      const previous = blobUrl;
      blobUrl = URL.createObjectURL(blob); blobFile = file;
      video.src = blobUrl;
      video.load();
      if (previous) URL.revokeObjectURL(previous);
      video.onloadedmetadata = () => {
        if (request !== token) return;
        const item = current();
        if ((item.stop_frame - item.source_first_frame + .5) / fps() >= video.duration) {
          error("表示範囲が動画の長さを超えています。data.json を再生成してください。"); return;
        }
        ready = true; video.playbackRate = Number($("speed").value); lockVideo(); seek(item.start_frame);
      };
    } catch (e) { if (request === token) error(`動画の取得に失敗：${e.message}。HTTPで開き、サーバーの起点を確認してください。`); }
  }
  function choose(nextIndex) {
    if (selected) { suspend(timer, Date.now()); persist(); }
    pause(); ready = false; seekTarget = seekQueue = null;
    index = nextIndex;
    selected = true;
    const item = current();
    timer = {ms:(label().seconds || 0) * 1000,last:null};
    $("itemTitle").textContent = `${index + 1}. 第${item.point}点 / ラリー${item.rally} / ${item.id}`;
    $("side").textContent = item.front_side === "N" ? "手前の前衛 N" : "奥の前衛 F";
    $("target").textContent = `追跡slot：${item.front_slot || "未確定"} · 表示範囲：打球の約0.5秒前〜0.8秒後`;
    canvas.width = item.box_width; canvas.height = item.box_height;
    $("timeline").min = item.start_frame; $("timeline").max = item.stop_frame;
    for (const [id, frame] of [["contactMarker",item.contact_frame],["endMarker",item.marker_04_frame]]) $(id).style.left = `${100 * (frame - item.start_frame) / (item.stop_frame - item.start_frame)}%`;
    $("validation").textContent = "";
    draw(item.start_frame); renderForm(); lockVideo(); persist();
    setVideoSource(item.clip);
  }
  function next() {
    act();
    const l = label(), message = valid(l);
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
    const a = document.createElement("a"); a.href = url; a.download = "motion_labels.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  // 全8方向を画面上の3×3配置に固定（中心は空欄）。
  for (const [prefix,field] of [["final","screen_dir_final"],["prelim","screen_dir_prelim"]]) {
    const positions = [[1,2],[1,3],[2,3],[3,3],[3,2],[3,1],[2,1],[1,1]];
    DIRS.forEach((dir,i) => {
      const b = document.createElement("button"); b.type = "button"; b.dataset.dir = dir; b.textContent = ARROWS[i];
      b.style.gridRow = positions[i][0]; b.style.gridColumn = positions[i][1]; b.setAttribute("aria-label", `画面の${["上","右上","右","右下","下","左下","左","左上"][i]}`);
      b.addEventListener("click", () => change(field,dir)); $(prefix+"Directions").append(b);
    });
  }
  for (const field of ["motion","feint","confidence"]) document.querySelectorAll(`[data-${field}]`).forEach(b => b.addEventListener("click", () => change(field,b.dataset[field])));
  $("unknownReason").addEventListener("change", e => change("unknown_reason",e.target.value));
  $("note").addEventListener("input", e => change("note",e.target.value));
  $("annotator").addEventListener("input", e => { act(); db.annotator = e.target.value; persist(); });
  $("play").addEventListener("click",togglePlay); video.addEventListener("click",togglePlay);
  document.querySelectorAll("[data-step]").forEach(b => b.addEventListener("click", () => step(Number(b.dataset.step))));
  $("timeline").addEventListener("input", e => { act(); pause(); seek(Number(e.target.value)); persist(); });
  $("speed").addEventListener("change", e => { act(); video.playbackRate = Number(e.target.value); persist(); });
  for (const id of ["loop","showBox"]) $(id).addEventListener("change", () => { act(); draw(shownFrame); persist(); });
  $("next").addEventListener("click",next);
  $("prev").addEventListener("click", () => { if (index > 0) { act(); choose(index-1); } });
  $("export").addEventListener("click",exportCsv);
  $("labelForm").addEventListener("submit", e => e.preventDefault());
  document.addEventListener("keydown", e => {
    if (!data || blocked || e.altKey || e.ctrlKey || e.metaKey || e.isComposing || e.repeat) return;
    if (e.target.closest("input,textarea,select,summary,[contenteditable=true]")) return;
    if ([" ","ArrowLeft","ArrowRight","1","2","3","Enter"].includes(e.key)) e.preventDefault();
    if (e.key === " ") togglePlay();
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") step((e.key === "ArrowLeft" ? -1 : 1)*(e.shiftKey?6:1));
    else if (["1","2","3"].includes(e.key)) change("motion",["none","single","compound"][Number(e.key)-1]);
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
      db = {version:1,dataset_id:data.dataset_id,annotator:"",index:0,labels:{}};
      let raw;
      try { raw = localStorage.getItem(STORAGE); } catch (e) { storageWarning = `保存領域を読めません（${e.name}）。CSVで保存してください。`; }
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.version !== 1 || saved.dataset_id !== data.dataset_id || !saved.labels || typeof saved.labels !== "object" || typeof saved.annotator !== "string") throw new Error("保存形式・対象データが一致しません。既存の保存を上書きせず停止しました");
        db = saved;
      }
      $("annotator").value = db.annotator;
      document.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = false; });
      clearError(); choose(Math.max(0,Math.min(data.items.length-1,Number(db.index)||0)));
      if (storageWarning) $("storage").textContent = storageWarning;
      setInterval(() => { if (!blocked) { $("timing").textContent = `作業時間 ${ (elapsed(timer,Date.now())/1000).toFixed(1)}秒（60秒以上の無操作は除外）`; persist(); } }, 5000);
    } catch (e) { blocked = true; error(`開始できません：${e.message}。README の起動方法を確認してください。`); }
  }
  init();
})();
