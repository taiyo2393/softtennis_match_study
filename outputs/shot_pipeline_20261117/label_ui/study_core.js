/* Pure bookkeeping shared by the UI and the offline checks. No network or dependencies. */
"use strict";
globalThis.LabelStudyCore = (() => {
  const IDLE_MS = 60000;
  const iso = ms => ms == null ? "" : new Date(ms).toISOString();
  function frameFromTime(seconds, clip) {
    return Math.max(0, Math.min(clip.frame_count - 1,
      Math.floor(seconds * clip.fps_num / clip.fps_den + 0.0001)));
  }
  function makeSession(clip, participant, condition, version, now, id) {
    return { id, clip_id:clip.id, point:clip.point, rally:clip.rally, participant,
      condition, candidate_version:condition === "with_candidates" ? version : "none",
      created_at:now, start:null, end:null, last_seen:null, actions:[], away:[],
      playback_ms:0, labels:[], rally_breaks:[], compared_at:null, reference_exposed_before_start:false };
  }
  function timing(s, now) {
    if (s.start == null) return { elapsed:0, active:0, idle:0, away:0 };
    const stop = s.end ?? now;
    const intervals = [];
    const times = s.actions.map(a => a.at).concat(stop);
    for (let i = 1; i < times.length; i++) {
      if (times[i] - times[i - 1] >= IDLE_MS) intervals.push([times[i - 1], times[i]]);
    }
    let away = 0;
    for (const a of s.away) {
      const left = Math.max(s.start, a.start), right = Math.min(stop, a.end ?? stop);
      if (right > left) { intervals.push([left, right]); away += right - left; }
    }
    intervals.sort((a,b) => a[0] - b[0]);
    let idle = 0, rightmost = s.start;
    for (const [left,right] of intervals) {
      idle += Math.max(0, right - Math.max(left, rightmost));
      rightmost = Math.max(rightmost, right);
    }
    const elapsed = Math.max(0, stop - s.start);
    return { elapsed, active:Math.max(0,elapsed - idle), idle, away };
  }
  function record(s, type, now, detail = {}) {
    if (s.end != null) return false;
    if (s.start == null) s.start = now;
    s.last_seen = now;
    s.actions.push({at:now, type, ...detail});
    return true;
  }
  function suspend(s, now, reason) {
    if (s.start != null && s.end == null && !s.away.some(a => a.end == null)) {
      s.away.push({start:now, end:null, reason});
    }
  }
  function resume(s, now) {
    for (const a of s.away) if (a.end == null) a.end = now;
    if (s.start != null && s.end == null) s.last_seen = now;
  }
  function nearest(labels, frame) {
    return labels.filter(l => Math.abs(l.frame-frame) <= 10)
      .sort((a,b) => Math.abs(a.frame-frame)-Math.abs(b.frame-frame) || a.frame-b.frame)[0];
  }
  function rallyNo(s, frame) {
    return 1 + s.rally_breaks.filter(f => f <= frame).length;
  }
  function normalize(s) {
    s.rally_breaks.sort((a,b) => a-b);
    s.labels.sort((a,b) => a.frame-b.frame);
    s.labels.forEach(l => { l.rally_no = rallyNo(s,l.frame); });
    s.labels.forEach((l,i) => {
      if (s.labels[i+1]?.rally_no === l.rally_no) l.outcome = "continue";
      else if (l.outcome === "continue") l.outcome = "unknown";
    });
  }
  function edit(s, kind, frame) {
    if (s.end != null) return false;
    const rally = rallyNo(s,frame);
    let l = nearest(s.labels.filter(l => rallyNo(s,l.frame) === rally),frame);
    if (kind === "near" || kind === "far") {
      if (!l) { l = {frame,side:"N",is_serve:false,outcome:"unknown"}; s.labels.push(l); }
      l.frame = frame; l.side = kind === "near" ? "N" : "F";
    } else if (kind === "serve") {
      if (!l) return false;
      l.is_serve = !l.is_serve;
    } else if (kind === "delete") {
      if (!l) return false;
      s.labels.splice(s.labels.indexOf(l),1);
    } else if (kind === "outcome") {
      l = s.labels.filter(l => rallyNo(s,l.frame) === rally).at(-1);
      if (!l) return false;
      const values = ["unknown","winner","out","net"];
      l.outcome = values[(values.indexOf(l.outcome)+1)%values.length];
    } else if (kind === "rally_break") {
      const index = s.rally_breaks.indexOf(frame);
      if (index < 0) s.rally_breaks.push(frame);
      else s.rally_breaks.splice(index,1);
    } else return false;
    normalize(s);
    return true;
  }
  function counts(s) {
    const result = {};
    s.actions.forEach(a => { result[a.type] = (result[a.type] || 0)+1; });
    return result;
  }
  function csv(rows, columns) {
    // Protect participant names from spreadsheet formula evaluation.
    const escape = v => '"' + String(v ?? "").replace(/^[=+@\-]/, "'$&").replace(/"/g,'""') + '"';
    return "\ufeff" + [columns.map(escape).join(","),...rows.map(r => columns.map(k => escape(r[k])).join(","))].join("\r\n")+"\r\n";
  }
  function exportRows(sessions, now) {
    const labels=[], work=[];
    for (const s of sessions.filter(s => s.start != null)) {
      const common = {trial_id:s.id, participant:s.participant, point:s.point, rally:s.rally,
        condition:s.condition, candidate_version:s.candidate_version, status:s.end == null ? "in_progress" : "completed"};
      s.labels.forEach(l => labels.push({...common,...l,rally_no:rallyNo(s,l.frame),is_serve:l.is_serve ? "true" : "false"}));
      const t=timing(s,now);
      work.push({...common,start:iso(s.start),end:iso(s.end),elapsed_seconds:(t.elapsed/1000).toFixed(3),
        active_seconds:(t.active/1000).toFixed(3),idle_seconds:(t.idle/1000).toFixed(3),
        away_seconds:(t.away/1000).toFixed(3),playback_seconds:(s.playback_ms/1000).toFixed(3),
        n_actions_by_type:JSON.stringify(counts(s)),compared_at:iso(s.compared_at),
        reference_exposed_before_start:s.reference_exposed_before_start});
    }
    return {labels,work};
  }
  return {IDLE_MS,iso,frameFromTime,makeSession,timing,record,suspend,resume,nearest,rallyNo,edit,counts,csv,exportRows};
})();
