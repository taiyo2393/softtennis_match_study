"""Build an offline shot-table viewer. Read existing CSVs; write only this folder."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parents[1]
SHOT_CSV = ROOT / "outputs/shot_table_20261117/shot_table.csv"
POINT_CSV = ROOT / "outputs/match2_point_inventory_20261117/point_inventory.csv"
SHOT_README = SHOT_CSV.with_name("README.md")
EVIDENCE_BASE = ROOT / "outputs/ball_quality_20261111/match2_points_4_6_stroke_review"
PLAYERS = ("N1", "N2", "F1", "F2")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    result = []
    integer_fields = {
        "point", "shot_index", "game", "players_available", "contact_frame_first",
        "contact_frame_last", "contact_frame_ref", "display_from_frame", "next_display_from_frame",
    }
    boolean_fields = {"is_serve", "hitter_won_point", "is_final_game"}
    for row in raw:
        item = {}
        for key, value in row.items():
            assert value is not None, f"Malformed CSV row: {path.name}, {key}"
            if value == "":
                item[key] = None
            elif key in boolean_fields:
                assert value in {"True", "False"}, (key, value)
                item[key] = value == "True"
            elif key in integer_fields or key.endswith("_frame_offset"):
                item[key] = int(value)
            elif key.endswith("_m"):
                item[key] = float(value)
                assert math.isfinite(item[key]), (key, value)
            else:
                item[key] = value
        result.append(item)
    return result


class HTMLAudit(HTMLParser):
    """Check nesting and IDs, and extract the actual embedded JSON after writing."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack = []
        self.ids = set()
        self.json_parts = []
        self.capture = False
        self.app_script_parts = []
        self.capture_app = False

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if "id" in attr:
            assert attr["id"] not in self.ids, f"Duplicate id: {attr['id']}"
            self.ids.add(attr["id"])
        assert not (tag == "script" and "src" in attr), "External scripts are forbidden"
        assert not (tag == "link" and attr.get("rel") == "stylesheet"), "External CSS is forbidden"
        if tag not in self.VOID:
            self.stack.append(tag)
        if tag == "script":
            self.capture = attr.get("id") == "viewer-data"
            self.capture_app = attr.get("id") == "viewer-app"

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, f"Unbalanced tag: {tag}; {self.stack}"
        self.stack.pop()
        if tag == "script":
            self.capture = self.capture_app = False

    def handle_data(self, data):
        if self.capture:
            self.json_parts.append(data)
        if self.capture_app:
            self.app_script_parts.append(data)


HTML = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="match2 第4〜6点の打球候補・推定配置・結果・根拠画像を対応づける、オフラインの最小試作。">
<title>match2 打球単位の戦術データ（試作）</title>
<style>
:root{color-scheme:light dark;--bg:#f3f5f8;--surface:#fff;--soft:#edf1f6;--text:#1d2c3b;--muted:#536474;--line:#c6d0db;--near:#1766b2;--far:#aa490c;--u:#006b60;--f:#7541a0;--warn:#9a4b00;--error:#b3253f;--court:#edf5ef;--court-line:#486858;--selected:#183e67;--shadow:0 3px 16px #20364c09}
@media(prefers-color-scheme:dark){:root{--bg:#111820;--surface:#1a2530;--soft:#22323f;--text:#e8eff6;--muted:#b3c3d1;--line:#4d6070;--near:#77bdff;--far:#ffb779;--u:#69d3be;--f:#d1a6ff;--warn:#ffd08b;--error:#ff9cac;--court:#20382e;--court-line:#b4cbbc;--selected:#9dcfff;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Yu Gothic","Hiragino Sans",sans-serif;font-size:14px;line-height:1.7}main{max-width:1320px;margin:auto;padding:28px 24px 38px}h1{font-size:clamp(19px,2.2vw,27px);line-height:1.55;margin:0 0 7px;letter-spacing:.025em}h2{font-size:18px;margin:0;line-height:1.6}h3{font-size:15px;margin:0 0 8px}p{margin:6px 0 12px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.12em;color:var(--muted);margin:0 0 6px}.subtitle{color:var(--muted);font-size:14px;margin-bottom:21px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-top:16px;box-shadow:var(--shadow)}.heading-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px}.small{font-size:12px;color:var(--muted)}.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:12px;color:var(--muted)}.swatch{display:inline-block;width:10px;height:10px;border-radius:3px;background:currentColor;margin-right:5px}.u{color:var(--u)}.f{color:var(--f)}.near{color:var(--near)}.far{color:var(--far)}button,a{font:inherit}button{color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:6px;cursor:pointer}button:hover{background:var(--soft)}button:focus-visible,a:focus-visible{outline:3px solid var(--selected);outline-offset:3px}button:disabled{opacity:.4;cursor:default}a{color:var(--near)}.flow-scroll{overflow-x:auto;padding:4px 2px 10px}.flow-track{display:flex;gap:6px;width:max-content;min-width:100%}.game-group{display:flex;flex-direction:column;gap:4px;border-right:1px solid var(--line);padding-right:5px;flex:1}.game-group:last-child{border-right:0;padding-right:0}.game-label{font-size:10px;color:var(--muted)}.game-cells{display:flex;gap:2px}.point-cell{width:15px;height:27px;padding:0;font-size:9px;line-height:1;border:1px solid transparent;border-radius:3px;color:var(--surface);background:var(--u);position:relative}.point-cell.winner-f{background:var(--f)}.point-cell:hover{background:var(--u);filter:brightness(1.18)}.point-cell.winner-f:hover{background:var(--f)}.point-cell.has-shots{border:3px solid var(--text)}.point-cell[aria-pressed="true"]{outline:2px solid var(--selected);outline-offset:3px;z-index:1}.point-cell[aria-pressed="true"]:after{content:"";position:absolute;width:5px;height:5px;border-radius:50%;background:var(--text);left:50%;bottom:-9px;transform:translateX(-50%)}.point-context{margin:5px 0 0}.shot-buttons{display:flex;flex-wrap:wrap;gap:8px}.shot-button{padding:7px 12px;min-height:40px;font-size:13px}.shot-button[aria-pressed="true"]{border:2px solid var(--selected);padding:6px 11px;background:var(--soft);font-weight:700}.outcome-mark{color:var(--warn);font-weight:700}.nav-buttons{display:flex;gap:7px}.nav-buttons button{padding:4px 11px;font-size:12px}.empty{background:var(--soft);border:1px dashed var(--line);padding:18px;border-radius:8px;color:var(--muted)}.detail-grid{display:grid;grid-template-columns:minmax(280px,.88fr) minmax(360px,1.12fr);gap:25px;align-items:start}.court-wrap{max-width:440px;margin:auto}.court-wrap svg{display:block;width:100%;height:auto;max-height:640px}.court-line{fill:none;stroke:var(--court-line);stroke-width:.075}.net{stroke:var(--text);stroke-width:.12;stroke-dasharray:.23 .12}.court-label{font-size:.48px;fill:var(--muted)}.player-label{font-size:.57px;font-weight:700;paint-order:stroke;stroke:var(--court);stroke-width:.16px;stroke-linejoin:round}.dimension-label{font-size:.45px;font-weight:700;paint-order:stroke;stroke:var(--court);stroke-width:.19px;stroke-linejoin:round}.pair-line{fill:none;stroke-width:.055;stroke-dasharray:.16 .10;opacity:.85}.movement{stroke-width:.10;opacity:.30}.metric-row{display:flex;gap:8px;margin:5px 0}.metric-row div{flex:1;background:var(--soft);border-radius:7px;padding:7px 9px;font-size:12px}.missing{font-size:13px;margin:6px 0;color:var(--warn)}table{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}caption{text-align:left;font-weight:700;font-size:15px;margin-bottom:6px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:7px 9px;overflow-wrap:anywhere}th{width:37%;font-weight:500;color:var(--muted);padding-left:0}td{padding-right:0}.offset-note{font-size:12px;color:var(--warn);margin-top:8px}.evidence{margin:16px 0 0}.evidence img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:7px;background:var(--soft)}figcaption{font-size:11px;line-height:1.7;color:var(--muted);overflow-wrap:anywhere;margin-top:5px}.trend-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.trend-chart{margin:0;min-width:0}.trend-chart svg{display:block;width:100%;height:auto}.trend-chart text{font-family:inherit;font-size:11px;fill:var(--text)}.trend-chart .axis{stroke:var(--line);stroke-width:1}.trend-chart .grid{stroke:var(--line);stroke-dasharray:3 4;opacity:.6}.trend-chart .chart-muted{fill:var(--muted);font-size:10px}.trend-chart .gain{color:var(--u);fill:var(--u);stroke:var(--u)}.trend-chart .loss{color:var(--error);fill:var(--error);stroke:var(--error)}.trend-warning{font-weight:700;color:var(--warn);font-size:13px;margin:10px 0 0}.notes{font-size:12px;color:var(--muted);padding-left:1.4em;margin:8px 0}.notes li{margin:4px 0}.footer{font-size:11px;color:var(--muted);margin-top:16px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}[hidden]{display:none!important}
@media(max-width:760px){main{padding:18px 12px 28px}.panel{padding:14px}.detail-grid,.trend-grid{grid-template-columns:1fr}.court-wrap{max-width:380px}.legend{gap:5px 12px}.point-cell{width:18px;font-size:10px}.detail-grid{gap:18px}}
</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">11/20 指導教員向けデモ · 22打球候補 / 66点</p>
<h1>match2（2024 全日本選手権 男子決勝）打球単位の戦術データ（試作）</h1>
<p class="subtitle">位置は推定値、打球は候補。第4〜6点のみ打球データあり</p>
</header>
<section class="panel" aria-labelledby="flow-title">
<div class="heading-row"><h2 id="flow-title">試合の流れ</h2><span class="small">66点 · 太枠＝打球データあり · マスをクリックして選択</span></div>
<div class="legend"><span class="u"><i class="swatch"></i>U＝内本・内田</span><span class="f"><i class="swatch"></i>F＝船水・上松</span><span>色＝点の得点ペア / スコアはU–F順</span></div>
<div class="flow-scroll" tabindex="0" aria-label="66点の一覧。幅が狭い場合は横にスクロール"><div id="point-flow" class="flow-track"></div></div>
<p id="point-context" class="small point-context" aria-live="polite"></p>
</section>
<section class="panel" aria-labelledby="shots-title">
<div class="heading-row"><h2 id="shots-title">選んだ点の打球の列</h2><div class="nav-buttons"><button id="prev-shot" type="button">← 前の打球</button><button id="next-shot" type="button">次の打球 →</button></div></div>
<div id="shot-buttons" class="shot-buttons" aria-label="点内の打球候補"></div>
<p class="small" style="margin-bottom:0">左右キーでも前後の候補へ移動（第4〜6点を通して順送り）。⚠＝ネット / ?＝結果未判定。</p>
</section>
<section class="panel" aria-labelledby="detail-title">
<div class="heading-row"><h2 id="detail-title">選んだ打球の詳細</h2><span id="shot-status" class="small"></span></div>
<div id="no-shot" class="empty" hidden>この点には打球データがありません。第4・5・6点を選択してください。</div>
<div id="detail-content" class="detail-grid">
<div>
<div class="legend"><span class="near">● N1 / N2＝手前</span><span class="far">● F1 / F2＝奥</span><span>★＝打者（位置がある場合）</span></div>
<div class="court-wrap"><svg id="court" role="img" aria-label="選んだ打球候補の推定配置"></svg></div>
<p id="missing-players" class="missing"></p>
<div id="pair-metrics" class="metric-row"></div>
<p id="movement-note" class="small"></p>
<p class="small" style="margin-bottom:0">yは手前0 → 奥23.77 m。破線の折れ線＝横間隔・前後差。外枠外の推定値も表示。選手IDのFは「奥側」で、上の得点ペアFとは別です。</p>
</div>
<div>
<table><caption id="detail-caption">打球候補の記録</caption><tbody id="detail-table"></tbody></table>
<p id="offset-note" class="offset-note"></p>
<figure class="evidence"><h3>根拠画像</h3><a id="evidence-link" target="_blank" rel="noopener"><img id="evidence-image" alt="" decoding="async" hidden></a><div id="no-image" class="empty" hidden>画像なし</div><figcaption id="evidence-caption"></figcaption></figure>
</div>
</div>
</section>
<section class="panel" aria-labelledby="trend-title">
<div class="heading-row"><h2 id="trend-title">傾向の例</h2><span class="small">第4〜6点の全候補 · 指標ごとに欠測を除外</span></div>
<p class="small">打者側がその点を取ったか・失ったかで群分け。当該打球の成功・失敗ではありません。点1つ＝打球候補1件、縦線＝中央値。</p>
<div class="trend-grid"><figure id="lateral-trend" class="trend-chart"></figure><figure id="depth-trend" class="trend-chart"></figure></div>
<p class="trend-warning">3点だけの例。検定はしていない。打球は独立ではない</p>
<p class="small" style="margin-bottom:0">同一点の複数候補を含みます。配置の良否や得失点への因果効果、一般的な戦術傾向は判断できません。</p>
</section>
<section class="panel" aria-labelledby="notes-title">
<h2 id="notes-title">この試作の注意点</h2>
<ul class="notes">
<li>位置は検出箱の下端から推定した値です。遮蔽等の欠測は補完していません。位置があることは、位置や人物対応の正しさを保証しません。</li>
<li>打球は候補です。順序・打者・打球数の完全性は独立検証していません。第4点最後の画面外N1も仮の打者です。</li>
<li>接触の参照フレームは候補区間の中央で、接触確定時刻ではありません。選手ごとの採用時刻がずれる場合があり、厳密な同時刻の配置とは限りません。</li>
<li>サーブ3候補は全員の位置が欠測。第4・5点の終わり方は未判定です。continueは「次の候補あり」、第6点最後のみネットが人手確認済み（正確な接触時刻は未特定）です。</li>
<li>全66点の一覧はスコア表示の書き起こしに基づき、第3ゲーム1–1には補完を含みます。表示開始フレームはサーブ・ラリー開始時刻ではありません。</li>
</ul>
<p class="small" style="margin-bottom:0">示すのは既存データ・欠測・根拠画像の対応です。自動抽出の精度、戦術の有効性、映像確認を支援する効果は未評価です。</p>
</section>
<footer class="footer">入力: shot_table_20261117/shot_table.csv / match2_point_inventory_20261117/point_inventory.csv · 画像は既存ファイルの相対参照 · オフライン動作</footer>
<noscript><p class="empty">この試作を表示するにはブラウザのJavaScriptを有効にしてください。</p></noscript>
</main>
<script type="application/json" id="viewer-data">__VIEWER_JSON__</script>
<script id="viewer-app">
"use strict";
const DATA = JSON.parse(document.getElementById("viewer-data").textContent);
const SHOTS = DATA.shots, POINTS = DATA.points, PLAYERS = ["N1", "N2", "F1", "F2"];
const $ = id => document.getElementById(id);
const fmt = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
const frame = value => value === null ? "—" : "F" + value;
const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const winnerName = p => p.point_winner === "U" ? "U（内本・内田）" : p.point_winner === "F" ? "F（船水・上松）" : "—";
const inferredScore = p => (p.before_state_source || "").startsWith("inferred");
const sideName = side => side === "near" ? "手前（near）" : side === "far" ? "奥（far）" : "—";
const outcomeName = {continue:"次の候補あり（continue）", error_net:"⚠ ネット（人手確認）", unknown:"? 未判定（unknown）"};
const statusName = {candidate:"候補", offscreen:"画面外・打者も仮候補", user_confirmed:"人手確認あり"};
const available = (s,p) => Number.isFinite(s[p+"_x_m"]) && Number.isFinite(s[p+"_y_m"]);
let selectedPoint = 6, selectedIndex = SHOTS.findIndex(s => s.point === 6 && s.shot_index === 1);
const pointButtons = new Map();

function buildFlow() {
  const groups = new Map();
  POINTS.forEach(p => {
    if (!groups.has(p.game)) {
      const group = document.createElement("div"); group.className = "game-group";
      const label = document.createElement("span"); label.className = "game-label";
      label.textContent = p.is_final_game ? "FINAL" : "G" + p.game;
      const cells = document.createElement("div"); cells.className = "game-cells";
      group.append(label, cells); $("point-flow").append(group); groups.set(p.game,cells);
    }
    const hasShots = SHOTS.some(s => s.point === p.point);
    const b = document.createElement("button"); b.type = "button";
    b.className = "point-cell" + (p.point_winner === "F" ? " winner-f" : "") + (hasShots ? " has-shots" : "");
    b.textContent = p.point;
    const tip = `第${p.point}点 / 第${p.game}ゲーム / ゲーム数 ${p.games_before} / スコア ${p.score_before} → ${p.score_after}（U–F順） / 得点 ${winnerName(p)}${inferredScore(p) ? " / スコア補完あり" : ""}`;
    b.title = tip; b.setAttribute("aria-label",tip + (hasShots ? " / 打球データあり" : " / 打球データなし"));
    b.addEventListener("click",() => {selectedPoint=p.point; selectedIndex=SHOTS.findIndex(s => s.point===p.point); renderSelection();});
    groups.get(p.game).append(b); pointButtons.set(p.point,b);
  });
}

function drawCourt(s, previous) {
  const svg = $("court"), sy = y => 23.77-y, pos = p => [s[p+"_x_m"], sy(s[p+"_y_m"])];
  svg.setAttribute("viewBox","-3.6 -3.4 18.17 31.77");
  const title = `第${s.point}点 ${s.shot_index}打目 ${s.hitter}、推定配置。打者は星。`;
  svg.setAttribute("aria-label",title);
  const parts = [`<title>${esc(title)}</title><desc>コート外の位置も表示。数値は既存表の横間隔と前後差。薄い矢印は直前候補からの推定位置差。</desc><defs>`];
  for (const side of ["near","far"]) parts.push(`<marker id="arrow-${side}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6 Z" fill="var(--${side})"/></marker>`);
  parts.push(`</defs><rect x="0" y="0" width="10.97" height="23.77" style="fill:var(--court)" class="court-line"/><path class="court-line" d="M1.37 0V23.77 M9.60 0V23.77 M1.37 ${sy(5.49)}H9.60 M1.37 ${sy(18.28)}H9.60 M5.485 ${sy(5.49)}V${sy(18.28)}"/><line x1="-.25" x2="11.22" y1="${sy(11.885)}" y2="${sy(11.885)}" class="net"/><text x="11.48" y="${sy(11.885)+.16}" class="court-label">ネット</text><text x="5.485" y="-2.65" text-anchor="middle" class="court-label">奥（far） y=23.77 m</text><text x="5.485" y="27.55" text-anchor="middle" class="court-label">手前（near） y=0 m</text>`);
  let arrowCount = 0;
  if (previous) for (const p of PLAYERS) {
    if (!available(s,p) || !available(previous,p)) continue;
    const [x,y] = pos(p), px=previous[p+"_x_m"], py=sy(previous[p+"_y_m"]);
    if (x===px && y===py) continue;
    const side = p[0]==="N" ? "near":"far";
    parts.push(`<line x1="${px}" y1="${py}" x2="${x}" y2="${y}" stroke="var(--${side})" class="movement" marker-end="url(#arrow-${side})"/>`); arrowCount++;
  }
  const metricCards=[];
  for (const [side,ids] of [["near",["N1","N2"]],["far",["F1","F2"]]]) {
    const prefix = s.hitter_side===side ? "hitter_pair":"opponent_pair";
    const lateral=s[prefix+"_lateral_m"], depth=s[prefix+"_depth_m"];
    metricCards.push(`<div><strong class="${side}">${side==="near"?"手前 N1/N2":"奥 F1/F2"}</strong><br>横 ${fmt(lateral)} m / 前後 ${fmt(depth)} m</div>`);
    if (!ids.every(p=>available(s,p))) continue;
    const [x1,y1]=pos(ids[0]), [x2,y2]=pos(ids[1]);
    parts.push(`<path d="M${x1} ${y1}H${x2}V${y2}" stroke="var(--${side})" class="pair-line"/><text x="${(x1+x2)/2}" y="${y1-.35}" text-anchor="middle" fill="var(--${side})" class="dimension-label">横 ${fmt(lateral)} m</text><text x="${x2+(side==="near"?-.35:.35)}" y="${(y1+y2)/2}" text-anchor="${side==="near"?"end":"start"}" fill="var(--${side})" class="dimension-label">前後 ${fmt(depth)} m</text>`);
  }
  for (const p of PLAYERS) {
    if (!available(s,p)) continue;
    const [x,y]=pos(p), side=p[0]==="N"?"near":"far", hitter=s.hitter===p;
    if (hitter) {
      const points=Array.from({length:10},(_,i)=>{const a=-Math.PI/2+i*Math.PI/5,r=i%2===0?.43:.20;return `${x+Math.cos(a)*r},${y+Math.sin(a)*r}`;}).join(" ");
      parts.push(`<polygon points="${points}" fill="var(--${side})" stroke="var(--surface)" stroke-width=".07"><title>${p} 打者 / x=${fmt(s[p+"_x_m"])} m, y=${fmt(s[p+"_y_m"])} m</title></polygon>`);
    } else parts.push(`<circle cx="${x}" cy="${y}" r=".25" fill="var(--${side})" stroke="var(--surface)" stroke-width=".07"><title>${p} / x=${fmt(s[p+"_x_m"])} m, y=${fmt(s[p+"_y_m"])} m</title></circle>`);
    const shift = p.endsWith("1") ? -.48 : .48;
    parts.push(`<text x="${x+shift}" y="${y+.18}" text-anchor="${shift<0?"end":"start"}" class="player-label" fill="var(--${side})">${p}${hitter?" ★":""}</text>`);
  }
  svg.innerHTML=parts.join("");
  $("pair-metrics").innerHTML=metricCards.join("");
  const missing=PLAYERS.filter(p=>!available(s,p));
  $("missing-players").textContent=missing.length ? "欠測: "+missing.join("・") : "欠測: なし（4人の推定位置あり）";
  $("movement-note").textContent=!previous ? "点の先頭候補のため、前候補からの矢印はありません。" : arrowCount ? `薄い矢印: 同一点の直前候補 → 選択候補の推定位置差（${arrowCount}人）。実際の移動経路ではありません。` : "直前候補と両方の位置がある選手がいないため、移動矢印はありません。";
}

function renderDetails(s) {
  const previous=SHOTS.find(p=>p.point===s.point&&p.shot_index===s.shot_index-1);
  drawCourt(s,previous);
  $("shot-status").textContent=`第${s.point}点 · ${s.shot_index}打目 / ${statusName[s.status]||s.status}`;
  $("detail-caption").textContent=`第${s.point}点 ${s.shot_index}打目の記録`;
  const inventory=POINTS.find(p=>p.point===s.point);
  const coordinates=PLAYERS.map(p=>available(s,p)?`${p}: あり（x ${fmt(s[p+"_x_m"])} / y ${fmt(s[p+"_y_m"])} m）`:`${p}: —（欠測）`).join("<br>");
  const rows=[
    ["打者",`${s.hitter} / ${sideName(s.hitter_side)}${s.is_serve?" / サーブ":""}${s.status==="offscreen"?" / 仮候補":""}`],
    ["接触の参照フレーム",`${frame(s.contact_frame_ref)}（候補区間 ${frame(s.contact_frame_first)}〜${frame(s.contact_frame_last)}）`],
    ["4人の位置の有無",`${s.players_available}/4人<br>${coordinates}`],
    ["相手ペアの横間隔",`${fmt(s.opponent_pair_lateral_m)} m`],
    ["相手ペアの前後差",`${fmt(s.opponent_pair_depth_m)} m`],
    ["打者からネットまで",`${fmt(s.hitter_net_distance_m)} m（コート平面上）`],
    ["結果",outcomeName[s.shot_outcome]||s.shot_outcome],
    ["点の得点側",`${winnerName(inventory)} / ${sideName(s.point_winner_side)}`],
    ["打者側の点の勝敗",s.hitter_won_point===true?"得点（打球自体の成功判定ではない）":s.hitter_won_point===false?"失点（打球自体の失敗判定ではない）":"—"],
  ];
  // Row text comes from the local CSV and fixed labels; only coordinates include controlled <br> markup.
  $("detail-table").innerHTML=rows.map(([label,value])=>`<tr><th scope="row">${esc(label)}</th><td>${value.split("<br>").map(esc).join("<br>")}</td></tr>`).join("");
  const offsets=PLAYERS.filter(p=>available(s,p)&&Number.isFinite(s[p+"_frame_offset"])&&s[p+"_frame_offset"]!==0).map(p=>`${p}: 参照${s[p+"_frame_offset"]>0?"+":""}${s[p+"_frame_offset"]}フレーム`);
  $("offset-note").textContent=offsets.length?"採用時刻のずれ: "+offsets.join(" / ")+"。同時刻の配置とは限りません。":"採用時刻: 位置がある選手は参照フレームと一致。参照は接触確定時刻ではありません。";
  if (s.players_available===0) $("offset-note").textContent="候補区間内の位置データなし。座標・ペア指標・ネット距離は欠測のまま表示。";
  const img=$("evidence-image"), link=$("evidence-link"), empty=$("no-image");
  img.onerror=null; img.hidden=true; link.hidden=true; empty.hidden=true;
  img.removeAttribute("src"); link.removeAttribute("href");
  if (s.evidence_url) {
    img.alt=`第${s.point}点 ${s.shot_index}打目 ${s.hitter}の根拠画像（接触確定フレームとは限りません）`;
    img.onerror=()=>{if(img.getAttribute("src")===s.evidence_url){img.hidden=true;link.hidden=true;empty.hidden=false;}};
    link.href=s.evidence_url; img.src=s.evidence_url; img.hidden=false; link.hidden=false;
  } else empty.hidden=false;
  $("evidence-caption").textContent=`${s.evidence||"—"} / クリックで元画像を開く。画像のフレームと接触参照フレームは一致しない場合があります。`;
}

function renderSelection() {
  pointButtons.forEach((b,p)=>b.setAttribute("aria-pressed",String(p===selectedPoint)));
  const p=POINTS.find(p=>p.point===selectedPoint), shots=SHOTS.filter(s=>s.point===selectedPoint);
  $("point-context").textContent=`第${p.point}点 / 第${p.game}ゲーム / ゲーム数 ${p.games_before} / スコア ${p.score_before} → ${p.score_after}（U–F順） / 得点 ${winnerName(p)} / ${shots.length}候補${inferredScore(p)?" / スコア補完あり":""}`;
  $("shot-buttons").replaceChildren();
  for(const s of shots) {
    const index=SHOTS.indexOf(s), b=document.createElement("button");
    b.type="button"; b.className="shot-button"; b.setAttribute("aria-pressed",String(index===selectedIndex));
    const mark=s.shot_outcome==="error_net"?" ⚠ ネット":s.shot_outcome==="unknown"?" ? 未判定":"";
    b.textContent=`${s.shot_index} ${s.hitter}${s.is_serve?" サーブ":""}${mark}`;
    b.addEventListener("click",()=>{selectedIndex=index;renderSelection();});
    $("shot-buttons").append(b);
  }
  if(!shots.length){const text=document.createElement("p");text.className="small";text.textContent="打球データなし（第4〜6点のみ記録）";$("shot-buttons").append(text);}
  $("prev-shot").disabled=selectedIndex===0 || (selectedIndex<0&&!SHOTS.some(s=>s.point<selectedPoint));
  $("next-shot").disabled=selectedIndex===SHOTS.length-1 || (selectedIndex<0&&!SHOTS.some(s=>s.point>selectedPoint));
  $("detail-content").hidden=selectedIndex<0;$("no-shot").hidden=selectedIndex>=0;
  if(selectedIndex>=0)renderDetails(SHOTS[selectedIndex]);else $("shot-status").textContent=`第${selectedPoint}点 · 打球データなし`;
}

function moveShot(direction) {
  let next=selectedIndex+direction;
  if(selectedIndex<0) next=direction>0?SHOTS.findIndex(s=>s.point>selectedPoint):SHOTS.findLastIndex(s=>s.point<selectedPoint);
  if(next<0||next>=SHOTS.length)return;
  selectedIndex=next;selectedPoint=SHOTS[next].point;renderSelection();
}

function drawTrend(id,key,label) {
  const groups=[{win:true,name:"打者側が得点",cls:"gain"},{win:false,name:"打者側が失点",cls:"loss"}];
  const values=SHOTS.map(s=>s[key]).filter(Number.isFinite), max=Math.max(2,Math.ceil(Math.max(...values,0)/2)*2);
  const x=v=>110+v/max*342;
  const parts=[`<h3>${label}</h3><svg viewBox="0 0 490 204" role="img" aria-label="${label}の得失点別ドット図"><title>${label}：打者側の点の勝敗別。縦線は中央値。</title>`];
  for(let i=0;i<=4;i++) {const value=max*i/4,px=x(value);parts.push(`<line x1="${px}" x2="${px}" y1="24" y2="166" class="grid"/><text x="${px}" y="187" text-anchor="middle" class="chart-muted">${fmt(value)}</text>`);}
  const summaries=[];
  groups.forEach((g,index)=>{
    const all=SHOTS.filter(s=>s.hitter_won_point===g.win), rows=all.filter(s=>Number.isFinite(s[key])), y=62+index*70;
    const sorted=rows.map(s=>s[key]).sort((a,b)=>a-b), n=sorted.length;
    const median=n?(n%2?sorted[(n-1)/2]:(sorted[n/2-1]+sorted[n/2])/2):null;
    parts.push(`<text x="2" y="${y-6}">${g.name}</text><text x="2" y="${y+12}" class="chart-muted">有効 ${n} / ${all.length}候補</text>`);
    rows.forEach((s,i)=>{const dy=((i*3)%7-3)*4.4;parts.push(`<circle cx="${x(s[key])}" cy="${y+dy}" r="3.7" class="${g.cls}" fill-opacity=".78" stroke-width=".6"><title>第${s.point}点 ${s.shot_index}打目 ${s.hitter}: ${fmt(s[key])} m</title></circle>`);});
    if(median!==null)parts.push(`<line x1="${x(median)}" x2="${x(median)}" y1="${y-22}" y2="${y+22}" class="${g.cls}" stroke-width="2.5"/><text x="${x(median)}" y="${y-29}" text-anchor="middle" class="chart-muted">中央値 ${fmt(median)}</text>`);
    summaries.push(`${g.name}: 有効${n}件、中央値 ${fmt(median)} m`);
  });
  parts.push(`<line x1="110" x2="452" y1="166" y2="166" class="axis"/><text x="476" y="187" class="chart-muted">m</text></svg><figcaption>${summaries.join(" / ")}。ドットの上下ずらしは重なりを避ける表示で、値は横位置のみ。</figcaption>`);
  $(id).innerHTML=parts.join("");
}

$("prev-shot").addEventListener("click",()=>moveShot(-1));
$("next-shot").addEventListener("click",()=>moveShot(1));
document.addEventListener("keydown",event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.target.isContentEditable||/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName))return;
  if(event.key==="ArrowLeft"||event.key==="ArrowRight"){event.preventDefault();moveShot(event.key==="ArrowLeft"?-1:1);}
});
buildFlow();renderSelection();
drawTrend("lateral-trend","opponent_pair_lateral_m","相手ペアの横間隔");
drawTrend("depth-trend","opponent_pair_depth_m","相手ペアの前後差");
</script>
</body>
</html>
'''


README = '''# match2 打球単位の戦術データ・最小画面試作

11/20の指導教員向けデモ用。既存の表・画像を対応づけて閲覧する、日本語・オフラインの1枚HTMLです。

## 開き方

`index.html` をブラウザで開いてください。サーバー、ネット接続、CDN、外部ライブラリは不要です。データはHTML内のJSONに埋め込まれています。
画像は既存フォルダへの相対参照なので、リポジトリ内のフォルダ関係を保ってください。HTMLだけを別の場所へコピーすると画像が開けなくなります。

再生成はリポジトリのルートで、次を実行します。

```powershell
& study\\.venv\\Scripts\\python.exe outputs\\shot_viewer_20261117\\build_viewer.py
```

スクリプトの場所を基準に入力を解決するため、絶対パスで実行しても動作します。Python標準ライブラリのみを使用します。既存入力は読み取り専用で、書き出すのはこのフォルダの `index.html` と `README.md` だけです。

## 画面と操作

1. 見出しと、推定位置・打球候補・対象3点という前提。
2. 試合の流れ: 全66点を横1列に表示。ゲームで区切り、得点ペアをU/Fで色分けします。スコアはU–F順です。太枠の第4・5・6点には打球データがあります。各マスのツールチップにゲーム・スコア・得点側を表示。狭い画面では横スクロールできます。ほかの点も選べますが「打球データなし」と表示します。
3. 選んだ点の打球ボタン: 順序・打者・サーブ・結果未判定/ネットを表示。初期状態は第6点1打目の **F1 サーブ**（表の記録どおり）。左右キーまたは前後ボタンで全22候補を順送りし、点の境界も移動します。先頭・末尾では止まります。
4. 詳細: 左の縦長SVGコートに位置がある選手だけを描画し、打者は星形。手前N1/N2は青、奥F1/F2は橙です。欠測を明記し、コート外の値も切り詰めません。両ペアの横間隔・前後差を折れ線と数値で示します。薄い矢印は同一点の直前候補からの推定位置差で、経路を復元したものではありません。右にフレーム、座標の有無、表の指標・結果・得点側と根拠画像を表示します。画像はクリックで元画像を別タブに開きます。生成時に存在しない画像は参照URLを埋め込まず「画像なし」とし、表示時の読み込み失敗にも同じ表示を出します。
5. 傾向の例: 全22候補を打者側の「点の得失点」で群分けし、相手ペアの横間隔/前後差をドット表示。指標ごとに欠測を除き、中央値と有効件数/候補数を示します。縦方向のずらしは表示上の重なり回避のみです。サーブを含みますが、その指標は欠測です。
6. 注意点: 候補、欠測、時刻のずれ、未判定結果、3点の非独立性を明記。

フォントは `"Yu Gothic", "Hiragino Sans", sans-serif`。OS/ブラウザのライト・ダーク設定に追従し、狭い画面では詳細と傾向図を縦に並べます。距離と座標は小数2桁、欠測値は「—」。画面表示でのみ丸め、埋め込みJSONにはCSVの数値精度を保ちます。

## 使ったデータ

- `../shot_table_20261117/shot_table.csv`: UTF-8 BOM、全22打球候補（第4点6、第5点7、第6点9）。全列をJSONへ保存し、空欄をnull、True/Falseを明示的な真偽値へ変換します。
- `../shot_table_20261117/README.md`: 列の定義、結果ラベル、限界の参照。
- `../match2_point_inventory_20261117/point_inventory.csv`: 全66点。全列をJSONへ保存します。
- `../match2_point_inventory_20261117/README.md`: スコアのU–F順、補完、表示フレームの意味を確認。
- 根拠画像: 元の `evidence` は `../ball_quality_20261111/match2_points_4_6_stroke_review/` を基準に解決し、このHTMLからの相対URLへ変換します。画像は埋め込まず、移動・コピー・再解析もしません。

選手IDの **F1/F2のFはfar（奥）**、全点一覧の **得点ペアFは船水・上松**です。同じ記号を同一視しません。第4点は奥側/Uが得点、第5・6点は手前側/Fが得点と、各表に記録されています。

両ペアの数値は `hitter_side` に従い `hitter_pair_*` / `opponent_pair_*` を割り当て、距離の再推定は行いません。コート線は指定の外枠10.97×23.77 m、ネットy=11.885、サービスラインy=5.49/18.28、シングルス線x=1.37/9.60、センターサービス線を描画します。

## 限界

- 足元は検出箱下端による推定で、実距離の測量精度や人物対応の正しさを保証しません。ガード等で欠測となった値を補間・外挿しません。
- 打球候補・順序・打者・打球数の完全性は独立検証なし。第4点最後の画面外N1も仮候補です。
- 候補区間の中央フレームは接触確定時刻ではありません。第6点3打目F1は参照+1、第6点9打目F2は参照−11フレーム。各選手の位置がそろっても同時刻の配置とは限らず、該当時は詳細欄に明記します。
- サーブ3候補は全員欠測。`continue` は次の候補があるという記録で、成功確認ではありません。第4・5点最後は未判定、第6点最後のみネットが人手確認済みです。
- `hitter_won_point` は点の勝敗で、当該打球自体の成否ではありません。傾向図は3点の記述例で、同一点の打球は独立ではありません。検定なし。戦術の良否・因果効果・一般化可能性は評価できません。中央値と件数は要求された図の表示用集計のみで、新たな解析・推定は行いません。
- 全点一覧には第3ゲーム1–1のスコア補完が1件あります。`display_from_frame` は前スコアの表示開始で、サーブ・ラリーの開始時刻ではありません。
- 本試作で示すのはデータと根拠の対応です。完全自動抽出の精度、研究上の新規性、映像確認支援の効果は別途検証が必要です。

## 実行時の検証

- CSVの打球数22・点数66、点番号と打球順序、必要列、真偽値・有限数値をassertで確認。
- 書き出したHTMLを読み戻してJSONを抽出し、全22打球・66点が元データと一致することをassertで確認。
- 参照画像URLはHTMLの場所から再解決して存在をassert。画像候補{image_count}件、存在{existing_count}件、欠落{missing_count}件（存在しない画像はURLなし）。
- HTMLの要素の入れ子とID重複をHTMLParserで検証し、300,000 bytes未満をassert。
- 入力CSV2点・列定義READMEのSHA256が実行前後で変わらないことをassert。
- 動画は読まず、外部通信・追加ライブラリの取得・git commitは行いません。

ブラウザ上の見た目・クリック操作は利用者が確認する前提です。生成スクリプトの実行、埋め込みデータ・画像パスのassert、およびHTML/CSS/JavaScriptのソース確認までを実施します。
'''


def main():
    source_paths = (SHOT_CSV, POINT_CSV, SHOT_README)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    shots, points = read_csv(SHOT_CSV), read_csv(POINT_CSV)
    assert len(shots) == 22, f"Expected 22 shots, got {len(shots)}"
    assert len(points) == 66, f"Expected 66 points, got {len(points)}"
    assert [p["point"] for p in points] == list(range(1, 67))
    assert {s["point"] for s in shots} == {4, 5, 6}
    assert [(s["point"], s["shot_index"]) for s in shots] == [(p, i) for p, count in ((4, 6), (5, 7), (6, 9)) for i in range(1, count + 1)]
    assert all(p["point_winner"] in {"U", "F"} for p in points)
    required = {"hitter", "hitter_side", "status", "is_serve", "contact_frame_ref", "contact_frame_first", "contact_frame_last", "players_available", "shot_outcome", "point_winner_side", "hitter_won_point", "hitter_net_distance_m", "evidence"}
    required |= {f"{p}_{axis}_m" for p in PLAYERS for axis in ("x", "y")}
    required |= {f"{side}_pair_{metric}_m" for side in ("hitter", "opponent") for metric in ("lateral", "depth")}
    assert required <= shots[0].keys()
    assert {"game", "games_before", "score_before", "score_after", "point_winner", "display_from_frame", "before_state_source", "is_final_game"} <= points[0].keys()
    missing = []
    original_shots = [dict(s) for s in shots]
    for s in shots:
        assert s["hitter"] in PLAYERS and s["hitter_side"] in {"near", "far"}
        assert s["shot_outcome"] in {"continue", "error_net", "unknown"}
        assert isinstance(s["is_serve"], bool) and isinstance(s["hitter_won_point"], bool)
        assert sum(s[f"{p}_x_m"] is not None and s[f"{p}_y_m"] is not None for p in PLAYERS) == s["players_available"]
        image_path = (EVIDENCE_BASE / s["evidence"]).resolve() if s["evidence"] else None
        if image_path and image_path.is_file():
            relative = Path(os.path.relpath(image_path, OUTPUT_DIR)).as_posix()
            s["evidence_url"] = quote(relative, safe="/")
            assert (OUTPUT_DIR / unquote(s["evidence_url"])).resolve() == image_path
            assert (OUTPUT_DIR / unquote(s["evidence_url"])).is_file()
        else:
            s["evidence_url"] = None
            missing.append({"point": s["point"], "shot_index": s["shot_index"], "evidence": s["evidence"]})
    assert [{k: v for k, v in s.items() if k != "evidence_url"} for s in shots] == original_shots
    data = {"shots": shots, "points": points, "missing_images": missing}
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    embedded = embedded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = HTML.replace("__VIEWER_JSON__", embedded)
    assert "__VIEWER_JSON__" not in html
    assert len(html.encode("utf-8")) < 300_000
    html_path = OUTPUT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8", newline="\n")
    # Assert against the emitted artifact, not just the in-memory source.
    audit = HTMLAudit()
    audit.feed(html_path.read_text(encoding="utf-8"))
    audit.close()
    assert not audit.stack, audit.stack
    decoded = json.loads("".join(audit.json_parts))
    assert decoded == data
    assert len(decoded["shots"]) == 22 and len(decoded["points"]) == 66
    referenced = [s["evidence_url"] for s in decoded["shots"] if s["evidence_url"]]
    assert all(not url.startswith(("/", "http:", "https:", "data:")) for url in referenced)
    assert all((html_path.parent / unquote(url)).is_file() for url in referenced)
    assert len(referenced) + len(missing) == 22
    assert html_path.stat().st_size < 300_000
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    (OUTPUT_DIR / "README.md").write_text(README.format(image_count=len(shots), existing_count=len(referenced), missing_count=len(missing)), encoding="utf-8", newline="\n")
    print(f"PASS: shots={len(shots)}, points={len(points)}; embedded JSON round trip matches all rows")
    print(f"PASS: image references={len(referenced)}, unique={len(set(referenced))}, missing={len(missing)}")
    for item in missing:
        print(f"MISSING: point={item['point']} shot={item['shot_index']} {item['evidence']}")
    print("PASS: HTML nesting/IDs; relative image paths; input SHA256 unchanged")
    for name in ("build_viewer.py", "index.html", "README.md"):
        path = OUTPUT_DIR / name
        print(f"{name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
