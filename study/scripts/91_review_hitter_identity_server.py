"""Review physical hitter identity without altering prior annotations or thresholds."""
import argparse
import csv
import html
import json
import secrets
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv'
PREVIEWS = BASE / 'data/pose/target_front/previews'
FIELDS = ['event_id', 'target_front_player_id', 'impact_source_frame', 'target_front_camera_end',
          'hitter_id', 'hitter_description', 'review_status', 'note']

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def atomic_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

class Store:
    def __init__(self, output, summary):
        self.output, self.summary = output, summary
        self.lock = threading.RLock()
        self.features = read_csv(SOURCE)
        self.events = {}
        seen = set()
        for r in self.features:
            key = (r['event_id'], r['timepoint_sec'])
            if key in seen: raise ValueError('Duplicate event/timepoint')
            seen.add(key)
            e = {k:r[k] for k in FIELDS[:4]}
            if r['event_id'] in self.events and e != self.events[r['event_id']]:
                raise ValueError('Inconsistent event identity')
            self.events[r['event_id']] = e
        self.order = sorted(self.events, key=lambda k:int(self.events[k]['impact_source_frame']))
        self.saved = {r['event_id']:r for r in read_csv(output)} if output.exists() else {}
        if not set(self.saved) <= set(self.events): raise ValueError('Unknown saved event')

    def rows(self):
        return [{**self.events[k], 'hitter_id':'', 'hitter_description':'',
                 'review_status':'unreviewed', 'note':'', **self.saved.get(k,{})} for k in self.order]

    def save(self, event, hitter, description, status, note):
        if event not in self.events or status not in ('confirmed','uncertain'):
            raise ValueError('Invalid event or status')
        if status == 'confirmed' and (hitter not in ('H1','H2','H3','H4') or not description.strip()):
            raise ValueError('打者IDと、名前または人物を見分ける説明を入力してください。')
        with self.lock:
            if status == 'confirmed':
                descriptions = {r['hitter_description'] for r in self.saved.values()
                                if r['review_status']=='confirmed' and r['hitter_id']==hitter and r['event_id']!=event}
                if descriptions and descriptions != {description.strip()}:
                    raise ValueError('同じH番号の説明は統一してください。別人物なら別のH番号を選んでください。')
            self.saved[event] = {**self.events[event], 'hitter_id':hitter if status=='confirmed' else '',
                                 'hitter_description':description.strip() if status=='confirmed' else '',
                                 'review_status':status, 'note':note.strip()}
            atomic_csv(self.output,FIELDS,self.rows())
            self.aggregate()

    def aggregate(self):
        groups=defaultdict(list)
        for f in self.features:
            r=self.saved.get(f['event_id'],{})
            if r.get('review_status')=='confirmed':
                groups[(f['target_front_player_id'],r['hitter_id'],f['timepoint_sec'])].append(f)
        fields=['front_id','hitter_id','timepoint_sec','reviewed_events','directional_events','matches','agreement_percent','identity_review_complete']
        rows=[]
        for (front,hitter,t),g in sorted(groups.items()):
            n=sum(r['movement_is_directional']=='1' for r in g)
            m=sum(r['movement_is_directional']=='1' and r['direction_matches_actual']=='1' for r in g)
            rows.append(dict(zip(fields,[front,hitter,t,len(g),n,m,round(100*m/n,4) if n else '',
                                        int(len(self.saved)==len(self.events) and all(r['review_status']=='confirmed' for r in self.saved.values()))])))
        atomic_csv(self.summary,fields,rows)

def handler(store):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def document(self,status,body):
            data=body.encode('utf-8');self.send_response(status)
            self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            u=urlparse(self.path);q=parse_qs(u.query)
            if u.path.startswith('/video/'):
                key=u.path[len('/video/'):]
                if key not in store.events:return self.send_error(404)
                path=PREVIEWS/(key+'.mp4')
                if not path.is_file():return self.send_error(404)
                size=path.stat().st_size;start,end=0,size-1;raw=self.headers.get('Range')
                if raw:
                    try:
                        unit,part=raw.split('='); a,b=part.split('-');assert unit=='bytes'
                        if a:start=int(a);end=min(int(b),end) if b else end
                        else:start=max(0,size-int(b))
                        assert 0<=start<=end<size
                    except (ValueError,AssertionError):return self.send_error(416)
                self.send_response(206 if raw else 200);self.send_header('Content-Type','video/mp4')
                self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if raw:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                with path.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    try:
                        while remaining:
                            data=f.read(min(65536,remaining));self.wfile.write(data);remaining-=len(data)
                    except (BrokenPipeError,ConnectionResetError):pass
                return
            if u.path!='/':return self.send_error(404)
            with store.lock: rows=store.rows()
            try:n=max(1,min(len(rows),int(q.get('page',['1'])[0])))
            except ValueError:n=1
            r=rows[n-1];esc=html.escape
            confirmed=sum(v['review_status']=='confirmed' for v in rows)
            uncertain=sum(v['review_status']=='uncertain' for v in rows)
            pending=next((i+1 for i,v in enumerate(rows) if v['review_status']=='unreviewed'),n)
            roster={v['hitter_id']:v['hitter_description'] for v in rows if v['review_status']=='confirmed'}
            opts='<option value="">選択</option>'+''.join(f'<option value="H{i}" {"selected" if r["hitter_id"]==f"H{i}" else ""}>H{i} {esc(roster.get(f"H{i}","未登録"))}</option>' for i in range(1,5))
            target='手前' if r['target_front_camera_end']=='near' else '奥'
            hitter_end='奥' if target=='手前' else '手前'
            body=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>打者本人の確認</title>
<style>body{{font-family:system-ui;max-width:1250px;margin:12px auto;padding:12px}}video{{width:100%;max-height:65vh;background:#111}}button,select,input{{font-size:18px;padding:7px;margin:5px}}p{{line-height:1.6}}.warn{{background:#fff3cc;padding:10px}}</style>
<h1>打者だけを確認 {n}/{len(rows)}</h1><p>確定 {confirmed} ／ 不明 {uncertain} ／ 未確認 {len(rows)-confirmed-uncertain}</p>
<p><a href="/?page={max(1,n-1)}">前へ</a>　<a href="/?page={min(len(rows),n+1)}">次へ</a>　<a href="/?page={pending}">最初の未確認へ</a>　{esc(q.get('message',[''])[0])}</p>
<p>対象前衛：{esc(r['target_front_player_id'])}（{target}側）／ 確認する打者：反対の<strong>{hitter_end}側</strong> ／ 原動画 F{r['impact_source_frame']}</p>
<video id="v" controls preload="metadata" src="/video/{esc(r['event_id'])}"></video>
<p><button onclick="v.currentTime=Math.max(0,v.currentTime-1)">1秒戻る</button><button onclick="v.paused?v.play():v.pause()">再生・停止</button><button onclick="v.requestFullscreen()">動画を全画面</button></p>
<p class="warn">黄色枠の前衛ではなく、相手側でボールを打った本人を選びます。H番号は初登場順など自由ですが、同じ人物には試合全体で同じ番号を使用してください。前衛P番号とは別の番号体系です。見えない場合は「不明」で保存してください。</p>
<form id="f" method="post" action="/save"><input type="hidden" name="token" value="{token}"><input type="hidden" name="event" value="{esc(r['event_id'])}"><input type="hidden" name="page" value="{n}">
<label>打者 <select id="h" name="hitter">{opts}</select></label><br><label>名前・人物の説明 <input id="d" name="description" size="45" value="{esc(r['hitter_description'],quote=True)}"></label><br>
<label><input type="checkbox" name="uncertain" id="u" {'checked' if r['review_status']=='uncertain' else ''}>打者本人が不明</label><br>
<label>メモ <input name="note" size="50" value="{esc(r['note'],quote=True)}"></label><button>保存して次へ</button></form>
<p>1〜4：H番号選択 ／ X：不明 ／ Enter：保存して次へ（文字入力中を除く）。各時点の一致率は確認済み場面のみで別ファイルへ集計します。未完了の集計を最終結果として使わないでください。</p>
<script>const roster={json.dumps(roster,ensure_ascii=True)},h=document.getElementById('h'),d=document.getElementById('d'),u=document.getElementById('u'),v=document.getElementById('v');
function change(){{d.value=roster[h.value]||'';u.checked=false;}}h.onchange=change;
document.addEventListener('keydown',e=>{{if(['INPUT','SELECT','TEXTAREA','BUTTON','VIDEO'].includes(document.activeElement.tagName))return;
if(['1','2','3','4'].includes(e.key)){{h.value='H'+e.key;change();e.preventDefault();}}if(e.key.toLowerCase()==='x'){{u.checked=true;e.preventDefault();}}if(e.key==='Enter'){{e.preventDefault();document.getElementById('f').requestSubmit();}}}});</script></html>'''
            self.document(200,body)
        def do_POST(self):
            if self.path!='/save':return self.send_error(404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<20000:return self.send_error(400)
                q=parse_qs(self.rfile.read(length).decode(),keep_blank_values=True)
                get=lambda k:q.get(k,[''])[0]
                if get('token')!=token:return self.send_error(403)
                store.save(get('event'),get('hitter'),get('description'),'uncertain' if 'uncertain' in q else 'confirmed',get('note'))
                n=min(len(store.order),int(get('page'))+1)
            except (ValueError,KeyError) as e:return self.document(400,html.escape(str(e))+'<p>ブラウザで戻って修正してください。</p>')
            self.send_response(303);self.send_header('Location','/?'+urlencode({'page':n,'message':'保存しました'}));self.end_headers()
    return Handler

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8805)
    args=p.parse_args()
    store=Store(BASE/'annotations/match1_hitter_identity_review.csv',BASE/'analysis/match1_front_hitter_agreement_reviewed_only.csv')
    print(f'http://127.0.0.1:{args.port}/ | {len(store.order)} events',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(store)).serve_forever()
