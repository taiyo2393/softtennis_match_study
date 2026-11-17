"""Local-only phase review of all123 frozen training times; no coordinate edits."""
import argparse,datetime,json,math,os,re,secrets,threading,copy,tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

OUT=ROOT.parent/'ball_quality_20261109/training_phase_scope_review'
LOCK=threading.Lock()
PUBLIC_FIELDS=['sample_id','clip_id','candidate_id','local_frame','source_frame','clip_time_sec','phase_status','phase_detail','review_status','notes','locked']

class Conflict(ValueError):pass

def validate_batch(payload,rows,clips,revision):
    if not isinstance(payload,dict) or set(payload)!={'clip_id','revision','annotations','notes','helper_range'}:raise ValueError('保存項目が不正です。')
    if payload['revision']!=revision:raise Conflict('別の画面で更新されています。最新状態を読み直して再確認してください。')
    cid=payload['clip_id']
    if cid not in clips:raise ValueError('動画が不正です。')
    group=[r for r in rows if r['clip_id']==cid];ids={r['sample_id'] for r in group};answers=payload['annotations']
    if not isinstance(answers,list) or len(answers)!=len(ids):raise ValueError('この動画の全時点を確認してください。')
    if any(not isinstance(a,dict) or set(a)!={'sample_id','phase_status'} for a in answers):raise ValueError('区分の項目が不正です。')
    if len({a['sample_id'] for a in answers})!=len(ids) or {a['sample_id'] for a in answers}!=ids:raise ValueError('対象時点の重複または不足があります。')
    proposed={a['sample_id']:a['phase_status'] for a in answers}
    if any(v not in ('play','nonplay','uncertain') for v in proposed.values()):raise ValueError('未入力の時点があります。分からない場合は判断できないを選んでください。')
    if not isinstance(payload['notes'],str) or len(payload['notes'])>1000:raise ValueError('メモは1000文字以内です。')
    helper=payload['helper_range']
    if helper is not None:
        if not isinstance(helper,dict) or set(helper)!={'start_sec','end_sec'}:raise ValueError('補助範囲が不正です。')
        a,b=helper['start_sec'],helper['end_sec']
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [a,b]) or not 0<=a<=b<=clips[cid]['duration_sec']:raise ValueError('補助範囲が動画外です。')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();updated=[dict(r) for r in rows];before=[];after=[]
    for r in updated:
        if r['clip_id']!=cid:continue
        if int(r['locked']):
            if proposed[r['sample_id']]!=r['phase_status']:raise ValueError('以前確認した終了後の回答は、この画面では変更できません。')
            continue
        before.append(dict(r));r.update(phase_status=proposed[r['sample_id']],phase_detail='',review_status='human_phase_review',
            reviewer_id='human_user',reviewed_at_utc=now,notes=payload['notes'],evidence_origin='human_phase_scope_ui')
        after.append(dict(r))
    event=dict(clip_id=cid,before=before,after=after,helper_range=helper,
        range_scope='Input aid only; not exact rally boundaries',reviewed_at_utc=now)
    return updated,event

def persist(path,events,rows,event):
    # The event is durable first; a mismatch is detected on restart, never hidden.
    with events.open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    temporary=path.with_suffix('.pending.csv');write(temporary,rows);temporary.replace(path)

def verify_history(folder):
    initial=read(folder/'initial_phase_labels.csv');state={r['sample_id']:dict(r) for r in initial}
    events=folder/'annotation_events.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            if not line:continue
            event=json.loads(line)
            for a,b in zip(event['before'],event['after'],strict=True):
                assert state[a['sample_id']]==a and a['sample_id']==b['sample_id']
                assert not int(a['locked'])
                state[a['sample_id']]=b
    assert read(folder/'phase_labels.csv')==[state[r['sample_id']] for r in initial],'Phase CSV/history mismatch; preserve both and inspect.'

def self_test():
    rows=[dict(sample_id='synthetic_1',clip_id='synthetic',phase_status='nonplay',phase_detail='ended',review_status='prior_human_confirmed',locked='1',reviewer_id='human_user',reviewed_at_utc='fixture',notes='prior',evidence_origin='fixture'),
          dict(sample_id='synthetic_2',clip_id='synthetic',phase_status='unreviewed',phase_detail='',review_status='unreviewed',locked='0',reviewer_id='',reviewed_at_utc='',notes='',evidence_origin='')]
    clips={'synthetic':dict(duration_sec=10.)};good=dict(clip_id='synthetic',revision='fixture',annotations=[dict(sample_id='synthetic_1',phase_status='nonplay'),dict(sample_id='synthetic_2',phase_status='play')],notes='',helper_range=dict(start_sec=0.,end_sec=5.))
    output,event=validate_batch(good,rows,clips,'fixture');assert output[0]==rows[0] and output[1]['phase_status']=='play' and rows[1]['phase_status']=='unreviewed'
    invalid=[]
    q=copy.deepcopy(good);q['annotations'][1]['phase_status']='unreviewed';invalid.append(q)
    q=copy.deepcopy(good);q['annotations'][0]['phase_status']='play';invalid.append(q)
    q=copy.deepcopy(good);q['annotations'].pop();invalid.append(q)
    q=copy.deepcopy(good);q['annotations'][1]['sample_id']='synthetic_1';invalid.append(q)
    q=copy.deepcopy(good);q['helper_range']['end_sec']=float('nan');invalid.append(q)
    q=copy.deepcopy(good);q['revision']='stale';invalid.append(q)
    q=copy.deepcopy(good);q['truth_x_px']=123;invalid.append(q)
    for q in invalid:
        try:validate_batch(q,rows,clips,'fixture')
        except (ValueError,TypeError):pass
        else:raise AssertionError('Invalid synthetic update accepted')
    with tempfile.TemporaryDirectory(prefix='phase_scope_synthetic_') as directory:
        folder=Path(directory);write(folder/'initial_phase_labels.csv',rows);write(folder/'phase_labels.csv',rows)
        persist(folder/'phase_labels.csv',folder/'annotation_events.jsonl',output,event);verify_history(folder)
        assert read(folder/'phase_labels.csv')[0]==rows[0]
    print('Passed synthetic save/replay,locked answer preservation,unknown/missing/duplicate targets,invalid bounds,stale revisions and coordinate-field rejection. No research labels written.')

def main(port):
    m=load(OUT/'manifest.json');clips={c['clip_id']:c for c in load(OUT/'clips.json')};assets=load(OUT/'image_assets.json')
    assert len(clips)==8 and len(assets)==123
    for table in [m['protected_input_hashes'],m['immutable_artifact_hashes']]:assert all(sha(Path(p))==h for p,h in table.items())
    for f in m['frozen_methods']:assert sha(Path(f['path']))==f['sha256']
    verify_history(OUT);prefix='/'+secrets.token_urlsafe(18)+'/'
    def public():
        rows=read(OUT/'phase_labels.csv')
        return dict(clips=[{k:c[k] for k in ['clip_id','candidate_id','fps','frame_count','duration_sec','source_frame_offset']} for c in clips.values()],
            samples=[{k:r[k] for k in PUBLIC_FIELDS} for r in rows],revision=sha(OUT/'phase_labels.csv'))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_blob(self,blob,mime,status=200):
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(blob)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(blob)
        def json(self,value,status=200):self.send_blob(json.dumps(value,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8',status)
        def file(self,path,mime):
            total=path.stat().st_size;start,end=0,total-1;partial=False
            if self.headers.get('Range'):
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
                if not match or not any(match.groups()):self.send_error(416);return
                a,b=match.groups()
                if a:start=int(a);end=min(total-1,int(b)) if b else total-1
                else:start=max(0,total-int(b))
                if start>end or start>=total:self.send_error(416);return
                partial=True
            self.send_response(206 if partial else 200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
            self.end_headers()
            try:
                with path.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        data=f.read(min(1048576,remaining))
                        if not data:break
                        self.wfile.write(data);remaining-=len(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            route=urlparse(self.path).path
            if not route.startswith(prefix):self.send_error(404);return
            route=route[len(prefix):]
            if route=='':self.send_blob((OUT/'review.html').read_bytes(),'text/html; charset=utf-8');return
            if route=='api/data':
                with LOCK:self.json(public())
                return
            match=re.fullmatch(r'video/(\d{3})\.mp4',route)
            if match and match[1] in clips:self.file(Path(clips[match[1]]['video_path']),'video/mp4');return
            match=re.fullmatch(r'image/([A-Za-z0-9_]+)\.jpg',route)
            if match and match[1] in assets:self.file(Path(assets[match[1]]['path']),'image/jpeg');return
            self.send_error(404)
        def do_POST(self):
            if urlparse(self.path).path!=prefix+'api/save':self.send_error(404);return
            try:
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}'):raise ValueError('送信元が不正です。')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768:raise ValueError('入力サイズが不正です。')
                payload=json.loads(self.rfile.read(length))
                with LOCK:
                    path=OUT/'phase_labels.csv';rows=read(path);updated,event=validate_batch(payload,rows,clips,sha(path))
                    persist(path,OUT/'annotation_events.jsonl',updated,event);self.json(public())
            except Conflict as e:self.json(dict(error=str(e)),409)
            except (ValueError,TypeError,KeyError) as e:self.json(dict(error=str(e)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    state=dict(url=f'http://127.0.0.1:{server.server_port}{prefix}',pid=os.getpid(),server_script_sha256=sha(Path(__file__)),clips=8,targets=123)
    (OUT/'server_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(state),flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test()
    else:main(args.port)
