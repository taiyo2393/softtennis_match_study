import sys
sys.path.insert(0,'C:\\Users\\ta239\\AndroidStudioProjects\\tennis_ai_app2\\outputs\\ball_quality_20261103')
"""Local prediction-assisted usability review with explicit, bounded timing."""
import argparse,datetime,json,math,os,re,secrets,tempfile,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from integrate_044_reviewed_tracks import ROOT,read,load,sha
from serve_ball_phase_challenge import atomic_write

LOCK=threading.Lock()


def append(path,event):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


class Timing:
    def __init__(self,path,clock=time.monotonic):
        self.path,self.clock=path,clock;self.active=False;self.client=None;self.last=None
        old=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        self.total=sum(e['active_ms'] for e in old);self.gap=sum(e['gap_ms'] for e in old)

    def record(self,kind,client,active_ms=0,gap_ms=0,reason=''):
        self.total+=active_ms;self.gap+=gap_ms
        append(self.path,dict(kind=kind,client_id=client,active_ms=active_ms,gap_ms=gap_ms,reason=reason,
            server_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

    def capture(self,kind,client,reason=''):
        if not self.active or self.client!=client:raise ValueError('計測を開始・再開してください。')
        now=self.clock();elapsed=max(0,round((now-self.last)*1000));self.last=now
        self.record(kind,client,min(elapsed,15000),max(0,elapsed-15000),reason)
        if elapsed>15000:self.active=False

    def control(self,action,client,reason=''):
        if not isinstance(client,str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',client):raise ValueError('画面識別子が不正です。')
        if action=='start':
            if self.active and self.clock()-self.last>15:self.capture('expired',self.client,'start_after_timeout')
            if self.active and self.client!=client:
                if self.clock()-self.last<=15:raise ValueError('別の画面で計測中です。先にその画面を休止してください。')
                self.capture('expired',self.client,'another_client_after_timeout')
            if not self.active:
                self.client=client;self.last=self.clock();self.active=True;self.record('start',client,reason=reason)
        elif action in {'tick','pause'}:
            if self.active and self.client==client:
                self.capture(action,client,reason)
                if action=='pause':self.active=False
        else:raise ValueError('計測操作が不正です。')

    def public(self,client=None):
        return dict(active=self.active and (client is None or self.client==client),captured_ms=self.total,gap_ms=self.gap)


def validate_answer(payload,cases):
    allowed={'case_id','decision','x','y','notes','revision','client_id'}
    if not isinstance(payload,dict) or not set(payload)<=allowed:raise ValueError('保存項目が不正です。')
    cid=payload.get('case_id');c=cases.get(cid)
    if c is None:raise ValueError('対象が不正です。')
    if c['reused_manual_center']:raise ValueError('この中心は以前の手動入力を保持します。次の未確認へ進んでください。')
    decision=payload.get('decision');x=y='';source=''
    if decision=='accept_prediction':
        if not c['prediction_visible']:raise ValueError('この時点には採用する予測がありません。')
        x,y=str(c['prediction_x_px']),str(c['prediction_y_px']);source='prediction_explicitly_accepted_by_human'
    elif decision=='correct_center':
        try:x,y=float(payload['x']),float(payload['y'])
        except (ValueError,TypeError,KeyError):raise ValueError('修正する球中心をクリックしてください。')
        if not (math.isfinite(x) and math.isfinite(y) and 0<=x<1920 and 0<=y<1080):raise ValueError('中心が画像の範囲外です。')
        x,y=str(round(x)),str(round(y))
        if int(x)>1919 or int(y)>1079:raise ValueError('中心が画像の範囲外です。')
        source='human_click_with_prediction_visible'
    elif decision not in {'not_visible','uncertain','nonplay'}:raise ValueError('採用、修正、見えない、判断保留、区間外のいずれかを選んでください。')
    notes=payload.get('notes','')
    if not isinstance(notes,str) or len(notes)>1000:raise ValueError('メモは1000文字以内にしてください。')
    return dict(review_status='human_assisted_review',decision=decision,x_px=x,y_px=y,coordinate_source=source,
        independent_new_center_gt=0,reviewer_id='human_user',reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),notes=notes)


def verify_history(folder):
    initial=read(folder/'initial_labels_snapshot.csv');state={r['case_id']:r for r in initial}
    if (folder/'review_events.jsonl').exists():
        for s in (folder/'review_events.jsonl').read_text(encoding='utf-8').splitlines():
            e=json.loads(s);assert state[e['case_id']]==e['before'];state[e['case_id']]=e['after']
    assert [state[r['case_id']] for r in initial]==read(folder/'review_labels.csv')


def main(folder,port):
    manifest=load(folder/'manifest.json')
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    assert all(sha(folder/n)==h for n,h in manifest['frozen_artifact_hashes'].items())
    cases={c['case_id']:c for c in load(folder/'cases.json')};assert len(cases)==201
    verify_history(folder);labels_path=folder/'review_labels.csv'
    timing=Timing(folder/'timing_events.jsonl');prefix='/'+secrets.token_urlsafe(18)+'/'
    def public(client=None):
        labels={r['case_id']:r for r in read(labels_path)}
        return dict(cases=[dict(c,**{k:labels[cid][k] for k in ['review_status','decision','x_px','y_px','coordinate_source','notes']}) for cid,c in cases.items()],
            revision=sha(labels_path),timing=timing.public(client))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def blob(self,b,mime,status=200,headers=None):
            self.send_response(status)
            for k,v in {'Content-Type':mime,'Content-Length':str(len(b)),'Cache-Control':'no-store',**(headers or {})}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(b)
        def json(self,value,status=200):self.blob(json.dumps(value,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8',status)
        def do_GET(self):
            route=urlparse(self.path).path
            if not route.startswith(prefix):self.send_error(404);return
            route=route[len(prefix):]
            if route=='':self.blob((folder/'review.html').read_bytes(),'text/html; charset=utf-8');return
            if route=='api/cases':
                with LOCK:self.json(public())
                return
            m=re.fullmatch(r'image/(R\d{3})\.jpg',route)
            if m and m[1] in cases:self.blob(Path(cases[m[1]]['source_image']).read_bytes(),'image/jpeg');return
            if route=='video/context.mp4':
                b=(folder/'context.mp4').read_bytes();rh=self.headers.get('Range')
                if not rh:self.blob(b,'video/mp4',headers={'Accept-Ranges':'bytes'});return
                m=re.fullmatch(r'bytes=(\d*)-(\d*)',rh)
                if not m or not any(m.groups()):self.send_error(416);return
                lo=max(0,len(b)-int(m[2])) if not m[1] else int(m[1]);hi=len(b)-1 if not m[1] or not m[2] else min(int(m[2]),len(b)-1)
                if lo>hi or lo>=len(b):self.send_error(416);return
                self.blob(b[lo:hi+1],'video/mp4',206,{'Content-Range':f'bytes {lo}-{hi}/{len(b)}','Accept-Ranges':'bytes'});return
            self.send_error(404)
        def do_POST(self):
            route=urlparse(self.path).path
            if route not in {prefix+'api/save',prefix+'api/timing'}:self.send_error(404);return
            payload=None
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('入力サイズが不正です。')
                if self.headers.get('Origin') not in {None,f'http://127.0.0.1:{self.server.server_port}'}:raise ValueError('送信元が不正です。')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('入力形式が不正です。')
                client=payload.get('client_id')
                with LOCK:
                    if route.endswith('/timing'):
                        if not set(payload)<={'client_id','action','reason'}:raise ValueError('計測項目が不正です。')
                        timing.control(payload.get('action'),client,str(payload.get('reason',''))[:100]);self.json(dict(timing=timing.public(client)));return
                    if payload.get('revision')!=sha(labels_path):self.json(dict(error='別の画面で更新されています。再読込してください。'),409);return
                    answer=validate_answer(payload,cases)
                    if not timing.active or timing.client!=client:raise ValueError('計測を開始・再開してください。')
                    timing.capture('save',client)
                    if not timing.active:raise ValueError('通信または操作の間隔が空いたため計測を休止しました。再開して保存してください。')
                    rows=read(labels_path);row=next(r for r in rows if r['case_id']==payload['case_id']);before=dict(row);row.update(answer)
                    # CSV readback uses strings; journal stores the same representation.
                    row={k:str(v) for k,v in row.items()}
                    rows=[row if r['case_id']==row['case_id'] else r for r in rows]
                    append(folder/'review_events.jsonl',dict(case_id=row['case_id'],before=before,after=row,predictions_visible=True))
                    atomic_write(labels_path,rows)
                    if all(r['review_status']!='unreviewed' for r in rows):timing.control('pause',client,'all_targets_completed')
                    self.json(public(client))
            except (ValueError,KeyError,TypeError,StopIteration) as e:self.json(dict(error=str(e),timing=timing.public(payload.get('client_id') if isinstance(payload,dict) else None)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    state=dict(url=f'http://127.0.0.1:{server.server_port}{prefix}',pid=os.getpid(),bind='127.0.0.1',targets=201,pending=195,server_script_sha256=sha(Path(__file__)))
    (folder/'server_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(state),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


def self_test():
    cases={'synthetic':dict(reused_manual_center=False,prediction_visible=1,prediction_x_px=123.,prediction_y_px=456.),'existing':dict(reused_manual_center=True)}
    a=validate_answer(dict(case_id='synthetic',decision='accept_prediction'),cases);assert a['x_px']=='123.0' and a['independent_new_center_gt']==0
    for bad in [dict(case_id='missing',decision='accept_prediction'),dict(case_id='existing',decision='correct_center',x=2,y=3),dict(case_id='synthetic',decision='correct_center',x=float('nan'),y=3),dict(case_id='synthetic',decision='correct_center',x=1920,y=3)]:
        try:validate_answer(bad,cases)
        except ValueError:pass
        else:raise AssertionError('Invalid synthetic answer accepted')
    assert validate_answer(dict(case_id='synthetic',decision='not_visible'),cases)['x_px']==''
    with tempfile.TemporaryDirectory(prefix='repair_timing_synthetic_') as name:
        now=[0.];timer=Timing(Path(name)/'timing.jsonl',lambda:now[0]);timer.control('start','synthetic_client')
        now[0]=2.5;timer.control('tick','synthetic_client');now[0]=7.;timer.control('pause','synthetic_client')
        assert timer.total==7000 and not timer.active
        timer.control('start','synthetic_client');now[0]=30.;timer.control('tick','synthetic_client')
        assert timer.total==22000 and timer.gap==8000 and not timer.active
        recovered=Timing(Path(name)/'timing.jsonl',lambda:now[0]);assert recovered.total==22000 and not recovered.active
    print('Passed synthetic acceptance,source distinction,range/reuse rejection,timer start/pause/gap cap/restart;no real answers or timing written.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=0);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:main(Path(__file__).resolve().parent,args.port)
