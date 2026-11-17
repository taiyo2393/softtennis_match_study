"""Local-only blinded four-case annotation; never writes the old300 labels."""
import argparse,csv,datetime,json,math,os,re,secrets,threading,tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,sha

OUT=ROOT/'ball_inplay_challenge'
LOCK=threading.Lock()

def validate_center(payload,ids):
    if payload.get('case_id') not in ids:raise ValueError('対象の画像が不正です。')
    state=payload.get('truth_visibility')
    if state not in ('visible','not_visible','uncertain'):raise ValueError('球の状態を選んでください。')
    phase=payload.get('phase_status','unknown');confidence=payload.get('label_confidence','')
    if phase not in ('unknown','play','preparation_or_ended') or confidence not in ('','high','medium','low'):raise ValueError('入力の選択肢が不正です。')
    x=y=''
    if state=='visible':
        try:px=float(payload['x']);py=float(payload['y'])
        except (KeyError,TypeError,ValueError):raise ValueError('見える場合は球中心をクリックしてください。')
        if not(math.isfinite(px) and math.isfinite(py) and 0<=px<1920 and 0<=py<1080):raise ValueError('中心が元画像の範囲外です。')
        x=str(round(px));y=str(round(py))
        if int(x)>1919 or int(y)>1079:raise ValueError('中心が元画像の範囲外です。')
    notes=payload.get('notes','')
    if not isinstance(notes,str) or len(notes)>1000:raise ValueError('メモは1000文字以内で入力してください。')
    return dict(truth_visibility=state,truth_x_px=x,truth_y_px=y,phase_status=phase,label_confidence=confidence,notes=notes,
        reviewer_id='human_user',annotated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

def validate(payload,ids):
    phase=payload.get('phase_status','unknown')
    if phase not in ('play','preparation_or_ended','uncertain'):raise ValueError('対象フレームがプレー中かを選んでください。')
    skipped=phase!='play' and payload.get('truth_visibility')=='unreviewed'
    center=dict(payload,phase_status='unknown' if phase=='uncertain' else phase)
    if skipped:center['truth_visibility']='uncertain'
    result=validate_center(center,ids)
    result.update(phase_status=phase,phase_review_status='human_uncertain' if phase=='uncertain' else 'human_confirmed',
        ball_annotation_scope='skipped_nonplay_or_phase_uncertain' if skipped else 'human_ball_review')
    return result

def atomic_write(path,rows):
    temporary=path.with_suffix('.pending.csv')
    with temporary.open('w',encoding='utf-8-sig',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temporary.replace(path)

def self_test():
    # Synthetic temporary rows only; never save a fabricated review of a real image.
    good=dict(case_id='synthetic',truth_visibility='visible',x=1919,y=1079,phase_status='play')
    assert validate(good,{'synthetic'})['truth_x_px']=='1919'
    for changes in [dict(x=1920),dict(y=-1),dict(x=float('nan')),dict(case_id='other'),dict(phase_status='unknown')]:
        try:validate(dict(good,**changes),{'synthetic'})
        except ValueError:pass
        else:raise AssertionError(changes)
    invisible=validate(dict(good,truth_visibility='not_visible'),{'synthetic'})
    assert invisible['truth_x_px']==invisible['truth_y_px']==''
    with tempfile.TemporaryDirectory(prefix='ball_annotation_test_') as folder:
        p=Path(folder)/'labels.csv';row=dict(sample_id='synthetic',source_frame='123',prediction_x_px='999',**invisible)
        atomic_write(p,[row]);saved=read(p);assert saved[0]==row and saved[0]['prediction_x_px']=='999'
    # Pixel mapping including letterboxing, scrolling and2x display: the image center remains(960,540).
    for width,height in [(1280,720),(1920,1080),(2560,1440),(1000,1000),(2400,1350)]:
        scale=min(width/1920,height/1080);ox=(width-1920*scale)/2;oy=(height-1080*scale)/2
        assert abs(((width/2-ox)/scale)-960)<1e-8 and abs(((height/2-oy)/scale)-540)<1e-8
    print('Passed: bounds,nonfinite inputs,unknown IDs,absence-state coordinates,atomic preservation,pixel geometry; no research data written.')

def main(port):
    manifest=load(OUT/'annotation_manifest.json');cases=load(OUT/'annotation_cases.json')
    assert sha(OUT/'annotation_cases.json')==manifest['cases_sha256']
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    by_id={c['case_id']:c for c in cases};path=OUT/'manual_center_labels.csv'
    assert len(by_id)==4 and {r['sample_id'] for r in read(path)}=={c['sample_id'] for c in cases}
    for c in cases:assert sha(Path(c['source_image']))==c['source_image_sha256'] and sha(Path(c['context_video']))==c['context_video_sha256']
    prefix='/'+secrets.token_urlsafe(18)+'/'
    def public_cases():
        labels={r['sample_id']:r for r in read(path)};result=[]
        for c in cases:
            r=labels[c['sample_id']]
            result.append({**{k:c[k] for k in ['case_id','candidate_id','local_frame','source_frame']},
                **{k:r[k] for k in ['truth_visibility','truth_x_px','truth_y_px','phase_status','label_confidence','notes']}})
        return result
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_bytes(self,data,kind,status=200,extra=None):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers();self.wfile.write(data)
        def send_json(self,obj,status=200):self.send_bytes(json.dumps(obj,ensure_ascii=False).encode(),'application/json; charset=utf-8',status)
        def do_GET(self):
            parsed=urlparse(self.path)
            if not parsed.path.startswith(prefix):self.send_error(404);return
            route=parsed.path[len(prefix):]
            if route=='':self.send_bytes((OUT/'phase_challenge_annotation.html').read_bytes(),'text/html; charset=utf-8');return
            if route=='api/cases':
                with LOCK:self.send_json(dict(cases=public_cases()))
                return
            match=re.fullmatch(r'image/([A-D])/(\d+)\.jpg',route)
            if match:
                cid,t=match.group(1),int(match.group(2));c=by_id[cid]
                if abs(t-c['local_frame'])>30:self.send_error(404);return
                image=SOURCE/'data/balltrack/racketvision/match1'/c['candidate_id']/'frames'/f'{t:06d}.jpg'
                if not image.exists():self.send_error(404);return
                self.send_bytes(image.read_bytes(),'image/jpeg');return
            match=re.fullmatch(r'video/([A-D])\.mp4',route)
            if match:
                blob=Path(by_id[match.group(1)]['context_video']).read_bytes();length=len(blob);range_header=self.headers.get('Range','')
                if range_header:
                    request=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                    if not request:self.send_error(416);return
                    start=int(request.group(1));end=min(length-1,int(request.group(2))) if request.group(2) else length-1
                    if start>end or start>=length:self.send_error(416);return
                    self.send_bytes(blob[start:end+1],'video/mp4',206,{'Content-Range':f'bytes {start}-{end}/{length}','Accept-Ranges':'bytes'})
                else:self.send_bytes(blob,'video/mp4',extra={'Accept-Ranges':'bytes'})
                return
            self.send_error(404)
        def do_POST(self):
            if urlparse(self.path).path!=prefix+'api/save':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>16384:raise ValueError('入力サイズが不正です。')
                payload=json.loads(self.rfile.read(length));answer=validate(payload,set(by_id))
                with LOCK:
                    rows=read(path);c=by_id[payload['case_id']];row=next(r for r in rows if r['sample_id']==c['sample_id']);previous=dict(row);row.update(answer)
                    atomic_write(path,rows)
                    with (OUT/'annotation_events.jsonl').open('a',encoding='utf-8') as log:
                        log.write(json.dumps(dict(case_id=c['case_id'],sample_id=c['sample_id'],before=previous,after=dict(row),
                            model_prediction_hidden=True,context_video_available=True),ensure_ascii=False)+'\n')
                    self.send_json(dict(cases=public_cases()))
            except (ValueError,KeyError,TypeError,StopIteration,json.JSONDecodeError) as error:self.send_json(dict(error=str(error)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);url=f'http://127.0.0.1:{server.server_port}{prefix}'
    state=dict(status='running',url=url,pid=os.getpid(),bind='127.0.0.1',cases=4,labels_path=str(path.resolve()),
        original300_path=str(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv'),original300_written=False,
        case_manifest_sha256=sha(OUT/'annotation_cases.json'),server_script_sha256=sha(Path(__file__)))
    (OUT/'annotation_server_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps(state,indent=2),flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main(a.port)
