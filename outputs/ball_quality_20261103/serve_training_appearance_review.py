"""Local-only object labels, training clips only. Never infer physical absence."""
import argparse, csv, datetime, json, os, re, secrets, tempfile, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from integrate_044_reviewed_tracks import ROOT, load, read, sha

OUT=ROOT/'ball_appearance_training_review'
LOCK=threading.Lock()
def validate(payload, ids):
    if not isinstance(payload,dict) or payload.get('case_id') not in ids: raise ValueError('対象が不正です。')
    if payload.get('object_label') not in ('ball','not_ball','uncertain'): raise ValueError('判定を選んでください。')
    notes=payload.get('notes','')
    if not isinstance(notes,str) or len(notes)>1000: raise ValueError('メモは1000文字以内です。')
    return dict(object_label=payload['object_label'],notes=notes,reviewer_id='human_user',
        reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

def atomic_write(path,rows):
    tmp=path.with_suffix('.pending.csv')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def self_test():
    for label in ('ball','not_ball','uncertain'):
        assert validate(dict(case_id='synthetic',object_label=label),{'synthetic'})['object_label']==label
    for p in [None,[],{},dict(case_id='bad',object_label='ball'),dict(case_id='synthetic',object_label=''),dict(case_id='synthetic',object_label='ball',notes=3)]:
        try:validate(p,{'synthetic'})
        except ValueError:pass
        else:raise AssertionError(p)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'synthetic.csv';r=dict(case_id='synthetic',object_label='uncertain',source_frame='42')
        atomic_write(p,[r]);assert read(p)==[r]
    print('Passed synthetic validation and atomic row preservation; no research labels written.')

def main(port):
    m=load(OUT/'manifest.json');cases=load(OUT/'cases.json');by_id={c['case_id']:c for c in cases}
    assert len(by_id)==9 and sha(OUT/'cases.json')==m['cases_sha256']
    assert sha(OUT/'existing_positive_centers.csv')==m['positive_table_sha256']
    for p,h in m['protected_input_hashes'].items(): assert sha(Path(p))==h,p
    for c in cases:
        for p,h in [('source_image','source_image_sha256'),('raw_patch96','patch_sha256'),('context_video','context_video_sha256')]:
            assert sha(Path(c[p]))==c[h],c['case_id']
    labels_path=OUT/'candidate_review_labels.csv'
    assert {r['sample_id'] for r in read(labels_path)}=={c['sample_id'] for c in cases}
    prefix='/'+secrets.token_urlsafe(18)+'/'
    def public():
        rows={r['case_id']:r for r in read(labels_path)}
        return [{**{k:c[k] for k in ['case_id','candidate_id','local_frame','source_frame','candidate_x_px','candidate_y_px']},
            **{k:rows[c['case_id']][k] for k in ['object_label','notes']}} for c in cases]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_bytes(self,b,kind,status=200,extra=None):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers();self.wfile.write(b)
        def json(self,x,status=200):self.send_bytes(json.dumps(x,ensure_ascii=False).encode(),'application/json; charset=utf-8',status)
        def do_GET(self):
            route=urlparse(self.path).path
            if not route.startswith(prefix):self.send_error(404);return
            route=route[len(prefix):]
            if route=='':self.send_bytes((ROOT/'training_appearance_review.html').read_bytes(),'text/html; charset=utf-8');return
            if route=='api/cases':
                with LOCK:self.json(dict(cases=public()))
                return
            match=re.fullmatch(r'(image|patch|video)/([A-I])\.(jpg|png|mp4)',route)
            if not match:self.send_error(404);return
            kind,cid,ext=match.groups();key,mime,expected={'image':('source_image','image/jpeg','jpg'),'patch':('raw_patch96','image/png','png'),'video':('context_video','video/mp4','mp4')}[kind]
            if ext!=expected:self.send_error(404);return
            blob=Path(by_id[cid][key]).read_bytes();length=len(blob)
            range_header=self.headers.get('Range','')
            if range_header and kind=='video':
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                if not match:self.send_error(416);return
                start=int(match[1]);end=min(length-1,int(match[2])) if match[2] else length-1
                if start>end or start>=length:self.send_error(416);return
                self.send_bytes(blob[start:end+1],mime,206,{'Content-Range':f'bytes {start}-{end}/{length}','Accept-Ranges':'bytes'})
            else:self.send_bytes(blob,mime,extra={'Accept-Ranges':'bytes'} if kind=='video' else None)
        def do_POST(self):
            if urlparse(self.path).path!=prefix+'api/save':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('入力サイズが不正です。')
                payload=json.loads(self.rfile.read(length));answer=validate(payload,set(by_id))
                # Same-origin browser writes only. Non-browser test clients omit Origin.
                origin=self.headers.get('Origin')
                if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':raise ValueError('不正な送信元です。')
                with LOCK:
                    rows=read(labels_path);row=next(r for r in rows if r['case_id']==payload['case_id']);before=dict(row);row.update(answer)
                    atomic_write(labels_path,rows)
                    with (OUT/'annotation_events.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(case_id=payload['case_id'],before=before,after=dict(row)),ensure_ascii=False)+'\n')
                    self.json(dict(cases=public()))
            except (ValueError,TypeError,KeyError,StopIteration) as e:self.json(dict(error=str(e)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    state=dict(url=f'http://127.0.0.1:{server.server_port}{prefix}',pid=os.getpid(),cases=9,bind='127.0.0.1',labels_path=str(labels_path),server_script_sha256=sha(Path(__file__)))
    (OUT/'server_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(state),flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main(a.port)
