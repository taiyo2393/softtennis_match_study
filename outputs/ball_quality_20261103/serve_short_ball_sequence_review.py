"""Prediction-blind local annotation of24 fixed sequence times, no fitting."""
import json,os,re,secrets,threading,datetime,argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from integrate_044_reviewed_tracks import ROOT,load,read,sha
from serve_ball_phase_challenge import validate,atomic_write,self_test

OUT=ROOT/'ball_short_sequence_evaluation';LOCK=threading.Lock()
def answer(payload,cases):
    if not isinstance(payload,dict) or payload.get('case_id') not in cases:raise ValueError('対象が不正です。')
    c=cases[payload['case_id']]
    if c['center_reused']:payload=dict(payload,truth_visibility='visible',x=c['reused_x'],y=c['reused_y'])
    result=validate(payload,set(cases))
    return {k:result[k] for k in ['truth_visibility','truth_x_px','truth_y_px','phase_status','reviewer_id','annotated_at_utc','notes']}

def main(port):
    m=load(OUT/'manifest.json');cases=load(OUT/'annotation_cases.json');by_id={c['case_id']:c for c in cases};labels_path=OUT/'manual_sequence_labels.csv'
    assert len(by_id)==24 and sha(OUT/'annotation_cases.json')==m['cases_sha256'] and sha(OUT/'frozen_predictions.csv')==m['predictions_sha256']
    assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
    assert len(read(labels_path))==26
    for c in cases:assert sha(Path(c['source_image']))==c['source_image_sha256'] and sha(Path(c['context_video']))==c['context_video_sha256']
    prefix='/'+secrets.token_urlsafe(18)+'/'
    def public():
        labels={r['case_id']:r for r in read(labels_path)}
        return [{**{k:c[k] for k in ['case_id','window_id','candidate_id','local_frame','source_frame','center_reused']},
            **{k:labels[c['case_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px','phase_status','review_status','notes']}} for c in cases]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_bytes(self,b,kind,status=200,extra=None):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers();self.wfile.write(b)
        def json(self,v,status=200):self.send_bytes(json.dumps(v,ensure_ascii=False).encode(),'application/json; charset=utf-8',status)
        def do_GET(self):
            route=urlparse(self.path).path
            if not route.startswith(prefix):self.send_error(404);return
            route=route[len(prefix):]
            if route=='':self.send_bytes((OUT/'annotation.html').read_bytes(),'text/html; charset=utf-8');return
            if route=='api/cases':
                with LOCK:self.json(dict(cases=public()))
                return
            match=re.fullmatch(r'(image|video)/([AB]\d{2})\.(jpg|mp4)',route)
            if not match or match[2] not in by_id:self.send_error(404);return
            kind,cid,ext=match.groups()
            if (kind,ext) not in [('image','jpg'),('video','mp4')]:self.send_error(404);return
            blob=Path(by_id[cid]['source_image' if kind=='image' else 'context_video']).read_bytes();mime='image/jpeg' if kind=='image' else 'video/mp4'
            rh=self.headers.get('Range','')
            if rh and kind=='video':
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',rh)
                if not match:self.send_error(416);return
                start=int(match[1]);end=min(len(blob)-1,int(match[2])) if match[2] else len(blob)-1
                if start>end or start>=len(blob):self.send_error(416);return
                self.send_bytes(blob[start:end+1],mime,206,{'Content-Range':f'bytes {start}-{end}/{len(blob)}','Accept-Ranges':'bytes'})
            else:self.send_bytes(blob,mime,extra={'Accept-Ranges':'bytes'} if kind=='video' else None)
        def do_POST(self):
            if urlparse(self.path).path!=prefix+'api/save':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('入力サイズが不正です。')
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}'):raise ValueError('不正な送信元です。')
                payload=json.loads(self.rfile.read(length));review=answer(payload,by_id);cid=payload['case_id'];c=by_id[cid]
                with LOCK:
                    rows=read(labels_path);row=next(r for r in rows if r['case_id']==cid);before=dict(row);row.update(review)
                    row['review_status']='human_phase_only_existing_center' if c['center_reused'] else 'human_frame_review'
                    if not c['center_reused']:row['center_source']='new_human_frame_review'
                    atomic_write(labels_path,rows)
                    with (OUT/'annotation_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(case_id=cid,before=before,after=dict(row)),ensure_ascii=False)+'\n')
                    self.json(dict(cases=public()))
            except (ValueError,TypeError,KeyError,StopIteration) as e:self.json(dict(error=str(e)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);state=dict(url=f'http://127.0.0.1:{server.server_port}{prefix}',pid=os.getpid(),annotation_cases=24,bind='127.0.0.1',server_script_sha256=sha(Path(__file__)))
    (OUT/'server_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(state),flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        self_test()
        synthetic={'synthetic':dict(center_reused=True,reused_x='321',reused_y='456')}
        r=answer(dict(case_id='synthetic',phase_status='uncertain',truth_visibility='not_visible',x=99,y=99),synthetic)
        assert r['truth_x_px']=='321' and r['truth_y_px']=='456' and r['phase_status']=='uncertain'
        print('Passed reused-center preservation for phase-only review; synthetic data only.')
    else:main(a.port)
