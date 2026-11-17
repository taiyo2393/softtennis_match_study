"""Loopback manual-reference UI. The public API never exposes model outputs."""
import argparse,json,secrets,threading,datetime,os
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse
BASE=Path(__file__).resolve().parent.parent/'ball_quality_20261110'
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

PILOT=BASE/'identity_position_bundle/reference_pilot'
SLOTS=['near_left','near_right','far_left','far_right']
STATES={'pending','point','unplaceable','absent','uncertain'}

def validate_entry(entry):
    state=entry.get('state');assert state in STATES
    if state=='point':
        x,y=float(entry['x']),float(entry['y']);assert 0<=x<3840 and 0<=y<2160
        return dict(state=state,x=x,y=y)
    return dict(state=state,x=None,y=None)

def make_server(pilot,token,label_origin='manual_reference_ui'):
    public=json.loads((pilot/'public_manifest.json').read_text(encoding='utf-8'));ids={r['image_id'] for r in public['items']};lock=threading.Lock();labels_path=pilot/'human_reference_labels.json'
    def labels():return json.loads(labels_path.read_text(encoding='utf-8')) if labels_path.exists() else dict(schema_version=1,label_origin=label_origin,revision=0,labels={})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_bytes(self,data,mime,status=200):
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def send_json(self,obj,status=200):self.send_bytes(json.dumps(obj,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8',status)
        def route(self):
            path=urlparse(self.path).path;prefix='/'+token+'/'
            return path[len(prefix):] if path.startswith(prefix) else None
        def do_GET(self):
            route=self.route()
            if route is None:self.send_json(dict(error='not_found'),404)
            elif route=='':self.send_bytes((pilot/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif route=='api/session':
                with lock:self.send_json(dict(manifest=public,annotations=labels()))
            elif route.startswith('images/') and route[7:].endswith('.png') and route[7:-4] in ids:self.send_bytes((pilot/'images'/route[7:]).read_bytes(),'image/png')
            else:self.send_json(dict(error='not_found'),404)
        def do_POST(self):
            if self.route()!='api/label':self.send_json(dict(error='not_found'),404);return
            try:
                n=int(self.headers.get('Content-Length','0'));assert 0<n<=8192;request=json.loads(self.rfile.read(n));image=request['image_id'];slot=request['slot'];assert image in ids and slot in SLOTS;entry=validate_entry(request['entry'])
                with lock:
                    current=labels();current['revision']+=1;current.setdefault('labels',{}).setdefault(image,{})[slot]=entry;current['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();temporary=labels_path.with_suffix('.tmp');temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(labels_path)
                self.send_json(dict(status='saved',revision=current['revision']))
            except (AssertionError,ValueError,KeyError,TypeError):self.send_json(dict(error='invalid_label'),400)
    return ThreadingHTTPServer(('127.0.0.1',0),Handler)

def run():
    token=secrets.token_urlsafe(24);server=make_server(PILOT,token);url=f'http://127.0.0.1:{server.server_port}/{token}/';dump(PILOT/'server_runtime.json',dict(pid=os.getpid(),url=url,bind='127.0.0.1',public_predictions_exposed=False));print(url,flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:
        assert validate_entry(dict(state='point',x=123,y=456))==dict(state='point',x=123.,y=456.)
        assert validate_entry(dict(state='unplaceable'))==dict(state='unplaceable',x=None,y=None)
        for bad in [dict(state='point',x=-1,y=100),dict(state='point',x=float('nan'),y=100),dict(state='guessed')]:
            try:validate_entry(bad)
            except (AssertionError,ValueError,TypeError):pass
            else:raise AssertionError('Bad label accepted')
        print('Passed5 label validation fixtures; no human label file created.')
    else:run()
