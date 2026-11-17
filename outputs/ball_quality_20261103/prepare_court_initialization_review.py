"""Render all selected initializations and the expanded calibration region."""
from pathlib import Path
import csv,json,hashlib
import cv2,numpy as np
from court_initialization_guard import eligible,court_point,PARAMETERS
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';TRIAL=BASE/'court_initialization_trial'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=load(TRIAL/'protocol.json');results=load(TRIAL/'run_result.json');h=np.array(load(Path(p['calibration']))['image_to_court_homography']);ih=np.linalg.inv(h);out=TRIAL/'initialization_image_review';out.mkdir();cap=cv2.VideoCapture(p['source_video']);cells=[];manifest=[]
    for plan,r in zip(p['cases'],results['cases']):
        tag=plan['tag'];baseline=r['modes']['baseline'];new=r['modes']['court_seed_only'];f=new['initialized_frame'] if new['initialized_frame'] is not None else plan['first'];rows=[x for x in read(Path(plan['raw_csv'])) if int(x['source_frame'])==f and int(x['track_id'])>=0 and float(x['confidence'])>=.25]
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for coords,color in [([(0,0),(1,0),(1,1),(0,1)],(255,255,255)), ([(PARAMETERS['court_x_min'],PARAMETERS['court_y_min']),(PARAMETERS['court_x_max'],PARAMETERS['court_y_min']),(PARAMETERS['court_x_max'],PARAMETERS['court_y_max']),(PARAMETERS['court_x_min'],PARAMETERS['court_y_max'])],(0,170,255))]:
            poly=cv2.perspectiveTransform(np.array([coords],np.float64),ih)[0];cv2.polylines(im,[np.round(poly).astype(np.int32)],True,color,5)
        seedids={str(x) for x in (new['seed_ids'] or {}).values()};oldids={str(x) for x in baseline['seed_ids'].values()};diagnosis=[]
        for row in rows:
            b=[round(float(row[k])) for k in ['x1','y1','x2','y2']];tid=row['track_id'];keep=eligible(row,h);color=(80,255,80) if tid in seedids else ((0,0,255) if tid in oldids else (150,150,150))
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,6 if tid in seedids|oldids else 2)
            if tid in seedids|oldids:
                xy=court_point(row,h);label=f'raw{tid} '+('SELECT' if tid in seedids else 'OLD')+f' court({xy[0]:.2f},{xy[1]:.2f})';cv2.putText(im,label,(b[0],max(40,b[1]-14)),0,1.,color,2,cv2.LINE_AA)
            diagnosis.append(dict(track_id=tid,court_candidate=court_point(row,h),eligible=keep,old_seed=tid in oldids,new_seed=tid in seedids))
        preview=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA);path=out/f'{tag}_F{f}.jpg';assert cv2.imwrite(str(path),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
        head=np.zeros((34,960,3),np.uint8);cv2.putText(head,f'{tag} F{f}: green=new, red=old-only; orange=expanded region',(10,25),0,.58,(255,255,255),1,cv2.LINE_AA);cells.append(np.vstack([head,cv2.resize(preview,(960,540))]));manifest.append(dict(tag=tag,source_frame=f,old_seed_frame=baseline['initialized_frame'],new_seed_frame=new['initialized_frame'],image_file=path.name,sha256=sha(path),candidates=diagnosis))
    cap.release()
    for page,start in enumerate(range(0,len(cells),6),1):
        group=cells[start:start+6]
        while len(group)%2:group.append(np.zeros_like(cells[0]))
        assert cv2.imwrite(str(out/f'page_{page:02d}.jpg'),np.vstack([np.hstack(group[i:i+2]) for i in range(0,len(group),2)]))
    (out/'manifest.json').write_text(json.dumps(dict(selection='All15 cases; no seed success selection.',geometry_transfer='Visual line overlay only; no independent calibration transfer accuracy measured.',images=manifest,script_sha256=sha(Path(__file__)),independent_identity_GT=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print('All15 initialization evidence images saved.')
if __name__=='__main__':main()
