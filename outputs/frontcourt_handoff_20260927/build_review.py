import csv, json
from pathlib import Path
from collections import Counter
import cv2

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def write(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

seg=ROOT/'data/segments'
if (OUT/'rally_label_review.csv').exists() or (OUT/'identity_review.csv').exists():
    raise SystemExit('Review labels already exist; refusing to overwrite annotations.')
windows=read(seg/'match1_rally_window_candidates.csv')
plays={r['source_window_id']:r for r in read(seg/'match1_reviewed_play_windows.csv')}
serves={r['window_id']:r for r in read(seg/'match1_serve_impact_review.csv')}
labels=[]
for w in windows:
    p=plays.get(w['window_id'],{});s=serves.get(w['window_id'],{})
    labels.append(dict(candidate_id=w['window_id'],video_id=w['video_id'],
        source_end_boundary_id=w['end_boundary_id'],legacy_window_status=w['window_status'],
        candidate_start_sec=w['source_start_timestamp_sec'],candidate_end_sec=w['source_end_timestamp_sec'],
        legacy_play_start_sec=p.get('source_start_timestamp_sec',''),
        legacy_start_method=p.get('start_establishment_method',''),
        reviewed_serve_impact_sec=s.get('human_valid_serve_impact_timestamp_sec','') if s.get('impact_review_status')=='impact_confirmed' else '',
        serve_review_status=s.get('impact_review_status',''),
        rally_id='',start_source_frame='',end_source_frame='',boundary_review_status='pending',
        winner_team_id='',winner_camera_side='',team_side_mapping_status='pending',
        winner_evidence_source_frame='',winner_evidence_type='',winner_review_status='unreviewed',
        reviewer='',review_note='',analysis_eligible='false'))
write('rally_label_review.csv',labels)
audit={'candidate_windows':len(windows),'legacy_window_status':dict(Counter(w['window_status'] for w in windows)),
       'play_windows':len(plays),'start_methods':dict(Counter(p['start_establishment_method'] for p in plays.values())),
       'serve_review_status':dict(Counter(s['impact_review_status'] for s in serves.values())),
       'serve_vs_legacy_start_differences':[{'id':r['candidate_id'],'legacy':r['legacy_play_start_sec'],'serve':r['reviewed_serve_impact_sec']} for r in labels if r['reviewed_serve_impact_sec'] and r['legacy_play_start_sec'] and abs(float(r['reviewed_serve_impact_sec'])-float(r['legacy_play_start_sec']))>0.001]}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
cap=cv2.VideoCapture(str(ROOT/'data/raw/match1.mp4'))
reviews=[]
for pilot in ('pilot_011','pilot_013'):
    d=ROOT/'analysis/person_tracker_comparison'/pilot
    link=json.loads((d/'bytetrack_four_player_reconnection.json').read_text())['accepted'][0]
    rows=read(d/'bytetrack.csv');a=link['last_observed_frame'];b=link['reappearance_frame']
    new=[r for r in rows if int(r['track_id'])==link['new_track_id']]
    end=max(int(r['source_frame']) for r in new)
    frames=sorted(set([a-12,a-4,a,*range(a+1,b),b,b+4,b+12,end]))
    tiles=[]
    # Inspect raw-video court crop with candidate boxes; all frames across the junction.
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read()
        if not ok: raise RuntimeError(idx)
        for r in rows:
            if int(r['source_frame'])!=idx or int(r['track_id']) not in (link['old_track_id'],link['new_track_id']):continue
            x1,y1,x2,y2=[round(float(r[k])) for k in ('x1','y1','x2','y2')]
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)
            cv2.putText(img,'ID'+r['track_id'],(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        crop=img[850:1400,900:2700]
        tile=cv2.resize(crop,(900,275));cv2.putText(tile,f'{pilot} F{idx}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        tiles.append(tile)
    import numpy as np
    while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
    sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
    cv2.imwrite(str(OUT/f'{pilot}_junction.jpg'),sheet)
    # Full new-ID duration at full rate, including 30 preceding frames.
    start=a-30;cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    writer=cv2.VideoWriter(str(OUT/f'{pilot}_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS),(1280,720))
    if not writer.isOpened():raise RuntimeError('writer')
    for idx in range(start,end+1):
        ok,img=cap.read()
        if not ok:raise RuntimeError(idx)
        for r in rows:
            if int(r['source_frame'])==idx and int(r['track_id']) in (link['old_track_id'],link['new_track_id']):
                x1,y1,x2,y2=[round(float(r[k])) for k in ('x1','y1','x2','y2')]
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),4)
        img=cv2.resize(img,(1280,720));cv2.putText(img,f'{pilot} F{idx} provisional identity',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2);writer.write(img)
    writer.release()
    reviews.append(dict(pilot=pilot,**link,new_id_end_frame=end,review_status='pending',reviewer='',evidence=f'{pilot}_junction.jpg;{pilot}_review.mp4',notes=''))
cap.release();write('identity_review.csv',reviews)
print(json.dumps(audit,ensure_ascii=False,indent=2))
