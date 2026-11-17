"""Change near-player size handling; preserve the accepted far-player paths."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import cv2

from adaptive_near import AdaptiveNearTracker,PARAMETERS,old
import run_temporal_experiment as temporal

ROOT=Path(__file__).resolve().parent


def baseline(tag):
    if tag in ('003','007','025'):
        base=ROOT.parent/'tracking_quality_20261102_stage8'
        return (base/'tracking_quality_20261102_stage4'/f'players_{tag}_temporal',
                base/'tracking_quality_20261102_stage7'/f'players_{tag}_global')
    return (ROOT.parent/'tracking_quality_20261102_stage4'/f'players_{tag}_temporal',
            ROOT.parent/'tracking_quality_20261102_stage7'/f'players_{tag}_global')


def write(path,rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag
    oldfolder,farfolder=baseline(tag)
    before=json.loads((oldfolder/'summary.json').read_text(encoding='utf-8'))
    interval=dict(tag=tag,first_frame=before['first_frame'],last_frame=before['last_frame'],frames=before['frames'])
    dest=ROOT/f'players_{tag}_temporal'
    assert not (dest/'summary.json').exists(),'Existing results retained; choose a new output folder'
    temporal.OUT=ROOT;temporal.TemporalTracker=AdaptiveNearTracker
    temporal.PARAMETERS=dict(old.PARAMETERS,near_size_adaptation=PARAMETERS)
    (ROOT/f'protocol_{tag}.json').write_text(json.dumps(dict(parameters=temporal.PARAMETERS,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'adaptive_near.py',ROOT/'run_adaptive_near.py',Path(old.__file__)]},
        far_path_source=str(farfolder/'far_pair_path.csv')),indent=2),encoding='utf-8')
    summary=temporal.process(interval,Path(before['input_csv']))
    allrows=old.a.read(dest/'selected_four.csv')
    near=[r for r in allrows if r['player_slot'].startswith('near_')]
    frozenfar=old.a.read(farfolder/'far_pair_path.csv')
    # Separate schemas; rendering uses unchanged far boxes directly.
    write(dest/'near_pair_path.csv',near)
    (dest/'frozen_far_path.csv').write_bytes((farfolder/'far_pair_path.csv').read_bytes())
    frames=defaultdict(list)
    for r in near+frozenfar:frames[int(r['source_frame'])].append(r)
    cap=cv2.VideoCapture(str(old.a.SOURCE/'data/raw/match1.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(1,interval['first_frame']);temp=dest/'combined_intermediate.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080));assert writer.isOpened()
    for frame in range(interval['first_frame'],interval['last_frame']+1):
        ok,im=cap.read();assert ok;im=cv2.resize(im,(1920,1080))
        for r in frames[frame]:
            i=old.a.SLOTS.index(r['player_slot']);box=[round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
            x1,y1,x2,y2=box;cv2.rectangle(im,(x1,y1),(x2,y2),old.a.COLORS[i],2)
            cv2.putText(im,f'{old.a.NAMES[i]} / raw {r["track_id"]}',(x1,max(22,y1-8)),0,.65,old.a.COLORS[i],2)
        cv2.rectangle(im,(0,1010),(1920,1080),(0,0,0),-1)
        cv2.putText(im,f'{tag} F{frame} | adaptive near-player size reference | far paths preserved',(15,1038),0,.62,(255,255,255),1)
        cv2.putText(im,'All boxes are detector observations. Missing positions remain empty; identity is provisional.',(15,1070),0,.59,(255,255,255),1)
        writer.write(im)
    writer.release();cap.release()
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-y','-i',str(temp),'-ss',str(interval['first_frame']/fps),
        '-t',str(interval['frames']/fps),'-i',str(old.a.SOURCE/'data/raw/match1.mp4'),'-map','0:v:0','-map','1:a:0?',
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',
        str(dest/'combined_review_h264.mp4')],check=True)
    previousnear=[r for r in old.a.read(oldfolder/'selected_four.csv') if r['player_slot'].startswith('near_')]
    keys=('source_frame','player_slot','track_id','x1','y1','x2','y2')
    prevset={tuple(r[k] for k in keys) for r in previousnear};newset={tuple(r[k] for k in keys) for r in near}
    report=dict(tag=tag,frames=interval['frames'],before_near_observations=len(previousnear),after_near_observations=len(near),
        old_near_rows_retained=len(prevset&newset),removed_near_rows=len(prevset-newset),added_near_rows=len(newset-prevset),
        near_counts={s:sum(r['player_slot']==s for r in near) for s in ('near_front','near_rear')},
        far_file_exactly_preserved=(dest/'frozen_far_path.csv').read_bytes()==(farfolder/'far_pair_path.csv').read_bytes(),
        warning='Availability and output changes, not identity accuracy. Same-match development experiment.')
    (dest/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
